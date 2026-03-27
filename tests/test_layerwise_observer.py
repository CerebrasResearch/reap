import copy

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.layerwise_observer import LayerwiseMoEObserver, LayerwiseMoEObserverConfig
from reap.observer import MoETransformerObserver, Qwen3MoEObserverHookConfig


def _make_qwen3_moe_model(num_hidden_layers=1):
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=False,
    )
    model = Qwen3MoeForCausalLM(config)
    model.eval()
    return model


def _make_layerwise_config():
    base_config = Qwen3MoEObserverHookConfig(record_pruning_metrics_only=True)
    layerwise_config = LayerwiseMoEObserverConfig(
        num_experts_attr_name=base_config.num_experts_attr_name,
        top_k_attr_name=base_config.top_k_attr_name,
        fused_experts=base_config.fused_experts,
        renormalize_router_weights=base_config.renormalize_router_weights,
        record_pruning_metrics_only=True,
    )
    layerwise_config.module_class_name_to_hook_regex = (
        base_config.module_class_name_to_hook_regex
    )
    return layerwise_config


def _assert_layerwise_states_match(actual_state, expected_state):
    assert actual_state.keys() == expected_state.keys()

    metrics_to_compare = [
        "total_tokens",
        "expert_frequency",
        "pairwise_expert_frequency",
        "weighted_expert_frequency_sum",
        "ean_sum",
        "weighted_ean_sum",
        "ean_mean",
        "reap",
    ]

    for layer_idx in actual_state:
        for metric in metrics_to_compare:
            actual_value = actual_state[layer_idx][metric]
            expected_value = expected_state[layer_idx][metric]

            if actual_value.is_floating_point():
                assert torch.allclose(
                    actual_value, expected_value, rtol=1e-5, atol=1e-6
                )
            else:
                assert torch.equal(actual_value, expected_value)


def test_layerwise_observer_matches_standard_observer_for_batched_input():
    torch.manual_seed(0)

    model = _make_qwen3_moe_model()
    layerwise_model = copy.deepcopy(model)

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
    }

    observer = MoETransformerObserver(
        model,
        hook_config=Qwen3MoEObserverHookConfig(record_pruning_metrics_only=True),
    )
    with observer.set_attention_mask(batch["attention_mask"]):
        _ = model(**batch)
    standard_state = observer.report_state()
    observer.close_hooks()

    layerwise_observer = LayerwiseMoEObserver(
        layerwise_model,
        hook_config=_make_layerwise_config(),
    )
    layerwise_state = layerwise_observer.collect_all_blocks([batch])
    layerwise_observer.close_hooks()

    expected_tokens = batch["attention_mask"].sum()
    assert layerwise_state[0]["total_tokens"] == expected_tokens
    assert standard_state[0]["total_tokens"] == expected_tokens

    assert torch.equal(
        layerwise_state[0]["expert_frequency"], standard_state[0]["expert_frequency"]
    )
    assert torch.equal(
        layerwise_state[0]["pairwise_expert_frequency"],
        standard_state[0]["pairwise_expert_frequency"],
    )
    assert torch.allclose(
        layerwise_state[0]["weighted_expert_frequency_sum"],
        standard_state[0]["weighted_expert_frequency_sum"],
        atol=1e-6,
    )
    assert torch.allclose(
        layerwise_state[0]["ean_sum"],
        standard_state[0]["ean_sum"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.allclose(
        layerwise_state[0]["weighted_ean_sum"],
        standard_state[0]["weighted_ean_sum"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.allclose(
        layerwise_state[0]["ean_mean"],
        standard_state[0]["ean_mean"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.allclose(
        layerwise_state[0]["reap"],
        standard_state[0]["reap"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_layerwise_observer_grouped_batches_match_single_pass():
    torch.manual_seed(0)

    model = _make_qwen3_moe_model(num_hidden_layers=2)
    grouped_model = copy.deepcopy(model)

    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long
            ),
        },
        {
            "input_ids": torch.tensor([[6, 7, 8, 9], [10, 11, 12, 0]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long
            ),
        },
        {
            "input_ids": torch.tensor([[13, 14, 0, 0], [15, 16, 17, 18]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long
            ),
        },
    ]

    single_pass_observer = LayerwiseMoEObserver(
        model,
        hook_config=_make_layerwise_config(),
    )
    single_pass_state = single_pass_observer.collect_all_blocks(batches)
    single_pass_observer.close_hooks()

    grouped_observer = LayerwiseMoEObserver(
        grouped_model,
        hook_config=_make_layerwise_config(),
    )
    grouped_state = grouped_observer.collect_all_blocks(batches, batch_group_size=1)
    grouped_observer.close_hooks()

    _assert_layerwise_states_match(grouped_state, single_pass_state)
