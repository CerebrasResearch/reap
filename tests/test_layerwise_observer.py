import copy

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.layerwise_observer import LayerwiseMoEObserver, LayerwiseMoEObserverConfig
from reap.observer import MoETransformerObserver, Qwen3MoEObserverHookConfig


def _make_qwen3_moe_model():
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=1,
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
    observer.set_attention_mask(batch["attention_mask"])
    _ = model(**batch)
    observer.clear_attention_mask()
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
