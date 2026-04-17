import copy
import sys
from dataclasses import replace

import pytest
import torch
import transformers.utils as transformers_utils
from transformers import (
    DeepseekV2Config,
    Ernie4_5_MoeConfig,
    Glm4MoeConfig,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
from transformers.utils import TransformersKwargs

from reap.args import DatasetArgs, LayerwiseArgs, ObserverArgs, PruneArgs
from reap.layerwise_prune import record_activations_layerwise
from reap.main import _setup_observer
from reap.model_util import MODEL_ATTRS, get_moe
from reap.prune import prune


def _install_local_model_import_shims():
    import transformers.models.deepseek_v2.configuration_deepseek_v2 as deepseek_config
    import transformers.models.ernie4_5_moe.configuration_ernie4_5_moe as ernie_config

    sys.modules.setdefault("reap.models.configuration_deepseek", deepseek_config)
    sys.modules.setdefault("reap.models.configuration_ernie4_5_moe", ernie_config)
    if not hasattr(transformers_utils, "LossKwargs"):
        transformers_utils.LossKwargs = TransformersKwargs


_install_local_model_import_shims()

from reap.models.glm.modeling_glm4_moe import Glm4MoeForCausalLM
from reap.models.modeling_deepseek import DeepseekV2ForCausalLM
from reap.models.modeling_ernie4_5_moe import Ernie4_5_MoeForCausalLM


def _make_mock_batches(*, include_use_cache: bool = False):
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
    ]
    if include_use_cache:
        for batch in batches:
            batch["use_cache"] = False
    return batches


def _make_qwen3_model():
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=3,
            num_experts_per_tok=1,
            norm_topk_prob=False,
        )
    )
    model.eval()
    return model


def _make_glm_model():
    model = Glm4MoeForCausalLM(
        Glm4MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_routed_experts=3,
            num_experts_per_tok=1,
            norm_topk_prob=False,
        )
    )
    model.eval()
    return model


def _make_deepseek_model():
    model = DeepseekV2ForCausalLM(
        DeepseekV2Config(
            vocab_size=32,
            pad_token_id=0,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_routed_experts=3,
            num_experts_per_tok=1,
            n_shared_experts=None,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            scoring_func="softmax",
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
            n_group=1,
            topk_group=1,
            use_cache=False,
            norm_topk_prob=False,
        )
    )
    model.eval()
    return model


def _make_ernie_model():
    model = Ernie4_5_MoeForCausalLM(
        Ernie4_5_MoeConfig(
            vocab_size=32,
            pad_token_id=0,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            moe_num_experts=3,
            moe_k=1,
            use_moe=True,
            sinkhorn_2gate=False,
            sinkhorn_temp=1.0,
            moe_gate_act="softmax",
            moe_use_aux_free=True,
            hidden_dropout_prob=0.0,
            num_nextn_predict_layers=0,
            weight_share_add_bias=False,
            multi_token_pred_lambda=0.0,
            moe_capacity=[3, 3, 3],
            moe_layer_start_index=1,
            moe_layer_end_index=2,
            moe_layer_interval=1,
            output_router_logits=False,
        )
    )
    model.eval()
    return model


def _main_observer_data(model, batches, obs_args, results_dir):
    observer = _setup_observer(model, obs_args)
    try:
        for batch in batches:
            attention_mask = batch.get("attention_mask")
            with observer.set_attention_mask(attention_mask):
                model(**batch)
        observer.save_state(results_dir / "main" / obs_args.output_file_name)
        return observer.report_state()
    finally:
        observer.close_hooks()


def _layerwise_observer_data(model, batches, obs_args, results_dir):
    return record_activations_layerwise(
        model=model,
        tokenizer=None,
        data_batches=batches,
        ds_args=DatasetArgs(dataset_name="mock", split="train"),
        obs_args=obs_args,
        layerwise_args=LayerwiseArgs(batch_group_size=1, save_intermediate=False),
        results_dir=results_dir,
    )


def _moe_layer_indices(observer_data):
    return sorted(int(layer_idx) for layer_idx in observer_data)


def _expert_count_for_layer(model, layer_idx):
    moe = get_moe(model, layer_idx)
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    experts = getattr(moe, model_attrs["experts"])
    if isinstance(experts, torch.nn.ModuleList):
        return len(experts)
    return moe.num_experts


def _router_out_features_for_layer(model, layer_idx):
    moe = get_moe(model, layer_idx)
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    router = getattr(moe, model_attrs["router"])
    return router.out_features


def _assert_prune_result(observer_data, pruned_model, n_experts_to_prune):
    layer_indices = _moe_layer_indices(observer_data)
    assert layer_indices, "Expected at least one MoE layer to be observed"

    for layer_idx in layer_indices:
        original_num_experts = observer_data[layer_idx]["expert_frequency"].shape[0]
        expected_num_experts = original_num_experts - n_experts_to_prune
        assert _expert_count_for_layer(pruned_model, layer_idx) == expected_num_experts
        assert _router_out_features_for_layer(pruned_model, layer_idx) == expected_num_experts


def _run_prune(observer_data, model, tmp_path, subdir_name):
    pruned_model_dir = tmp_path / subdir_name
    prune_args = PruneArgs(prune_method="frequency", n_experts_to_prune=1)
    prune(
        observer_data=observer_data,
        model=model,
        prune_args=prune_args,
        n_experts_to_prune=prune_args.n_experts_to_prune,
        pruned_model_dir=pruned_model_dir,
    )
    return pruned_model_dir, prune_args.n_experts_to_prune


ARCHITECTURE_CASES = [
    pytest.param(
        _make_qwen3_model,
        _make_mock_batches,
        id="qwen3",
    ),
    pytest.param(
        _make_glm_model,
        _make_mock_batches,
        id="glm-local",
    ),
    pytest.param(
        _make_deepseek_model,
        lambda: _make_mock_batches(include_use_cache=True),
        id="deepseek-local",
    ),
    pytest.param(
        _make_ernie_model,
        _make_mock_batches,
        id="ernie-local",
    ),
]


@pytest.mark.parametrize("model_factory,batch_factory", ARCHITECTURE_CASES)
def test_e2e_layerwise_pruning_functionality(model_factory, batch_factory, tmp_path):
    torch.manual_seed(0)

    base_model = model_factory()
    main_model = copy.deepcopy(base_model)
    layerwise_model = copy.deepcopy(base_model)
    main_prune_model = copy.deepcopy(base_model)
    layerwise_prune_model = copy.deepcopy(base_model)

    batches = batch_factory()
    obs_args = ObserverArgs(
        output_file_name="mock_observations.pt",
        record_pruning_metrics_only=True,
        renormalize_router_weights=False,
    )

    main_results_dir = tmp_path / "main_results"
    layerwise_results_dir = tmp_path / "layerwise_results"

    main_observer_data = _main_observer_data(
        model=main_model,
        batches=batches,
        obs_args=obs_args,
        results_dir=main_results_dir,
    )
    layerwise_observer_data = _layerwise_observer_data(
        model=layerwise_model,
        batches=batches,
        obs_args=replace(obs_args),
        results_dir=layerwise_results_dir,
    )

    assert _moe_layer_indices(main_observer_data) == _moe_layer_indices(
        layerwise_observer_data
    )

    _, n_experts_to_prune = _run_prune(
        observer_data=main_observer_data,
        model=main_prune_model,
        tmp_path=tmp_path,
        subdir_name="main_pruned",
    )
    _assert_prune_result(main_observer_data, main_prune_model, n_experts_to_prune)

    _, n_experts_to_prune = _run_prune(
        observer_data=layerwise_observer_data,
        model=layerwise_prune_model,
        tmp_path=tmp_path,
        subdir_name="layerwise_pruned",
    )
    _assert_prune_result(
        layerwise_observer_data, layerwise_prune_model, n_experts_to_prune
    )
