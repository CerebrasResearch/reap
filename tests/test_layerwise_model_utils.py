import pytest
from transformers import (
    DeepseekV2Config,
    DeepseekV2ForCausalLM,
    Ernie4_5_MoeConfig,
    Ernie4_5_MoeForCausalLM,
    Glm4MoeConfig,
    Glm4MoeForCausalLM,
    Llama4ForCausalLM,
    Llama4TextConfig,
    MixtralConfig,
    MixtralForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from reap.layerwise_observer import LayerwiseMoEObserver
from reap.layerwise_model_utils import extract_model_components, is_decoder_block
from reap.observer import (
    DeepSeekMoEObserverHookConfig,
    Ernie4_5MoEObserverHookConfig,
    Glm44MoEObserverHookConfig,
    Llama4MoEObserverHookConfig,
    MixtralMoEObserverHookConfig,
    Qwen3MoEObserverHookConfig,
)


EXPECTED_BLOCK_NAMES = ["model.layers.0", "model.layers.1", "model.layers.2"]
EXPECTED_OUTSIDE_LAYER_MODULES = ["model.embed_tokens", "model.norm", "lm_head"]


def _make_qwen3_moe_model():
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=8,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_experts=2,
            num_experts_per_tok=1,
            norm_topk_prob=False,
        )
    )
    model.eval()
    return model


def _make_llama4_model():
    model = Llama4ForCausalLM(
        Llama4TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            intermediate_size_mlp=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=1,
            num_local_experts=2,
            num_experts_per_tok=1,
            layer_types=["full_attention", "full_attention", "full_attention"],
        )
    )
    model.eval()
    return model


def _make_mixtral_model():
    model = MixtralForCausalLM(
        MixtralConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_local_experts=2,
            num_experts_per_tok=1,
        )
    )
    model.eval()
    return model


def _make_deepseek_v2_model():
    model = DeepseekV2ForCausalLM(
        DeepseekV2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=1,
            num_key_value_heads=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_shared_experts=None,
        )
    )
    model.eval()
    return model


def _make_ernie4_5_moe_model():
    model = Ernie4_5_MoeForCausalLM(
        Ernie4_5_MoeConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=1,
            num_key_value_heads=1,
            moe_num_experts=2,
            moe_k=1,
        )
    )
    model.eval()
    return model


def _make_glm4_moe_model():
    model = Glm4MoeForCausalLM(
        Glm4MoeConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=1,
            num_key_value_heads=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
        )
    )
    model.eval()
    return model


MODEL_FACTORIES = [
    pytest.param(_make_qwen3_moe_model, id="Qwen3MoeForCausalLM"),
    pytest.param(_make_llama4_model, id="Llama4ForCausalLM"),
    pytest.param(_make_mixtral_model, id="MixtralForCausalLM"),
    pytest.param(_make_deepseek_v2_model, id="DeepseekV2ForCausalLM"),
    pytest.param(_make_ernie4_5_moe_model, id="Ernie4_5_MoEForCausalLM"),
    pytest.param(_make_glm4_moe_model, id="Glm4MoeForCausalLM"),
]


def _make_qwen3_hook_config():
    return Qwen3MoEObserverHookConfig()


def _make_llama4_hook_config():
    return Llama4MoEObserverHookConfig()


def _make_mixtral_hook_config():
    return MixtralMoEObserverHookConfig()


def _make_deepseek_hook_config():
    return DeepSeekMoEObserverHookConfig()


def _make_ernie4_5_hook_config():
    return Ernie4_5MoEObserverHookConfig(
        module_class_name_to_hook_regex="Ernie4_5_MoeSparseMoeBlock",
        num_experts_attr_name="num_experts",
        top_k_attr_name="top_k",
    )


def _make_glm4_hook_config():
    return Glm44MoEObserverHookConfig()


MOE_LOOKUP_CASES = [
    pytest.param(
        _make_qwen3_moe_model,
        _make_qwen3_hook_config,
        [0, 1, 2],
        "mlp",
        id="Qwen3MoeForCausalLM",
    ),
    pytest.param(
        _make_llama4_model,
        _make_llama4_hook_config,
        [0, 1, 2],
        "feed_forward",
        id="Llama4ForCausalLM",
    ),
    pytest.param(
        _make_mixtral_model,
        _make_mixtral_hook_config,
        [0, 1, 2],
        "block_sparse_moe",
        id="MixtralForCausalLM",
    ),
    pytest.param(
        _make_deepseek_v2_model,
        _make_deepseek_hook_config,
        [0, 1, 2],
        "mlp",
        id="DeepseekV2ForCausalLM",
    ),
    pytest.param(
        _make_ernie4_5_moe_model,
        _make_ernie4_5_hook_config,
        [1, 2],
        "mlp",
        id="Ernie4_5_MoEForCausalLM",
    ),
    pytest.param(
        _make_glm4_moe_model,
        _make_glm4_hook_config,
        [1, 2],
        "mlp",
        id="Glm4MoeForCausalLM",
    ),
]


def _detected_decoder_block_names(model):
    return [
        name
        for name, module in model.named_modules()
        if is_decoder_block(name, module)
    ]


def _model_layer_block_names(model):
    return [f"model.layers.{index}" for index in range(len(model.model.layers))]


def _relative_module_name(block, target_module):
    for name, module in block.named_modules():
        if module is target_module:
            return name
    return None


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_is_decoder_block_detects_only_transformer_layers(model_factory):
    model = model_factory()

    assert _model_layer_block_names(model) == EXPECTED_BLOCK_NAMES
    assert _detected_decoder_block_names(model) == EXPECTED_BLOCK_NAMES


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_extract_model_components_returns_layers_container_and_outside_modules(
    model_factory,
):
    model = model_factory()

    block_names = _model_layer_block_names(model)
    layers, outside_layer_modules = extract_model_components(model, block_names)

    assert layers is model.model.layers
    assert len(layers) == len(EXPECTED_BLOCK_NAMES)
    assert outside_layer_modules == EXPECTED_OUTSIDE_LAYER_MODULES


@pytest.mark.parametrize(
    "model_factory,hook_config_factory,expected_moe_block_indices,expected_name",
    MOE_LOOKUP_CASES,
)
def test_find_moe_module_in_block_matches_expected_modules_for_moe_layers(
    model_factory,
    hook_config_factory,
    expected_moe_block_indices,
    expected_name,
):
    model = model_factory()
    observer = LayerwiseMoEObserver(
        model,
        hook_config=hook_config_factory(),
        block_names=_model_layer_block_names(model),
    )

    try:
        for block_idx in expected_moe_block_indices:
            moe_module = observer._find_moe_module_in_block(block_idx)

            assert moe_module is not None
            assert (
                moe_module.__class__.__name__
                == observer.hook_config.module_class_name_to_hook_regex
            )
            assert (
                _relative_module_name(observer.layers[block_idx], moe_module)
                == expected_name
            )
    finally:
        observer.close_hooks()
