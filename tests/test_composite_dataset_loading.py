import pytest

from reap.data import load_category_batches, parse_composite_dataset_spec


REAL_COMPOSITE_SPEC = (
    "theblackcat102/evol-codealpaca-v1:8,"
    "Salesforce/xlam-function-calling-60k:8,"
    "open-r1/Mixture-of-Thoughts[code]:8,"
    "open-r1/Mixture-of-Thoughts[math]:8,"
    "open-r1/Mixture-of-Thoughts[science]:8,"
    "SWE-bench/SWE-smith-trajectories(tool):8"
)


def test_composite_dataset_loading_with_real_hf_datasets():
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
    except Exception as exc:
        pytest.skip(f"Required tokenizer could not be loaded: {exc}")

    composite_components = parse_composite_dataset_spec(
        REAL_COMPOSITE_SPEC,
        default_split="train",
    )
    assert composite_components is not None
    assert [
        (component.name, component.subset, component.split, component.num_samples)
        for component in composite_components
    ] == [
        ("theblackcat102/evol-codealpaca-v1", None, "train", 8),
        ("Salesforce/xlam-function-calling-60k", None, "train", 8),
        ("open-r1/Mixture-of-Thoughts", "code", "train", 8),
        ("open-r1/Mixture-of-Thoughts", "math", "train", 8),
        ("open-r1/Mixture-of-Thoughts", "science", "train", 8),
        ("SWE-bench/SWE-smith-trajectories", None, "tool", 8),
    ]

    combined_batches = []
    for component in composite_components:
        try:
            component_batches = load_category_batches(
                dataset_name=component.name,
                split=component.split,
                subset=component.subset,
                tokenizer=tokenizer,
                model_max_length=2048,
                split_by_category=False,
                return_vllm_tokens_prompt=False,
                truncate=True,
                samples_per_category=1,
                batch_size=2,
            )
        except Exception as exc:
            pytest.skip(
                f"Real dataset component {component.name} could not be loaded with "
                f"subset={component.subset} split={component.split}: {exc}"
            )

        assert list(component_batches.keys()) == ["all"]
        assert len(component_batches["all"]) == 1
        sample_batch = component_batches["all"][0]
        assert "input_ids" in sample_batch
        assert "attention_mask" in sample_batch

        sample_token_ids = sample_batch["input_ids"][0][
            sample_batch["attention_mask"][0].bool()
        ]
        first_token_ids = sample_token_ids[:128].tolist()
        last_token_ids = sample_token_ids[-128:].tolist()
        component_label = component.name
        if component.subset is not None:
            component_label += f"[{component.subset}]"
        if component.split is not None:
            component_label += f"({component.split})"
        print(f"\n=== {component_label} ===")
        print("***" * 20 + "\nfirst 128 decoded tokens:\n")
        print(tokenizer.decode(first_token_ids, skip_special_tokens=False))
        print("***" * 20 + "\nlast 128 decoded tokens:\n")
        print(tokenizer.decode(last_token_ids, skip_special_tokens=False))

        combined_batches.extend(component_batches["all"])

    assert len(combined_batches) == len(composite_components)
