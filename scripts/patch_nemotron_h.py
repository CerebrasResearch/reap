"""Snapshot Nemotron-3 / Nemotron-H locally and overwrite modeling_nemotron_h.py
with the REAP-patched copy.

Mirrors `scripts/patch_deepseek.py` and `scripts/patch_ernie4_5.py`. The patched
file:

  * exposes ``router_logits`` from ``NemotronHMOE.forward`` so the standard
    ``MoETransformerObserver`` works without changes,
  * provides a pure-torch fallback for ``mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn``
    so the model loads on CPU and AMD ROCm hosts,
  * skips ``torch.cuda.stream`` when the input tensor is not on CUDA.

Usage:

    python scripts/patch_nemotron_h.py
"""
import os
import shutil

from huggingface_hub import snapshot_download


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    new_file = os.path.normpath(
        os.path.join(
            script_dir, os.pardir, "src", "reap", "models", "modeling_nemotron_h.py"
        )
    )
    model_name = "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    artifacts_dir = os.path.normpath(
        os.path.join(script_dir, os.pardir, "artifacts", "models")
    )
    model_dir = os.path.join(artifacts_dir, model_name)
    snapshot_download(
        repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        repo_type="model",
        local_dir=model_dir,
    )
    cached_file = os.path.join(model_dir, "modeling_nemotron_h.py")
    shutil.copy2(new_file, cached_file)
    print(f"Replaced {cached_file} with {new_file}")


if __name__ == "__main__":
    main()
