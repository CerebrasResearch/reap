import os
import shutil
from huggingface_hub import snapshot_download, hf_hub_download

def main():
    # Locate this script’s directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths to local new files
    new_modeling = os.path.normpath(
        os.path.join(script_dir, os.pardir, "src", "reap", "models", "glm", "modeling_glm4_moe.py")
    )
    new_config = os.path.normpath(
        os.path.join(script_dir, os.pardir, "src", "reap", "models", "glm", "config.json")
    )

    # Download the full model into artifacts/models/<model_name>
    model_name = "GLM-4.5-Air"
    artifacts_dir = os.path.normpath(
        os.path.join(script_dir, os.pardir, "artifacts", "models")
    )
    model_dir = os.path.join(artifacts_dir, model_name)

    snapshot_download(
        repo_id="zai-org/GLM-4.5-Air",
        repo_type="model",
        revision="e7fdb9e0a52d2e0aefea94f5867c924a32a78d17",
        local_dir=model_dir,
    )

    # Patch files in the downloaded model directory
    shutil.copy2(new_modeling, os.path.join(model_dir, "modeling_glm4_moe.py"))
    shutil.copy2(new_config, os.path.join(model_dir, "config.json"))

    print(f"Replaced modeling and tokenizer in {model_dir}")

if __name__ == "__main__":
    main()
