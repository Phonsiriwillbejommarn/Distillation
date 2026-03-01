import os
from huggingface_hub import HfApi

# Define repositories and path
repo_id = "Phonsiri/Qwen2.5-3B-GRPO-Reasoning"
# You can change the 'checkpoint-10' string below to whatever checkpoint you want to push!
folder_path = "./grpo_output/checkpoint-10"

print(f"Uploading {folder_path} to Hugging Face Hub: {repo_id}...")

api = HfApi()
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
)

print(f"Upload Complete! Check it out at https://huggingface.co/{repo_id}/tree/main")
