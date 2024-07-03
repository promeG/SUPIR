import os
import requests
import boto3

def download_file(url, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path):
        print(f"File already exists: {file_path}")
    else:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"File successfully downloaded and saved: {file_path}")
        else:
            print(f"Error downloading the file. Status code: {response.status_code}")

# Prepare webui
from modules.launch_utils import prepare_environment
prepare_environment()

# Checkpoints
download_file(
    "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/40b5c7a3a390ab6f81d03386412be66cee22e134/open_clip_pytorch_model.bin?download=true",
    "models/ai/",
    "open_clip_pytorch_model.bin"
)
download_file(
    "https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0F.ckpt?download=true",
    "models/ai/",
    "v0F.ckpt"
)

# Upscaler Model
download_file(
    "https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0Q.ckpt?download=true",
    "models/ai/",
    "v0Q.ckpt"
)

download_file(
    "https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/resolve/main/Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors?download=true",
    "models/checkpoints/",
    "juggernautXL_v9-Lightning_4S_V9_+_RDPhoto_2.safetensors"
)

download_file(
    "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/caef85acb9a97535a15a0e7ebff0b601a081dd9b/config.json?download=true",
    "models/llava-v1.5-7b/",
    "config.json"
)

download_file(
    "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/caef85acb9a97535a15a0e7ebff0b601a081dd9b/generation_config.json?download=true",
    "models/llava-v1.5-7b/",
    "generation_config.json"
)

download_file(
    "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/caef85acb9a97535a15a0e7ebff0b601a081dd9b/pytorch_model-00001-of-00002.bin?download=true",
    "models/llava-v1.5-7b/",
    "pytorch_model-00001-of-00002.bin"
)

download_file(
    "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/caef85acb9a97535a15a0e7ebff0b601a081dd9b/pytorch_model-00002-of-00002.bin?download=true",
    "models/llava-v1.5-7b/",
    "pytorch_model-00002-of-00002.bin"
)

download_file(
    "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/caef85acb9a97535a15a0e7ebff0b601a081dd9b/pytorch_model.bin.index.json?download=true",
    "models/llava-v1.5-7b/",
    "pytorch_model.bin.index.json"
)

download_file(
    "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/caef85acb9a97535a15a0e7ebff0b601a081dd9b/special_tokens_map.json?download=true",
    "models/llava-v1.5-7b/",
    "special_tokens_map.json"
)

download_file(
    "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/caef85acb9a97535a15a0e7ebff0b601a081dd9b/tokenizer.model?download=true",
    "models/llava-v1.5-7b/",
    "tokenizer.model"
)

download_file(
    "https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/caef85acb9a97535a15a0e7ebff0b601a081dd9b/tokenizer_config.json?download=true",
    "models/llava-v1.5-7b/",
    "tokenizer_config.json"
)
