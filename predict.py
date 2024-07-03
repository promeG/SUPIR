
from modules import timer
from modules import initialize_util
from modules import initialize
from urllib.parse import urlparse
from fastapi import FastAPI
from io import BytesIO
import yaml
from PIL import Image
import io
from PIL.ExifTags import TAGS
from sd_parsers import ParserManager
import subprocess
import CKPT_PTH

import os, json
import numpy as np
import requests
import base64
import uuid
import time
import cv2

from cog import BasePredictor, Input, Path

Image.MAX_IMAGE_PIXELS = None

parser_manager = ParserManager()

LLAVA_URL = "https://weights.replicate.delivery/default/llava-v1.5-13b.tar"
LLAVA_CLIP_URL = (
    "https://weights.replicate.delivery/default/clip-vit-large-patch14-336.tar"
)
SDXL_CLIP1_URL = "https://weights.replicate.delivery/default/clip-vit-large-patch14.tar"
SDXL_CLIP2_URL = (
    "https://weights.replicate.delivery/default/CLIP-ViT-bigG-14-laion2B-39B-b160k.tar"
)
LLAVA_CLIP_PATH = CKPT_PTH.LLAVA_CLIP_PATH
SDXL_CLIP1_PATH = CKPT_PTH.SDXL_CLIP1_PATH
LLAVA_MODEL_PATH = CKPT_PTH.LLAVA_MODEL_PATH
SDXL_CLIP2_CACHE = f"{MODEL_CACHE}/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k"
SDXL_CLIP2_CACHE2 = f"{MODEL_CACHE}/CLIP-ViT-bigG-14-laion2B-39B-b160k"



def check_service_availability(client):
    while True:
        try:
            result = client.predict(api_name="/do_nothing")
            print(result)
            print(
                ">>>>>>>>>>***********>>>>>>>>>>>>>++++++++++++++___________Service is available."
            )
            time.sleep(1)
            break
        except httpx.ConnectError:
            print("Connection error. Retrying in 1 seconds...")
            time.sleep(1)

def download_weights(url, dest, extract=True):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    args = ["pget"]
    if extract:
        args.append("-x")
    subprocess.check_call(args + [url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:

        if not os.path.exists("models/ai/"):
            os.makedirs("models/ai/")
        # if not os.path.exists(LLAVA_MODEL_PATH):
            # download_weights(LLAVA_URL, LLAVA_MODEL_PATH)
        if not os.path.exists(LLAVA_CLIP_PATH):
            download_weights(LLAVA_CLIP_URL, LLAVA_CLIP_PATH)
        if not os.path.exists(SDXL_CLIP1_PATH):
            download_weights(SDXL_CLIP1_URL, SDXL_CLIP1_PATH)
        if not os.path.exists(SDXL_CKPT):
            download_weights(SDXL_URL, SDXL_CKPT, extract=False)
        if not os.path.exists(SDXL_CLIP2_CACHE):
            download_weights(SDXL_CLIP2_URL, SDXL_CLIP2_CACHE)

        if not os.path.isfile("models/open_clip_pytorch_model.bin"):
            subprocess.run(["ln", "-s", "models/ai/open_clip_pytorch_model.bin", "models/open_clip_pytorch_model.bin"])

        if not os.path.isfile("models/v0Q.ckpt"):
            subprocess.run(["ln", "-s", "models/ai/v0Q.ckpt", "models/v0Q.ckpt"])

        if not os.path.isfile("models/v0F.ckpt"):
            subprocess.run(["ln", "-s", "models/ai/v0F.ckpt", "models/v0F.ckpt"])
            
        subprocess.Popen(["python","gradio_demo.py","--loading_half_params", "--use_tile_vae", "--outputs_folder", "out/"])
        time.sleep(15)
        self.client = Client("http://127.0.0.1:7860/")

    def remove_metadata(self, input, output):
        image = Image.open(input)

        # next 3 lines strip exif
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)

        image_without_exif.save(output)

        # as a good practice, close the file handler after saving the image.
        image_without_exif.close()

    def predict(
        self,
        image: Path = Input(description="input image"),
        prompt: str = Input(description="Prompt", default="masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>"),
        # negative_prompt: str = Input(description="Negative Prompt", default="(worst quality, low quality, normal quality:2) JuggernautNegative-neg"),
        scale_factor: float = Input(
            description="Scale factor", default=2
        ),
        creativity: float = Input(
            description="Creativity, try from 0.3 - 0.9", ge=0, le=1, default=0.35
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=1337
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        print("Running prediction")
        start_time = time.time()

        # 输入图片
        input_image_path = image
        original_image = Image.open(input_image_path)
        origin_width, origin_height = original_image.size
        original_image.close()
        # 放大倍数，支持2x 4x 8x
        scale_factor = int(scale_factor)
        target_width = origin_width * scale_factor
        target_height = origin_height * scale_factor
        fixed_scale_factor = scale_factor
        if max(target_height, target_width) > 4000:
            # 超出目前supir最大允许范围，后处理
            fixed_scale_factor = 4000.0 / float(max(origin_width, origin_height))

        while True:
            try:
                result =  self.client.predict(
                    prompt,
                    # "Vivid, expansive detailing, enhanced natural textures, rich color depth, dynamic range optimization, realistic lighting and atmospheric effects, clear horizon lines, intricate detailing in flora and fauna, immersive depth of field, panoramic integrity preserved, true-to-life scale and perspective.",
                    # str  in 'Default Positive Prompt' Textbox component
                    "bf16",  # Literal['fp32', 'bf16']  in 'Auto-Encoder Data Type' Radio component
                    False,  # bool  in 'BG restoration' Checkbox component
                    False,  # bool  in 'Face restoration' Checkbox component
                    False,  # bool  in 'Apply LLaVa' Checkbox component
                    True,  # bool  in 'Apply SUPIR' Checkbox component
                    False,  # bool  in 'Auto Unload LLaVA (Low VRAM)' Checkbox component
                    "",  # str  in 'Batch Input Folder - Can use image captions from .txt' Textbox component
                    "juggernautXL_v9-Lightning_4S_V9_+_RDPhoto_2.safetensors",
                    # Literal['juggernautXL_v9-Lightning_4S_V9_+_RDPhoto_2.safetensors']  in 'Model' Dropdown component
                    "Wavelet",  # Literal['None', 'AdaIn', 'Wavelet']  in 'Color-Fix Type' Radio component
                    "bf16",  # Literal['fp32', 'fp16', 'bf16']  in 'Diffusion Data Type' Radio component
                    10,  # float (numeric value between 1 and 200) in 'Steps' Slider component
                    "",  # str  in 'Face Prompt' Textbox component
                    256,  # float (numeric value between 256 and 2048) in 'Text Guidance Scale' Slider component  todo
                    True,  # bool  in 'Linear CFG' Checkbox component
                    False,  # bool  in 'Linear Stage2 Guidance' Checkbox component
                    "",  # str  in 'Prompt' Textbox component
                    False,  # bool  in 'Generate Comparison Video' Checkbox component
                    "v0-Q",  # Literal['v0-Q', 'v0-F']  in 'Model Selection' Radio component
                    "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, low-res, over-smooth",
                    # str  in 'Default Negative Prompt' Textbox component
                    1,  # float (numeric value between 1 and 200) in 'Number Of Images To Generate' Slider component
                    1,  # float (numeric value between 1 and 4) in 'Batch Size' Slider component
                    "mp4",  # Literal['mp4', 'mkv']  in 'Video Format' Dropdown component
                    0.1,  # float (numeric value between 0.1 and 1.0) in 'Output Video Quality' Slider component
                    "otp",  # str  in 'Batch Output Path - Leave empty to save to default.' Textbox component
                    "Describe this image and its style in a very detailed manner. The image is a realistic photography, not an art painting.",
                    # str  in 'LLaVA prompt' Textbox component
                    True,  # bool  in 'Randomize Seed' Checkbox component
                    2,  # float (numeric value between 1.0 and 15.0) in 'Text Guidance Scale' Slider component
                    5,  # float (numeric value between 0 and 40) in 'S-Churn' Slider component
                    creativity,  # float (numeric value between 1.0 and 1.1) in 'S-Noise' Slider component
                    -1,  # float (numeric value between -1.0 and 6.0) in 'Stage1 Guidance Strength' Slider component
                    1,  # float (numeric value between 0.0 and 1.0) in 'Stage2 Guidance Strength' Slider component
                    "DPMPP2M",  # Literal['EDM', 'DPMPP2M']  in 'Sampler' Dropdown component
                    True,  # bool  in 'Save Captions' Checkbox component
                    -1,  # float (numeric value between -1 and 2147483647) in 'Seed' Slider component
                    2,  # float (numeric value between 1.0 and 9.0) in 'CFG Start' Slider component
                    0,  # float (numeric value between 0 and 1) in 'Guidance Start' Slider component
                    input_image_path,
                    # todo filepath  in 'Input' File component
                    0.2,  # float (numeric value between 0.0 and 1.0) in 'Temperature' Slider component
                    0.7,  # float (numeric value between 0.0 and 1.0) in 'Top P' Slider component
                    fixed_scale_factor,
                    "5",  # str  in 'Duration' Textbox component
                    3,  # float  in 'End Time' Number component
                    "30",  # str  in 'FPS' Textbox component
                    "1080",  # str  in 'Height' Textbox component
                    3,  # float  in 'Start Time' Number component
                    "1920",  # str  in 'Width' Textbox component
                    api_name="/start_single_process",
                )
                print(result)
                break
            except httpx.ConnectError as e:
                print("call_api failed: ", e.reason)
                time.sleep(1)

        print(result_img_path)

        print(f"Prediction took {round(time.time() - start_time,2)} seconds")
        return Path(result_img_path)
