import argparse
import copy
import datetime
import gc
import os
import time
import traceback
from datetime import datetime
from typing import Tuple, List, Any, Dict

import einops
import gradio as gr
import numpy as np
import torch
from PIL import Image
from PIL import PngImagePlugin
from gradio_imageslider import ImageSlider
from omegaconf import OmegaConf

from SUPIR.util import HWC3, upscale_image, fix_resize, convert_dtype
from SUPIR.util import create_SUPIR_model
from SUPIR.utils.compare import create_comparison_video
from SUPIR.utils.face_restoration_helper import FaceRestoreHelper
from SUPIR.utils.model_fetch import get_model
from SUPIR.utils.status_container import StatusContainer
from llava.llava_agent import LLavaAgent

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default='127.0.0.1')
parser.add_argument("--share", type=str, default=False)
parser.add_argument("--port", type=int)
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--use_image_slider", action='store_true', default=False)
parser.add_argument("--log_history", action='store_true', default=False)
parser.add_argument("--loading_half_params", action='store_true', default=True)
parser.add_argument("--use_tile_vae", action='store_true', default=True)
parser.add_argument("--encoder_tile_size", type=int, default=512)
parser.add_argument("--decoder_tile_size", type=int, default=64)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
parser.add_argument("--ckpt", type=str, default='models/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors')
parser.add_argument("--theme", type=str, default='default')
parser.add_argument("--open_browser", action='store_true', default=True)
parser.add_argument("--outputs_folder")
parser.add_argument("--debug")
args = parser.parse_args()
server_ip = args.ip
use_llava = not args.no_llava
if(args.debug):
    args.open_browser=False

if torch.cuda.device_count() >= 2:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

face_helper = None
model = None
llava_agent = None
models_loaded = False

status_container = StatusContainer()


def load_face_helper():
    global face_helper
    if face_helper is None:
        face_helper = FaceRestoreHelper(
            device='cpu',
            upscale_factor=1,
            face_size=1024,
            use_parse=True,
            det_model='retinaface_resnet50'
        )


def load_model(selected_model, progress=None):
    global model
    if model is None:
        if progress is not None:
            progress(1 / 2, desc="Loading SUPIR...")
        model = create_SUPIR_model('options/SUPIR_v0.yaml', supir_sign='Q', device='cpu', ckpt=args.ckpt)
        if args.loading_half_params:
            model = model.half()
        if args.use_tile_vae:
            model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
        model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(model.first_stage_model.denoise_encoder)
        model.current_model = 'v0-Q'

    if selected_model != model.current_model:
        config = OmegaConf.load('options/SUPIR_v0_tiled.yaml')
        device = 'cpu'
        if model_select == 'v0-Q':
            print('load v0-Q')
            if progress is not None:
                progress(1 / 2, desc="Updating SUPIR checkpoint...")
            ckpt_q = torch.load(config.SUPIR_CKPT_Q, map_location=device)
            model.load_state_dict(ckpt_q, strict=False)
            model.current_model = 'v0-Q'
        elif model_select == 'v0-F':
            print('load v0-F')
            if progress is not None:
                progress(1 / 2, desc="Updating SUPIR checkpoint...")
            ckpt_f = torch.load(config.SUPIR_CKPT_F, map_location=device)
            model.load_state_dict(ckpt_f, strict=False)
            model.current_model = 'v0-F'
    if progress is not None:
        progress(2 / 2, desc="SUPIR loaded.")


def load_llava():
    global llava_agent
    if llava_agent is None and use_llava:
        llava_path = get_model('liuhaotian/llava-v1.5-7b')
        llava_agent = LLavaAgent(llava_path, device='cuda', load_8bit=args.load_8bit_llava,
                                 load_4bit=False)


def all_to_cpu():
    global face_helper, model, llava_agent
    if face_helper is not None:
        face_helper = face_helper.to('cpu')
    if model is not None:
        model = model.to('cpu')
    if llava_agent is not None:
        llava_agent = llava_agent.to('cpu')


# This could probably be renamed and used to move devices to cpu as well...buuut...
def to_gpu(elem_to_load, device):
    if elem_to_load is not None:
        elem_to_load = elem_to_load.to(device)
        torch.cuda.set_device(device)
    return elem_to_load


def update_target_resolution(img, do_upscale):
    # Read the input image dimensions
    if img is None:
        return ""
    with Image.open(img) as img:
        width, height = img.size
        width_org, height_org = img.size

    # Apply the do_upscale ratio
    width *= do_upscale
    height *= do_upscale

    # Ensure both dimensions are at least 1024 pixels
    if min(width, height) < 1024:
        do_upscale_factor = 1024 / min(width, height)
        width *= do_upscale_factor
        height *= do_upscale_factor

    # Update the target resolution label
    return f"Input: {int(width_org)}x{int(height_org)} px, {width_org * height_org / 1e6:.2f} Megapixels / Estimated Output Resolution: {int(width)}x{int(height)} px, {width * height / 1e6:.2f} Megapixels"


def read_image_metadata(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        return "File does not exist."

    # Get the last modified date and format it
    last_modified_timestamp = os.path.getmtime(image_path)
    last_modified_date = datetime.fromtimestamp(last_modified_timestamp).strftime('%d %B %Y, %H:%M %p - UTC')

    # Open the image and extract metadata
    with Image.open(image_path) as img:
        width, height = img.size
        megapixels = (width * height) / 1e6

        metadata_str = f"Last Modified Date: {last_modified_date}\nMegapixels: {megapixels:.2f}\n"

        # Extract metadata based on image format
        if img.format == 'JPEG':
            exif_data = img._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = Image.ExifTags.TAGS.get(tag, tag)
                    metadata_str += f"{tag_name}: {value}\n"
        else:
            metadata = img.info
            if metadata:
                for key, value in metadata.items():
                    metadata_str += f"{key}: {value}\n"
            else:
                metadata_str += "No additional metadata found."

    return metadata_str


# prompt, stage_1_output_image, result_gallery, result_slider, event_id, fb_score, fb_text, seed, face_gallery, comparison_video
def update_elements(status_label):
    print(f"Label changed: {status_label}")
    prompt_el = gr.update()
    stage_1_output_image_el = gr.update()
    result_gallery_el = gr.update()
    result_slider_el = gr.update()
    event_id_el = gr.update()
    fb_score_el = gr.update()
    fb_text_el = gr.update()
    seed_el = gr.update()
    face_gallery_el = gr.update()
    comparison_video_el = gr.update()

    if "Processing Complete" in status_label:
        print(status_label)
        if "LLaVA" in status_label:
            status_container.llava_caption = status_container.llava_captions[0]
            prompt_el = gr.update(value=status_container.llava_caption)
            print(f"LLaVA caption: {status_container.llava_caption}")
            result_gallery_el = gr.update(visible=False)
        elif "Stage 1" in status_label:
            print("Updating stage 1 output image")
            # Get the first value from status_container.image_data dict
            out_image = list(status_container.image_data.values())[0]
            stage_1_output_image_el = gr.update(value=out_image)
            result_gallery_el = gr.update(visible=False)
        elif "Stage 2" in status_label:
            print("Updating stage 2 output image")
            result_slider_el = gr.update(value=status_container.result_gallery, visible=True)
            result_gallery_el = gr.update(visible=False)
            event_id_el = gr.update(value=status_container.event_id)
            fb_score_el = gr.update(value=status_container.fb_score)
            fb_text_el = gr.update(value=status_container.fb_text)
            seed_el = gr.update(value=status_container.seed)
            face_gallery_el = gr.update(value=status_container.face_gallery)
            comparison_video_el = gr.update(value=status_container.comparison_video)
        elif "Batch" in status_label:
            print("Updating batch outputs")
            result_gallery_el = gr.update(value=status_container.result_gallery, visible=True)
            result_slider_el = gr.update(visible=False)
            event_id_el = gr.update(value=status_container.event_id)
            fb_score_el = gr.update(value=status_container.fb_score)
            fb_text_el = gr.update(value=status_container.fb_text)
            seed_el = gr.update(value=status_container.seed)
            face_gallery_el = gr.update(value=status_container.face_gallery)
            comparison_video_el = gr.update(value=status_container.comparison_video)
    return (prompt_el, stage_1_output_image_el, result_gallery_el, result_slider_el, event_id_el, fb_score_el,
            fb_text_el, seed_el, face_gallery_el, comparison_video_el)


batch_processing_val = False


def llava_process_single(image, temp, p, question=None, unload=True, progress=gr.Progress()):
    global status_container
    status_container = StatusContainer()
    with Image.open(image) as img:
        image_data = np.array(img)
        input_data = {image: image_data}
    return llava_process(input_data, temp, p, question, unload, progress)


def llava_process(inputs: Dict[str, List[np.ndarray[Any, np.dtype]]], temp, p, question=None, unload=True,
                  progress=None):
    global llava_agent, status_container
    output_captions = []
    status_container.llava_captions = []
    if use_llava:
        total_steps = len(inputs.keys()) + (2 if unload else 1)
        step = 1
        if progress is not None:
            progress(step / total_steps, desc="Loading LLaVA...")
        load_llava()
        llava_agent = to_gpu(llava_agent, LLaVA_device)
        if progress is not None:
            progress(step / total_steps, desc="LLaVA loaded, captioning images...")
        for img_path, img in inputs.items():
            if progress is not None:
                progress(step / total_steps, desc=f"Processing image {step}/{len(inputs)} with LLaVA...")
            lq = HWC3(img)
            lq = Image.fromarray(lq.astype('uint8'))
            captions = llava_agent.gen_image_caption([lq], temperature=temp, top_p=p, qs=question)
            output_captions.append(captions[0])
            status_container.llava_caption = captions[0]
            step += 1
            if not batch_processing_val:  # Check if batch processing has been stopped
                break
        if progress is not None and unload:
            progress(step / total_steps, desc="Unloading LLaVA...")
            llava_agent = llava_agent.to('cpu')
            step += 1
            progress(step / total_steps, desc="LLaVA processing complete.")
        status_container.llava_captions = output_captions
        return f"LLaVA Processing Complete: {len(inputs)} images processed"
    else:
        status_container.llava_caption = ""
        return "LLaVA is not available."


def stage1_process_single(image, gamma, unload=True, progress=gr.Progress()):
    global status_container
    status_container = StatusContainer()
    with Image.open(image) as img:
        image_data = np.array(img)
        input_data = {image: image_data}
    return stage1_process(input_data, gamma, unload, progress)


def stage1_process(inputs: Dict[str, List[np.ndarray[Any, np.dtype]]], gamma, unload=True, progress=None) -> str:
    global model
    global status_container
    output_data = {}
    total_steps = len(inputs.keys()) + (1 if unload else 0)
    step = 0

    load_model(model_select, progress)
    model = to_gpu(model, SUPIR_device)

    for image_path, img in inputs.items():
        step += 1
        if progress is not None:
            progress(step / total_steps, desc=f"Processing image {step}/{len(inputs)}...")
        lq = HWC3(img)
        lq = fix_resize(lq, 512)
        # stage1
        lq = np.array(lq) / 255 * 2 - 1
        lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
        lq = model.batchify_denoise(lq, is_stage1=True)
        lq = (lq[0].permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
        # gamma correction
        lq = lq / 255.0
        lq = np.power(lq, gamma)
        lq *= 255.0
        lq = lq.round().clip(0, 255).astype(np.uint8)
        status_container.stage_1_output_image = lq
        output_data[image_path] = lq
        if not batch_processing_val:  # Check if batch processing has been stopped
            break
    if unload:
        step += 1
        if progress is not None:
            progress(step / total_steps, desc="Unloading models...")
        all_to_cpu()
    status_container.image_data = output_data
    return f"Stage 1 Processing Complete: processed {len(inputs)} images"


def stage2_process_single(image, p, ap, n_p, ns, us, edms, sstage1, sstage2, scfg, sseed, schurn, snoise, cfix_type,
                          ddtype, aedtype, g_correction, l_cfg, ls_stage2, slinear_cfg, slinear_stage2, modelselect,
                          n_images, r_seed, a_stage_1, f_resolution, a_bg, a_face, f_prompt, make_video, v_duration,
                          v_fps, v_width, v_height):
    global status_container
    status_container = StatusContainer()
    with Image.open(image) as img:
        image_data = np.array(img)
        input_data = {image: image_data}
    captions = [p]
    return stage2_process(input_data, captions, ap, n_p, ns, us, edms, sstage1, sstage2, scfg, sseed, schurn, snoise,
                          cfix_type,
                          ddtype, aedtype, g_correction, l_cfg, ls_stage2, slinear_cfg, slinear_stage2, modelselect,
                          n_images, r_seed, a_stage_1, f_resolution, a_bg, a_face, f_prompt, make_video, v_duration,
                          v_fps, v_width, v_height, progress=gr.Progress())


def stage2_process(inputs: Dict[str, List[np.ndarray[Any, np.dtype]]], captions, a_prompt, n_prompt, num_samples,
                   upscale, edm_steps,
                   s_stage1, s_stage2, s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype,
                   gamma_correction, linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select,
                   num_images, random_seed, apply_stage_1, face_resolution, apply_bg, apply_face, face_prompt,
                   make_comparison_video, video_duration, video_fps, video_width, video_height,
                   dont_update_progress=False, out_folder="outputs", batch_process_folder="", unload=True,
                   progress=gr.Progress()):
    global model, status_container, event_id

    load_model(model_select, progress)
    to_gpu(model, SUPIR_device)
    model.ae_dtype = convert_dtype(ae_dtype)
    model.model.dtype = convert_dtype(diff_dtype)

    if len(out_folder) < 1:
        out_folder = "outputs"
    if args.outputs_folder:
        out_folder = args.outputs_folder
    if len(batch_process_folder) > 1:
        out_folder = batch_process_folder
    output_dir = os.path.join(out_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata_dir = os.path.join(output_dir, "images_meta_data")
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    compare_videos_dir = os.path.join(output_dir, "compare_videos")
    if not os.path.exists(compare_videos_dir):
        os.makedirs(compare_videos_dir)

    idx = 0

    output_data = {}

    for image_path, img in inputs.items():
        output_data[image_path] = []
        all_results = []
        img_prompt = captions[idx]
        event_id = str(time.time_ns())
        event_dict = {'event_id': event_id, 'localtime': time.ctime(), 'prompt': img_prompt, 'base_model': args.ckpt,
                      'a_prompt': a_prompt,
                      'n_prompt': n_prompt, 'num_samples': num_samples, 'upscale': upscale, 'edm_steps': edm_steps,
                      's_stage1': s_stage1, 's_stage2': s_stage2, 's_cfg': s_cfg, 'seed': seed, 's_churn': s_churn,
                      's_noise': s_noise, 'color_fix_type': color_fix_type, 'diff_dtype': diff_dtype,
                      'ae_dtype': ae_dtype,
                      'gamma_correction': gamma_correction, 'linear_CFG': linear_CFG,
                      'linear_s_stage2': linear_s_stage2,
                      'spt_linear_CFG': spt_linear_CFG, 'spt_linear_s_stage2': spt_linear_s_stage2,
                      'model_select': model_select, 'apply_stage_1': apply_stage_1, 'face_resolution': face_resolution,
                      'apply_bg': apply_bg, 'face_prompt': face_prompt}

        img = HWC3(img)
        img = upscale_image(img, upscale, unit_resolution=32, min_size=1024)
        lq = np.array(img)
        lq = lq / 255 * 2 - 1
        lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

        counter = 1
        _faces = []
        if not dont_update_progress and progress is not None:
            progress(0 / num_images, desc="Generating images")
        video_path = None

        # Only load face model if face restoration is enabled
        bg_caption = img_prompt
        face_captions = img_prompt

        if apply_face:
            load_face_helper()
            if face_helper is None or not isinstance(face_helper, FaceRestoreHelper):
                raise ValueError('Face helper not loaded')
            face_helper.clean_all()
            face_helper.read_image(lq)
            # get face landmarks for each face
            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

            lq = lq / 255 * 2 - 1
            lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

            if len(face_prompt) > 1:
                face_captions = face_prompt
            to_gpu(face_helper, SUPIR_device)

        for _ in range(num_images):
            if random_seed or num_images > 1:
                seed = np.random.randint(0, 2147483647)
            start_time = time.time()  # Track the start time

            if apply_face:
                faces = []
                for face in face_helper.cropped_faces:
                    _faces.append(face)
                    face = np.array(face) / 255 * 2 - 1
                    face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3,
                           :,
                           :]
                    faces.append(face)

                for face, caption in zip(faces, face_captions):
                    caption = [caption]

                    from torch.nn.functional import interpolate
                    face = interpolate(face, size=face_resolution, mode='bilinear', align_corners=False)
                    if face_resolution < 1024:
                        face = torch.nn.functional.pad(face, (512 - face_resolution // 2, 512 - face_resolution // 2,
                                                              512 - face_resolution // 2, 512 - face_resolution // 2),
                                                       'constant', 0)

                    samples = model.batchify_sample(face, caption, num_steps=edm_steps, restoration_scale=s_stage1,
                                                    s_churn=s_churn,
                                                    s_noise=s_noise, cfg_scale=s_cfg, control_scale=s_stage2, seed=seed,
                                                    num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                                                    color_fix_type=color_fix_type,
                                                    use_linear_cfg=linear_CFG, use_linear_control_scale=linear_s_stage2,
                                                    cfg_scale_start=spt_linear_CFG,
                                                    control_scale_start=spt_linear_s_stage2)
                    if face_resolution < 1024:
                        samples = samples[:, :, 512 - face_resolution // 2:512 + face_resolution // 2,
                                  512 - face_resolution // 2:512 + face_resolution // 2]
                    samples = interpolate(samples, size=face_helper.face_size, mode='bilinear', align_corners=False)
                    x_samples = (
                            einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(
                        0, 255).astype(np.uint8)

                    face_helper.add_restored_face(x_samples[0])
                    _faces.append(x_samples[0])
                    # img_before_face_apply = Image.fromarray(x_samples[0])
                    # img_before_face_apply.save("applied_face_1.png", "PNG")

                if apply_bg:
                    caption = [img_prompt]
                    samples = model.batchify_sample(lq, caption, num_steps=edm_steps, restoration_scale=s_stage1,
                                                    s_churn=s_churn,
                                                    s_noise=s_noise, cfg_scale=s_cfg, control_scale=s_stage2, seed=seed,
                                                    num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                                                    color_fix_type=color_fix_type,
                                                    use_linear_cfg=linear_CFG, use_linear_control_scale=linear_s_stage2,
                                                    cfg_scale_start=spt_linear_CFG,
                                                    control_scale_start=spt_linear_s_stage2)
                else:
                    samples = lq
                _bg = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(
                    0, 255).astype(np.uint8)
                # img_before_face_apply = Image.fromarray(_bg[0])
                # img_before_face_apply.save("before_face_apply.png", "PNG")
                face_helper.get_inverse_affine(None)
                results = [face_helper.paste_faces_to_input_image(upsample_img=_bg[0])]
                # img_before_face_apply = Image.fromarray(results[0])
                # img_before_face_apply.save("after_face_apply.png", "PNG")
                # img_before_face_apply = Image.fromarray(_faces[0])
                # img_before_face_apply.save("applied_face.png", "PNG")
            else:
                caption = [img_prompt]
                samples = model.batchify_sample(lq, caption, num_steps=edm_steps, restoration_scale=s_stage1,
                                                s_churn=s_churn,
                                                s_noise=s_noise, cfg_scale=s_cfg, control_scale=s_stage2, seed=seed,
                                                num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                                                color_fix_type=color_fix_type,
                                                use_linear_cfg=linear_CFG, use_linear_control_scale=linear_s_stage2,
                                                cfg_scale_start=spt_linear_CFG, control_scale_start=spt_linear_s_stage2)
                x_samples = (
                        einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(
                    0, 255).astype(np.uint8)
                results = [x_samples[i] for i in range(num_samples)]

                image_generation_time = time.time() - start_time
                desc = f"Generated image {counter}/{num_images} in {image_generation_time:.2f} seconds"
                counter += 1
                if not dont_update_progress and progress is not None:
                    progress(counter / num_images, desc=desc)
                print(desc)  # Print the progress
                start_time = time.time()  # Reset the start time for the next image

            for i, result in enumerate(results):
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                if len(base_filename) > 250:
                    base_filename = base_filename[:250]
                save_path = os.path.join(output_dir, f'{base_filename}.png')
                video_path = os.path.join(compare_videos_dir, f'{base_filename}.mp4')
                index = 1
                while os.path.exists(save_path):
                    save_path = os.path.join(output_dir, f'{base_filename}_{str(index).zfill(4)}.png')
                    video_path = os.path.join(compare_videos_dir, f'{base_filename}_{str(index).zfill(4)}.mp4')
                    index += 1

                # Embed metadata into the image
                img = Image.fromarray(result)
                meta = PngImagePlugin.PngInfo()
                for key, value in event_dict.items():
                    meta.add_text(key, str(value))
                img.save(save_path, "PNG", pnginfo=meta)
                metadata_path = os.path.join(metadata_dir,
                                             f'{os.path.splitext(os.path.basename(save_path))[0]}.txt')
                with open(metadata_path, 'w') as f:
                    for key, value in event_dict.items():
                        f.write(f'{key}: {value}\n')
                video_path = os.path.abspath(video_path)
                if not make_comparison_video:
                    video_path = None
                if make_comparison_video:
                    full_save_image_path = os.path.abspath(save_path)
                    create_comparison_video(image_path, full_save_image_path, video_path, video_duration, video_fps,
                                            video_width, video_height)

                all_results.extend(results)
        if len(inputs.keys) == 1:
            # Prepend the first input image to all_results
            all_results.insert(0, list(inputs.values())[0])
        output_data[image_path] = all_results
        status_container.prompt = img_prompt
        status_container.result_gallery = all_results
        status_container.event_id = event_id
        status_container.seed = seed
        status_container.face_gallery = _faces
        status_container.comparison_video = video_path

        if args.log_history:
            os.makedirs(f'./history/{event_id[:5]}/{event_id[5:]}', exist_ok=True)
            with open(f'./history/{event_id[:5]}/{event_id[5:]}/logs.txt', 'w') as f:
                f.write(str(event_dict))
            f.close()
            Image.fromarray(img).save(f'./history/{event_id[:5]}/{event_id[5:]}/LQ.png')
            for i, result in enumerate(all_results):
                Image.fromarray(result).save(f'./history/{event_id[:5]}/{event_id[5:]}/HQ_{i}.png')
        if not batch_processing_val:  # Check if batch processing has been stopped
            break

    status_container.result_gallery = output_data
    if not batch_processing_val or unload:
        all_to_cpu()

    return f"Stage 2 Processing Complete: Processed {num_images} images."


def batch_upscale(batch_process_folder, outputs_folder, main_prompt, a_prompt, n_prompt, num_samples, upscale,
                  edm_steps, s_stage1, s_stage2, s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype,
                  gamma_correction, linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select,
                  num_images, random_seed, apply_stage_1, face_resolution, apply_bg, apply_face, face_prompt,
                  batch_process_llava, temperature, top_p, qs, make_comparison_video, video_duration, video_fps,
                  video_width, video_height, progress=gr.Progress()):
    
    global batch_processing_val, llava_agent
    batch_processing_val = True
    # Get the list of image files in the folder
    image_files = [file for file in os.listdir(batch_process_folder) if
                   file.lower().endswith((".png", ".jpg", ".jpeg"))]

    total_images = len(image_files)

    # Make a dictionary to store the image data and path
    img_data = {}
    for file in image_files:
        img = Image.open(os.path.join(batch_process_folder, file))
        img_data[file] = np.array(img)

    # Store it globally
    status_container.image_data = img_data

    # Create an array of captions
    if batch_process_llava:
        print('Processing LLaVA')
        llava_process(img_data, temperature, top_p, qs, unload=True, progress=progress)
        captions = status_container.llava_captions
    else:
        captions = [main_prompt] * total_images

    if not batch_processing_val:
        return "Batch Processing Complete: Cancelled"
    
    if apply_stage_1:
        print("Processing images (Stage 1)")
        stage1_process(img_data, gamma_correction, unload=True, progress=progress)
    
    if not batch_processing_val:
        return "Batch Processing Complete: Cancelled"
    
    print("Processing images (Stage 2)")
    stage2_process(img_data, captions, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2, s_cfg,
                   seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction, linear_CFG,
                   linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select, num_images, random_seed,
                   apply_stage_1, face_resolution, apply_bg, apply_face, face_prompt, make_comparison_video,
                   video_duration, video_fps, video_width, video_height, dont_update_progress=True,
                   out_folder=outputs_folder, batch_process_folder=batch_process_folder, unload=True, progress=progress)
    
    batch_processing_val = False
    return f"Batch Processing Complete: processed {num_images} images"


def stop_batch_upscale(progress=gr.Progress()):
    global batch_processing_val
    progress(1, f"Stop command giving please wait to stop")
    print('\n***Stop command giving please wait to stop***\n')
    batch_processing_val = False


def load_and_reset(param_setting):
    e_steps = 50
    sstage2 = 1.0
    sstage1 = -1.0
    schurn = 5
    snoise = 1.003
    ap = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - ' \
               'realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore ' \
               'detailing, hyper sharpness, perfect without deformations.'
    np = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, ' \
               '3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, ' \
               'signature, jpeg artifacts, deformed, lowres, over-smooth'
    cfix_type = 'Wavelet'
    l_s_stage2 = 0.0
    l_s_s_stage2 = False
    l_cfg = True
    if param_setting == "Quality":
        s_cfg = 7.5
        spt_linear_CFG = 4.0
    elif param_setting == "Fidelity":
        s_cfg = 4.0
        spt_linear_CFG = 1.0
    else:
        raise NotImplementedError
    return e_steps, s_cfg, sstage2, sstage1, schurn, snoise, ap, np, cfix_type, l_cfg, l_s_stage2, spt_linear_CFG, l_s_s_stage2


def submit_feedback(evt_id, f_score, f_text):
    if args.log_history:
        with open(f'./history/{evt_id[:5]}/{evt_id[5:]}/logs.txt', 'r') as f:
            event_dict = eval(f.read())
        f.close()
        event_dict['feedback'] = {'score': f_score, 'text': f_text}
        with open(f'./history/{evt_id[:5]}/{evt_id[5:]}/logs.txt', 'w') as f:
            f.write(str(event_dict))
        f.close()
        return 'Submit successfully, thank you for your comments!'
    else:
        return 'Submit failed, the server is not set to log history.'


def toggle_compare_elements(enable: bool) -> Tuple[gr.update, gr.update]:
    return gr.update(visible=enable), gr.update(visible=enable)


title_md = """
# **SUPIR: Practicing Model Scaling for Photo-Realistic Image Restoration**

1 Click Installer (auto download models as well) : https://www.patreon.com/posts/99176057

FFmpeg Install Tutorial : https://youtu.be/-NjNy7afOQ0 &emsp; [[Paper](https://arxiv.org/abs/2401.13627)] &emsp; [[Project Page](http://supir.xpixel.group/)] &emsp; [[How to play](https://github.com/Fanghua-Yu/SUPIR/blob/master/assets/DemoGuide.png)]
"""

claim_md = """
## **Terms of use**

By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research. Please submit a feedback to us if you get any inappropriate answer! We will collect those to keep improving our models. For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.

## **License**
While the original readme for the project *says* it's non-commercial, it was *actually* released under the MIT license. That means that the project can be used for whatever you want.

And yes, it would certainly be nice if anything anybody stuck in a random readme were the ultimate gospel when it comes to licensing, unfortunately, that's just
not how the world works. MIT license means FREE FOR ANY PURPOSE, PERIOD.
The service is a research preview ~~intended for non-commercial use only~~, subject to the model [License](https://github.com/Fanghua-Yu/SUPIR#MIT-1-ov-file) of SUPIR.
"""
css_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'css', 'style.css'))
with open(css_file, 'r') as f:
    css = f.read()
    f.close()

block = gr.Blocks(title='SUPIR', theme=args.theme, css=css).queue()
with block:
    with gr.Tab("Main Upscale"):
        # Execution buttons
        with gr.Row():
            llava_button = gr.Button(value="LlaVa Run")
            stage_1_button = gr.Button(value="Stage1 Run")
            stage_2_button = gr.Button(value="Stage2 Run")
            start_batch_button = gr.Button(value="Start Batch")
            stop_batch_button = gr.Button(value="Cancel Batch")
        with gr.Row():
            output_label = gr.Label(label="Progress", elem_classes=["progress_label"])
        with gr.Row(equal_height=True):
            with gr.Column() as input_col:
                input_image = gr.Image(type="filepath", elem_id="image-input", label="Input Image",
                                       elem_classes=["preview_box"], height=300, sources=["upload"])
            with gr.Column(visible=False) as stage_1_out_col:
                stage_1_output_image = gr.Image(type="numpy", elem_id="image-s1", label="Stage1 Output",
                                                elem_classes=["preview_box"], height=300, interactive=False)
            with gr.Column(visible=False) as comparison_video_col:
                comparison_video = gr.Video(label="Comparison Video", elem_classes=["preview_box"], height=300)
            with gr.Column() as result_col:
                result_gallery = gr.Gallery(label='Output', elem_id="gallery1", elem_classes=["preview_box"], height=300, visible=False)
                result_slider = ImageSlider(label='Output', interactive=False, show_download_button=True,
                                                 elem_id="gallery1", elem_classes=["preview_box"], height=300)
        with gr.Row():
            with gr.Column():
                with gr.Accordion("General options", open=True):
                    upscale = gr.Slider(label="Upscale Size (Stage 2)", minimum=1, maximum=8, value=1, step=0.1)
                    prompt = gr.Textbox(label="Prompt", value="")
                    face_prompt = gr.Textbox(label="Face Prompt",
                                             placeholder="Optional, uses main prompt if not provided",
                                             value="")
                    target_res = gr.Textbox(label="Input / Output Resolution", value="", interactive=False)

                with gr.Accordion("Stage1 options", open=False):
                    gamma_correction = gr.Slider(label="Gamma Correction", minimum=0.1, maximum=2.0, value=1.0,
                                                 step=0.1)
                with gr.Accordion("Stage2 options", open=False):
                    with gr.Row():
                        with gr.Column():
                            num_images = gr.Slider(label="Number Of Images To Generate", minimum=1, maximum=200
                                                   , value=1, step=1)
                            num_samples = gr.Slider(label="Batch Size", minimum=1,
                                                    maximum=4 if not args.use_image_slider else 1
                                                    , value=1, step=1)
                        with gr.Column():
                            random_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    with gr.Row():
                        edm_steps = gr.Slider(label="Steps", minimum=20, maximum=200, value=50, step=1)
                        s_cfg = gr.Slider(label="Text Guidance Scale", minimum=1.0, maximum=15.0, value=7.5, step=0.1)
                        s_stage2 = gr.Slider(label="Stage2 Guidance Strength", minimum=0., maximum=1., value=1.,
                                             step=0.05)
                        s_stage1 = gr.Slider(label="Stage1 Guidance Strength", minimum=-1.0, maximum=6.0, value=-1.0,
                                             step=1.0)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        s_churn = gr.Slider(label="S-Churn", minimum=0, maximum=40, value=5, step=1)
                        s_noise = gr.Slider(label="S-Noise", minimum=1.0, maximum=1.1, value=1.003, step=0.001)
                    with gr.Row():
                        a_prompt = gr.Textbox(label="Default Positive Prompt",
                                              value='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R '
                                                    'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                                                    'Grading, ultra HD, extreme meticulous detailing, skin pore detailing, '
                                                    'hyper sharpness, perfect without deformations.')
                        n_prompt = gr.Textbox(label="Default Negative Prompt",
                                              value='painting, oil painting, illustration, drawing, art, sketch, oil painting, '
                                                    'cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                                                    'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                                                    'deformed, lowres, over-smooth')
                with gr.Accordion("LLaVA options", open=False):
                    temperature = gr.Slider(label="Temperature", minimum=0., maximum=1.0, value=0.2, step=0.1)
                    top_p = gr.Slider(label="Top P", minimum=0., maximum=1.0, value=0.7, step=0.1)
                    qs = gr.Textbox(label="Question",
                                    value="Describe this image and its style in a very detailed manner. "
                                          "The image is a realistic photography, not an art painting.")

            with gr.Column():
                with gr.Accordion("Batch options", open=True):
                    with gr.Row():
                        with gr.Column():
                            batch_process_folder = gr.Textbox(
                                label="Batch Input Folder - Can use image captions from .txt",
                                placeholder="R:\SUPIR video\comparison_images")
                            outputs_folder = gr.Textbox(
                                label="Batch Output Path - Leave empty to save to default.",
                                placeholder="R:\SUPIR video\comparison_images\outputs")
                    with gr.Row():
                        with gr.Column():
                            apply_stage_1 = gr.Checkbox(label="Apply Stage 1 Before Stage 2", value=False)
                            batch_process_llava = gr.Checkbox(label="Batch Process LLaVA", value=False)

                with gr.Accordion("Advanced options", open=False):
                    with gr.Row():
                        with gr.Column():
                            param_setting = gr.Dropdown(["Quality", "Fidelity"], interactive=True,
                                                        label="Param Setting",
                                                        value="Quality")
                        with gr.Column():
                            reset_button = gr.Button(value="Reset Param", scale=2)
                    with gr.Row():
                        with gr.Column():
                            linear_CFG = gr.Checkbox(label="Linear CFG", value=True)
                            spt_linear_CFG = gr.Slider(label="CFG Start", minimum=1.0,
                                                       maximum=9.0, value=4.0, step=0.5)
                        with gr.Column():
                            linear_s_stage2 = gr.Checkbox(label="Linear Stage2 Guidance", value=False)
                            spt_linear_s_stage2 = gr.Slider(label="Guidance Start", minimum=0.,
                                                            maximum=1., value=0., step=0.05)
                    with gr.Row():
                        with gr.Column():
                            diff_dtype = gr.Radio(['fp32', 'fp16', 'bf16'], label="Diffusion Data Type", value="fp16",
                                                  interactive=True)
                        with gr.Column():
                            ae_dtype = gr.Radio(['fp32', 'bf16'], label="Auto-Encoder Data Type", value="bf16",
                                                interactive=True)
                        with gr.Column():
                            color_fix_type = gr.Radio(["None", "AdaIn", "Wavelet"], label="Color-Fix Type",
                                                      value="Wavelet",
                                                      interactive=True)
                        with gr.Column():
                            model_select = gr.Radio(["v0-Q", "v0-F"], label="Model Selection", value="v0-Q",
                                                    interactive=True)
                with gr.Accordion("Face options", open=False):
                    face_resolution = gr.Slider(label="Text Guidance Scale", minimum=256, maximum=2048, value=1024,
                                                step=32)
                    with gr.Row():
                        with gr.Column():
                            apply_bg = gr.Checkbox(label="BG restoration", value=False)
                        with gr.Column():
                            apply_face = gr.Checkbox(label="Face restoration", value=False)
                with gr.Accordion("Comparison Video options", open=False):
                    with gr.Row():
                        make_comparison_video = gr.Checkbox(
                            label="Generate Comparison Video (Input vs Output) (You need to have FFmpeg installed)",
                            value=False)
                    with gr.Row(visible=False) as compare_video_row:
                        video_duration = gr.Textbox(label="Duration", value="5")
                        video_fps = gr.Textbox(label="FPS", value="30")
                        video_width = gr.Textbox(label="Width", value="1920")
                        video_height = gr.Textbox(label="Height", value="1080")

    with gr.Tab("Image Metadata"):
        with gr.Row():
            metadata_image_input = gr.Image(type="filepath", label="Upload Image")
            metadata_output = gr.Textbox(label="Image Metadata", lines=25, max_lines=50)
        metadata_image_input.change(fn=read_image_metadata, inputs=[metadata_image_input], outputs=[metadata_output])
    with gr.Tab("Restored Faces"):
        with gr.Row():
            face_gallery = gr.Gallery(label='Faces', show_label=False, elem_id="gallery2")
    with gr.Tab("About"):
        gr.Markdown(title_md)
        with gr.Row():
            gr.Markdown(claim_md)
            event_id = gr.Textbox(label="Event ID", value="", visible=False)
        with gr.Accordion("Feedback", open=False):
            fb_score = gr.Slider(label="Feedback Score", minimum=1, maximum=5, value=3, step=1,
                                 interactive=True)
            fb_text = gr.Textbox(label="Feedback Text", value="",
                                 placeholder='Please enter your feedback here.')
            submit_button = gr.Button(value="Submit Feedback")
    #
    stage2_ips = [input_image, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                  s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                  linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select, num_images,
                  random_seed, apply_stage_1, face_resolution, apply_bg, apply_face, face_prompt, make_comparison_video,
                  video_duration, video_fps, video_width, video_height]

    batch_ips = [batch_process_folder, outputs_folder, prompt, a_prompt, n_prompt, num_samples, upscale,
                 edm_steps, s_stage1, s_stage2, s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype,
                 gamma_correction, linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select,
                 num_images, random_seed, apply_stage_1, face_resolution, apply_bg, apply_face, face_prompt,
                 batch_process_llava, temperature, top_p, qs, make_comparison_video, video_duration, video_fps,
                 video_width, video_height]

    output_elements = [prompt, stage_1_output_image, result_gallery, result_slider, event_id, fb_score, fb_text, seed,
                       face_gallery, comparison_video]

    llava_button.click(fn=llava_process_single, inputs=[input_image, temperature, top_p, qs], outputs=output_label,
                       show_progress=True, queue=True)
    stage_1_button.click(fn=stage1_process_single, inputs=[input_image, gamma_correction], outputs=output_label,
                         show_progress=True, queue=True)
    stage_2_button.click(fn=stage2_process_single, inputs=stage2_ips, outputs=output_label,
                         show_progress=True, queue=True)
    start_batch_button.click(fn=batch_upscale, inputs=batch_ips, outputs=output_label,
                             show_progress=True, queue=True)
    stop_batch_button.click(fn=stop_batch_upscale, show_progress=True, queue=True)
    reset_button.click(fn=load_and_reset, inputs=[param_setting],
                       outputs=[edm_steps, s_cfg, s_stage2, s_stage1, s_churn, s_noise, a_prompt, n_prompt,
                                color_fix_type, linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2])

    # We just read the output_label and update all the elements when we find "Processing Complete"
    output_label.change(fn=update_elements, show_progress=False, queue=True, inputs=[output_label],
                        outputs=output_elements)

    make_comparison_video.change(fn=toggle_compare_elements, inputs=[make_comparison_video],
                                 outputs=[comparison_video_col, compare_video_row])
    submit_button.click(fn=submit_feedback, inputs=[event_id, fb_score, fb_text], outputs=[fb_text])
    input_image.change(fn=update_target_resolution, inputs=[input_image, upscale], outputs=[target_res])
    upscale.change(fn=update_target_resolution, inputs=[input_image, upscale], outputs=[target_res])

if args.port is not None:  # Check if the --port argument is provided
    block.launch(server_name=server_ip, server_port=args.port, share=args.share, inbrowser=args.open_browser)
else:
    block.launch(server_name=server_ip, share=args.share, inbrowser=args.open_browser)
