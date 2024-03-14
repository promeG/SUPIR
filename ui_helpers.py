import json
import os
import re
import subprocess
from typing import List, Dict, Union

import filetype
import gradio as gr
from ffmpeg_progress_yield import FfmpegProgress
from tqdm import tqdm

from SUPIR.perf_timer import PerfTimer
from SUPIR.utils.status_container import MediaData


def is_video(video_path: str) -> bool:
    return is_file(video_path) and filetype.helpers.is_video(video_path)


def is_image(image_path: str) -> bool:
    return is_file(image_path) and filetype.helpers.is_image(image_path)


def is_file(file_path: str) -> bool:
    return bool(file_path and os.path.isfile(file_path))


def detect_hardware_acceleration() -> (str, str, str):
    hw_accel_methods = [
        {'name': 'cuda', 'encoder': 'h264_nvenc', 'decoder': 'h264_cuvid', 'regex': re.compile(r'\bh264_nvenc\b')},
        {'name': 'qsv', 'encoder': 'h264_qsv', 'decoder': 'h264_qsv', 'regex': re.compile(r'\bh264_qsv\b')},
        {'name': 'vaapi', 'encoder': 'h264_vaapi', 'decoder': 'h264_vaapi', 'regex': re.compile(r'\bh264_vaapi\b')},
        # Add more methods here as needed, following the same structure
    ]

    ffmpeg_output = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True).stdout

    for method in hw_accel_methods:
        if method['regex'].search(ffmpeg_output):
            # Hardware acceleration method found
            return method['name'], method['decoder'], method['encoder']

    # No supported hardware acceleration found
    return '', '', ''


def extract_video(video_path: str, output_path: str, quality: int = 100, format: str = 'png') -> (bool, Dict[str, str]):
    # Extract video parameters from the original video
    video_params = get_video_params(video_path)

    # Determine the scale factor based on quality (100 being the best quality)
    scale = f"scale=iw*{quality / 100}:-1"

    # Auto-detect hardware acceleration and set video codec accordingly
    hw_acceleration, hw_decoder, hw_encoder = detect_hardware_acceleration()
    # Adjust codec for image output, especially for PNG
    if format.lower() == 'png':
        codec = 'png'
    else:
        codec = hw_encoder if hw_acceleration else 'libx264'

    # Update the output path to include the desired format
    output_path_with_format = f"{output_path}/%05d.{format}"

    # Construct ffmpeg command with quality, format, and potential hardware acceleration
    commands = ['-hwaccel', hw_acceleration, '-i', video_path, '-vf', scale, '-c:v', codec]

    # If framerate information is available, use it in the command
    if 'framerate' in video_params:
        commands.insert(-2, f"fps=fps={video_params['framerate']}")  # Insert before '-c:v' argument

    # Ensure the pixel format is appropriate for the output format
    if format.lower() == 'png':
        commands.extend(['-pix_fmt', 'rgb24'])

    # For formats that support variable quality, adjust accordingly
    if format.lower() in ['jpeg', 'jpg', 'webp']:
        # Adjust quality for lossy formats; scale from 0-100 to codec-specific scale if needed
        commands.extend(['-qscale:v', str(quality)])

    commands.append(output_path_with_format)

    # Update video_params based on the scale (quality adjustment) and format
    video_params['format'] = format
    # Update other parameters as necessary based on the scale and format

    if run_ffmpeg_progress(commands):
        return True, video_params

    return False, video_params


def get_video_params(video_path: str) -> Dict[str, str]:
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
           'stream=width,height,r_frame_rate,avg_frame_rate,codec_name', '-of', 'json', video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Parse ffprobe output to json
        info = json.loads(result.stdout)
        # Extract video stream information
        stream = info['streams'][0]  # Assuming the first stream is the video
        # Calculate framerate as float
        framerate = eval(stream['r_frame_rate'])

        return {
            'width': stream['width'],
            'height': stream['height'],
            'framerate': f"{framerate:.2f}",
            'codec': stream['codec_name']
        }
    except Exception as e:
        print(f"Error extracting video parameters: {e}")
        return {}


def compile_video(src_path, output_path, video_params: Dict[str, str], quality: int = 23,
                  file_type: str = 'mp4') -> Union[MediaData, bool]:
    # Determine the codec and hardware acceleration settings
    hw_acceleration, hw_decoder, hw_encoder = detect_hardware_acceleration()
    codec = hw_encoder if hw_encoder else 'libx264'

    # Adjust the output_path to include the desired filetype
    output_path_with_type = f"{output_path}.{file_type}"

    # Construct ffmpeg command with hardware acceleration, quality, and filetype
    commands = ['-y', '-f', 'image2', '-framerate', video_params.get('framerate', '30'), '-i', src_path]

    # Add hardware acceleration flags if detected
    if hw_acceleration:
        commands += ['-hwaccel', hw_acceleration]

    commands += ['-c:v', codec, '-crf', str(quality), '-pix_fmt', 'yuv420p', output_path_with_type]

    # If video_params include resolution, use it to scale the video up or down
    if 'width' in video_params and 'height' in video_params:
        commands += ['-vf', f"scale={video_params['width']}:{video_params['height']}"]

    # If video is compiled successfully, update status_container with the output path
    if run_ffmpeg_progress(commands):
        image_data = MediaData(src_path, 'video')
        image_data.outputs = [output_path_with_type]
        return image_data
    return False


def run_ffmpeg_progress(args: List[str], progress=gr.Progress()):
    commands = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    commands.extend(args)
    print(f"Executing ffmpeg: '{' '.join(commands)}'")
    try:
        ff = FfmpegProgress(commands)
        last_progress = 0  # Keep track of the last progress value
        with tqdm(total=100, position=1, desc="Processing") as pbar:
            for p in ff.run_command_with_progress():
                increment = p - last_progress  # Calculate the increment since the last update
                pbar.update(increment)  # Update tqdm bar with the increment
                pbar.set_postfix(progress=p)
                progress(p / 100, "Extracting frames")  # Update gr.Progress with the normalized progress value
                last_progress = p  # Update the last progress value
        return True
    except Exception as e:
        print(f"Exception in run_ffmpeg_progress: {e}")
        return False


last_time = None
ui_args = None
timer = None


def printt(msg, progress=gr.Progress(), reset: bool = False):
    global ui_args, last_time, timer
    graph = None
    if ui_args is not None and ui_args.debug:
        if timer is None:
            timer = PerfTimer(print_log=True)
        if reset:
            graph = timer.make_graph()
            timer.reset()
        if not timer.print_log:
            timer.print_log = True
        timer.record(msg)
    else:
        print(msg)
    if graph:
        return graph
