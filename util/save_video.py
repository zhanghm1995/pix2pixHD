'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-07-24 14:43:17
Email: haimingzhang@link.cuhk.edu.cn
Description: Save video for visualization
'''

from typing import Tuple, List
import os
import os.path as osp
import cv2
import subprocess
from tqdm import tqdm


def scan_image_folder(input_folder):
    return [file for file in os.listdir(input_folder) if file.endswith((".png", ".jpg"))]


def create_video_writer(dst_file, dst_fps, dst_size: Tuple):
    """Create the OpenCV video writer

    Args:
        dst_file ([type]): [description]
        dst_fps ([type]): [description]
        dst_size (Tuple): please note it is (W,H) order

    Returns:
        (VideoWriter): [description]
    """
    if dst_file.endswith(".avi"):
        writer = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), dst_fps, dst_size)
        # writer = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc(*"XVID"), dst_fps, dst_size)
    elif dst_file.endswith(".mp4"):
        writer = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc(*"mp4v"), dst_fps, dst_size)
        # writer = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc(*'XVID'), dst_fps, dst_size)
    else:
        raise ValueError("Unknown file format")

    return writer


def create_video_with_image_folder(image_root, 
                                   video_fps, 
                                   output_dir, 
                                   audio_path=None,
                                   video_name="video",
                                   need_avi_video=False):
    """Create a video file when given a folder contains all image files

    Args:
        image_root (str|Path): a folder path contains all image files
        video_fps (int): generated video FPS
        output_dir (str|Path): the destination folder
        video_name (str): the saving video file name
    """
    img_names = sorted(scan_image_folder(image_root))
    
    if not len(img_names):
        return
    
    ## Save the video
    output_video_name = f"{video_name}.avi"
    idx = -1
    for img_name in tqdm(img_names):
        idx += 1

        frame = cv2.imread(osp.join(image_root, img_name))
        
        if idx == 0:
            img_size = frame.shape[:2]
            writer = create_video_writer(osp.join(output_dir, output_video_name), 
                                         video_fps, 
                                         img_size[::-1])
        
        writer.write(frame)
    writer.release()

    ffmpeg_bin = "/usr/bin/ffmpeg"
    if audio_path is not None:
        ## Add audio channel into this video
        # We choose save .mp4 video by default
        video_path = osp.join(output_dir, output_video_name)

        if need_avi_video:
            ## Save .avi video
            output_video = osp.join(output_dir, f"{video_name}_with_audio.avi")
            command = f"{ffmpeg_bin} -i {audio_path} -i {video_path} -vcodec copy  -acodec copy -y {output_video}"
            subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)
        
        ## Save .mp4 video
        output_video = osp.join(output_dir, f"{video_name}_with_audio.mp4")
        command = f"{ffmpeg_bin} -y -i {audio_path} -i {video_path} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {output_video}"
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)
    else:
        ## Convert the .avi video to .mp4 video
        video_path = osp.join(output_dir, output_video_name)
        output_video = osp.join(output_dir, f"{video_name}.mp4")
        command = f"{ffmpeg_bin} -y -i {video_path} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {output_video}"
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)