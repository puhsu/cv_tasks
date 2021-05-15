import os
import math
from base64 import b64encode

import cv2
import torch
from IPython.display import HTML


def load_frames(path):
    cap = cv2.VideoCapture(path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frames.append(frame)
        count = count + 1

        if (count > (video_length-1)):
            cap.release()
            break

    return frames


def resize_frames(frames, divisor=64):
    H, W, _ = frames[0].shape
    H_ = int(math.ceil(H/divisor) * divisor)
    W_ = int(math.ceil(W/divisor) * divisor)
    return [cv2.resize(frame, (W_, H_)) for frame in frames]


def frames_to_torch(frames):
    return [torch.from_numpy(frame).permute(2,0,1).float()[None] / 255 for frame in frames]


def write_video(file_path, frames, fps=30):
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release()

    # ffmpeg encode to view in colab
    os.system(f"ffmpeg -i {file_path} -vcodec libx264 tmp.mp4")
    os.rename("tmp.mp4", file_path)


def play_video(file_path, width=400):
    with open(file_path,'rb') as mp4:
        data_url = "data:video/mp4;base64," + b64encode(mp4.read()).decode()
        return HTML(f"""
            <video width={width} controls>
                <source src="{data_url}" type="video/mp4">
            </video>
        """)
