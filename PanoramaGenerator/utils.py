import cv2
from matplotlib import pyplot as plt
import numpy as np

def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (720, 1080))
        frames.append(frame)
    return frames

def resize_frame(frame):
    h, w = frame.shape[:2]
    h_dash, w_dash = int(h * 1.1), int(w * 1.1)
    up = int((h_dash - h) / 2)
    left = int((w_dash - w) / 2)
    down = int(up + h)
    right = int(left + w)
    resized_frame = cv2.resize(frame, (w_dash, h_dash))
    res = resized_frame[up:down, left:right, ]
    return res

def padding_pano(img, pano):
    width = img.shape[1]
    h, w, c = pano.shape
    mask = np.zeros((h, width, c), dtype=np.uint8)
    pad = np.concatenate((mask, pano), axis=1)
    return pad