import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def plot_frame_transformations(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    dx_list = []
    dy_list = []

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])

        dx_list.append(dx)
        dy_list.append(dy)

        prev_gray = curr_gray

    cap.release()

    plt.figure(figsize=(10, 6))
    plt.plot(dx_list, label="dx (Displacement in x)", color="blue")
    plt.plot(dy_list, label="dy (Displacement in y)", color="red")
    plt.xlabel("Frame Number")
    plt.ylabel("Displacement")
    plt.title("Frame-to-Frame Transformations")
    plt.legend()
    plt.grid(True)
    plt.show()



frames = video_reader("./videos/1.MOV")
processed_frames,scene_changes =stabilization("./videos/1.MOV", "./processing/pre_processing_tmp.mp4")
sorted_frames =frame_sampling(processed_frames,scene_changes)
video_rendering(sorted_frames,"./processing/pre_processing_tmp.mp4")
#stablization_second_pass("./processing/pre_processing_tmp.mp4", "./processing/pre_processing_output.mp4")

plot_frame_transformations("./processing/pre_processing_tmp.mp4")