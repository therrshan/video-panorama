import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def plot_frame_transformations(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Initialize lists to store dx and dy transformations
    dx_list = []
    dy_list = []

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        cap.release()
        return

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, curr_frame = cap.read()
        if not ret:
            break  # End of video

        # Convert the current frame to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow using Farneback's method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx = np.mean(flow[..., 0])  # Average displacement in x
        dy = np.mean(flow[..., 1])  # Average displacement in y

        # Append the dx, dy values
        dx_list.append(dx)
        dy_list.append(dy)

        # Update the previous frame
        prev_gray = curr_gray

    # Release the video capture object
    cap.release()

    # Plot the transformations
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
processed_frames,scene_changes =stablization_first_pass("./videos/1.MOV", "./processing/pre_processing_tmp.mp4")
sorted_frames =directional_correction(processed_frames,scene_changes)
video_rendering(sorted_frames,"./processing/pre_processing_tmp.mp4")
#stablization_second_pass("./processing/pre_processing_tmp.mp4", "./processing/pre_processing_output.mp4")

plot_frame_transformations("./processing/pre_processing_tmp.mp4")