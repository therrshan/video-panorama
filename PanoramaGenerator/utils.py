import cv2
from matplotlib import pyplot as plt
import numpy as np

# read in frames from video file
def read_frames(video_path='./workfolder/pre_processing_output.mp4'):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        newframe = cv2.resize(frame, (720, 1080))
        frames.append(newframe)
    return frames


# show image using matplotlib, easy for debugging and illustration
def show(img):
    img = cv2.convertScaleAbs(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# utils for cropping the frame, not used for final implementation
def crop_frame(frame):
    height, width = frame.shape[:2]
    crop_width_A = int(0.05 * width)
    crop_width_B = int(0.95 * width)
    cropped_frame = frame[:, crop_width_A:crop_width_B]
    return cropped_frame


def resize_frame(frame):
    # Get the current size of the image
    h, w = frame.shape[:2]

    # Calculate the new size of the image
    new_h, new_w = int(h * 1.1), int(w * 1.1)
    padding_up = int((new_h - h) / 2)
    padding_left = int((new_w - w) / 2)
    padding_down = int(padding_up + h)
    padding_right = int(padding_left + w)
    # Resize the image to the new size using INTER_AREA interpolation method
    resized_img = cv2.resize(frame, (new_w, new_h))
    resized_img = resized_img[padding_up:padding_down, padding_left:padding_right, ]
    return resized_img

def padding_pano(appendedImg, pano):
    w = appendedImg.shape[1]
    # Get the image size
    height, width, channels = pano.shape
    # Define the desired padding size
    left_pad_size = w
    # Create a black rectangle with the desired padding size
    black_rect = np.zeros((height, left_pad_size, channels), dtype=np.uint8)
    # Concatenate the black rectangle and the original image horizontally
    new_img = np.concatenate((black_rect, pano), axis=1)
    return new_img