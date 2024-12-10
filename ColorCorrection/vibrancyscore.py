import cv2
import numpy as np

from PanoramaGenerator.utils import read_frames
from utils import *

def select_vibrant_frame(videofile):
    """
    Selects the most vibrant frame based on color saturation and intensity variance.

    Args:
        frames (list of ndarray): List of frames in BGR format.

    Returns:
        vibrant_frame (ndarray): The most vibrant frame.
        vibrant_index (int): Index of the most vibrant frame.
    """
    vibrancy_scores = []
    frames = read_frames(videofile)
    for frame in frames:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, saturation, intensity = cv2.split(hsv_frame)
        mean_saturation = np.mean(saturation)
        intensity_variance = np.var(intensity)
        vibrancy_score = mean_saturation + 0.5 * intensity_variance
        vibrancy_scores.append(vibrancy_score)

    vibrant_index = np.argmax(vibrancy_scores)
    vibrant_frame = frames[vibrant_index]

    return vibrant_frame, vibrant_index
