import cv2
import numpy as np
from skimage.exposure import match_histograms

from PanoramaGenerator.utils import read_frames
from utils import *

def color_enhancement_pipeline(videofile, reference_frame=None):
    """
    Enhances and normalizes colors in the input frames.

    Args:
        frames (list of ndarray): List of sampled and stabilized frames (BGR format).
        reference_frame (ndarray, optional): A frame to use as a color reference for histogram matching.
                                             If None, the first frame is used as the reference.

    Returns:
        enhanced_frames (list of ndarray): List of color-enhanced frames.
    """
    frames = read_frames(videofile)
    enhanced_frames = []

    # Use the first frame as the reference if none is provided
    if reference_frame is None:
        reference_frame = frames[0]

    # Histogram Matching Function
    def apply_histogram_matching(frame, ref_frame):
        return match_histograms(frame, ref_frame)

    # CLAHE Function
    def apply_clahe(frame):
        if frame.dtype != np.uint8:
            frame = np.uint8(np.clip(frame, 0, 255))

        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_frame)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Color Balance Function
    def adjust_color_balance(frame, target_color=[1.2, 1.0, 0.8]):
        b, g, r = cv2.split(frame)
        r = np.clip(r * target_color[0], 0, 255).astype('uint8')
        g = np.clip(g * target_color[1], 0, 255).astype('uint8')
        b = np.clip(b * target_color[2], 0, 255).astype('uint8')

        return cv2.merge((b, g, r))

    for frame in frames:
        #matched_frame = apply_histogram_matching(frame, reference_frame)
        clahe_frame = apply_clahe(frame)
        #balanced_frame = adjust_color_balance(clahe_frame)
        enhanced_frames.append(clahe_frame)

    return enhanced_frames
