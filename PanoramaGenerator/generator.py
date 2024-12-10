import cv2
import numpy as np

from .Stitcher import StitcherLeft, StitcherRight
from .utils import resize_frame, padding_pano, read_frames
from .cylindricalWarp import cylindricalwarp


def calc_right(img1, img2):
    stitcher = StitcherRight()
    result = stitcher.stitch([img1, img2], showMatches=True)

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    final_result = cv2.convertScaleAbs(final_result)
    final_result = stitcher.crop_result(final_result)
    return final_result

def calc_left(img1, img2):
    stitcher = StitcherLeft()
    result = stitcher.stitch([img1, img2])

    # remove blank rows and columns
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    final_result = cv2.convertScaleAbs(final_result)
    final_result = stitcher.crop_result(final_result)
    return final_result

def mergeLeft(image, result):
    frame_new = cylindricalwarp(image)
    frame_new = resize_frame(frame_new)
    result = padding_pano(frame_new, result)
    result = calc_left(result, frame_new)
    return result


def mergeRight(result, image):
    frame_new = cylindricalwarp(image)
    frame_new = resize_frame(frame_new)
    result = calc_right(result, frame_new)
    return result


def cut_corners(img):
    h, w = img.shape[:2]
    percent = 0.08
    new_width = int(w * (1 - 2 * percent))
    newheight = int(h * (1 - 2 * percent))

    height_start = int(h * percent)
    width_start = int(w * percent)

    new_img = img[height_start: height_start + newheight, width_start: width_start + new_width]
    return new_img


def generate_pano(video_path, output_path):
    frames = read_frames(video_path)
    length = len(frames)
    mid_index = length // 2
    result = frames[mid_index]
    result = resize_frame(result)
    i = 1

    while mid_index - i >= 0 or mid_index + i < length:
        if mid_index - i >= 0:
            result = mergeLeft(frames[mid_index - i], result)
        if mid_index + i < length:
            result = mergeRight(result, frames[mid_index + i])
        i += 1
    cv2.imwrite(output_path, cut_corners(result))


def generate_pano_images(frames, output_path):
    length = len(frames)
    mid_index = length // 2
    result = frames[mid_index]
    result = resize_frame(result)
    i = 1

    while mid_index - i >= 0 or mid_index + i < length:
        if mid_index - i >= 0:
            result = mergeLeft(frames[mid_index - i], result)
        if mid_index + i < length:
            result = mergeRight(result, frames[mid_index + i])
        i += 1

    cv2.imwrite(output_path, cut_corners(result))

def generate_pano_lr(video_path, output_path):
    frames = read_frames(video_path)
    length = len(frames)
    mid_index = length // 2
    result = frames[0]
    result = resize_frame(result)
    i = 1

    while  i < length:
        result = mergeRight(result, frames[i])
        i += 1

    cv2.imwrite(output_path, cut_corners(result))

def generate_pano_lr_images(frames, output_path):
    length = len(frames)
    mid_index = length // 2
    result = frames[0]
    result = resize_frame(result)
    i = 1

    while  i < length:
        result = mergeRight(result, frames[i])
        i += 1

    cv2.imwrite(output_path, cut_corners(result))