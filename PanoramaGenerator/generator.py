import cv2
import numpy as np

from .Stitcher import StitcherLeft, StitcherRight
from .utils import resize_frame, padding_pano, read_frames
from .cylindricalWarp import cylindricalwarp


def calc_right(img1, img2):
    # img 1 is pano, img2 is new img
    stitcher = StitcherRight()
    result = stitcher.stitch([img1, img2], showMatches=True)

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    final_result = cv2.convertScaleAbs(final_result)
    final_result = stitcher.crop_result(final_result)
    return final_result

# used to call Stitcher_pano_left class object to stitch images to left of panorama
def calc_left(img1, img2):
    # img 1 is new image, img2 is pano
    stitcher = StitcherLeft()
    # stitch
    result = stitcher.stitch([img1, img2])

    # remove blank rows and columns
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    final_result = cv2.convertScaleAbs(final_result)
    final_result = stitcher.crop_result(final_result)
    return final_result


# used to call calc_left to perform stitching
def mergeLeft(image, result):
    # do the fix of barrel distortion first
    frame_new = cylindricalwarp(image)
    # crop the image on left and right side to avoid too much distortion
    frame_new = resize_frame(frame_new)
    # padding the panorama on the left
    result = padding_pano(frame_new, result)
    # stitch images
    result = calc_left(result, frame_new)
    return result


def mergeRight(result, image):
    # do the fix of barrel distortion first
    frame_new = cylindricalwarp(image)
    # crop the image on left and right side to avoid too much distortion
    frame_new = resize_frame(frame_new)
    # stitch images
    result = calc_right(result, frame_new)
    return result


def cut_corners(img):
    # when the final panorama is generated, sometimes it will have small blank parts on the top and buttom, crop it a bit
    h, w = img.shape[:2]
    percent = 0.08
    new_width = int(w * (1 - 2 * percent))
    newheight = int(h * (1 - 2 * percent))

    height_start = int(h * percent)
    width_start = int(w * percent)

    # crop the panorama
    new_img = img[height_start: height_start + newheight, width_start: width_start + new_width]
    return new_img


# use this function to start from the middle point of video frames and stitch frames on its left and right repeatedly
def generate_pano(video_path, output_path):
    print("Generating panorama, please wait, you can check the status on the right window")

    frames = read_frames(video_path)
    length = len(frames)
    mid_index = length // 2
    result = frames[mid_index]
    result = resize_frame(result)
    i = 1

    # starting from the middle point, then stitch frame at index: mid-1, mid+1, mid-2, mid+2 .....until finished
    while mid_index - i >= 0 or mid_index + i < length:
        print(f"Stitching image progress {i * 2} of {length}.")
        if mid_index - i >= 0:
            result = mergeLeft(frames[mid_index - i], result)
        if mid_index + i < length:
            result = mergeRight(result, frames[mid_index + i])
        i += 1

    cv2.imwrite(output_path, cut_corners(result))

def generate_pano_images(frames, output_path):
    print("Generating panorama, please wait, you can check the status on the right window")

    length = len(frames)
    mid_index = length // 2
    result = frames[mid_index]
    result = resize_frame(result)
    i = 1

    # starting from the middle point, then stitch frame at index: mid-1, mid+1, mid-2, mid+2 .....until finished
    while mid_index - i >= 0 or mid_index + i < length:
        print(f"Stitching image progress {i * 2} of {length}.")
        if mid_index - i >= 0:
            result = mergeLeft(frames[mid_index - i], result)
        if mid_index + i < length:
            result = mergeRight(result, frames[mid_index + i])
        i += 1

    cv2.imwrite(output_path, cut_corners(result))

def generate_pano_lr(video_path, output_path):
    print("Generating panorama, please wait, you can check the status on the right window")

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
    print("Generating panorama, please wait, you can check the status on the right window")

    length = len(frames)
    mid_index = length // 2
    result = frames[0]
    result = resize_frame(result)
    i = 1

    while  i < length:
        result = mergeRight(result, frames[i])
        i += 1

    cv2.imwrite(output_path, cut_corners(result))