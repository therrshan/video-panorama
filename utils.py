import copy
from VideoStabilization import VidStab
import cv2

def video_rendering(frames, filename = "tmp.mp4"):
    h, w = frames[0].shape[0], frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, 30.0,(w, h))

    frames = frames
    for frame in frames:
        out.write(frame)
    out.release()

def images_from_paths(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Unable to load image at {path}")
        else:
            newframe = cv2.resize(img, (720, 1080))
            images.append(newframe)
    return images

def video_reader(filename = "input.mp4"):
    cap = cv2.VideoCapture(filename)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()
    return frames



def stablization_first_pass(file_path = 'input.mp4', output_path = "tmp.mp4"):

    stabilizer = VidStab(kp_method='FAST', threshold=32, nonmaxSuppression=False)
    processed_frames = stabilizer.stabilize(input_path=file_path,
                                            smoothing_window=5,
                                            output_path=output_path,
                                            show_progress=True)

    scene_changes = stabilizer.smoothed_trajectory
    stabilizer.plot_trajectory()

    return processed_frames, scene_changes


def directional_correction(processed_frames, scene_changes):
    x_corrds = []

    for val in scene_changes:
        valX = val[0]
        x_corrds.append(valX)

    indices = list(range(len(x_corrds)))

    indices.sort(key=lambda i: x_corrds[i], reverse=True)
    sorted_indices = [int(i) for i in indices]


    max_x_idx = sorted_indices[0]
    min_x_idx = sorted_indices[-1]


    sorted_frames = []
    processed_frames
    sorted_indices
    x_corrds
    kepted_idx = []

    if max_x_idx < min_x_idx:
        sorted_frames.append(processed_frames[max_x_idx])
        curr_idx = max_x_idx
        kepted_idx.append(curr_idx)
        for idx in range(max_x_idx, min_x_idx+1):
            if x_corrds[idx] < x_corrds[curr_idx] - 40:
                sorted_frames.append(processed_frames[idx])
                curr_idx = idx
                kepted_idx.append(curr_idx)

    else:
        sorted_frames.append(processed_frames[min_x_idx])
        curr_idx = min_x_idx
        kepted_idx.append(curr_idx)
        for idx in range(min_x_idx, max_x_idx+1):
            if x_corrds[idx] > x_corrds[curr_idx] + 40:
                sorted_frames.append(processed_frames[idx])
                curr_idx = idx
                kepted_idx.append(curr_idx)
        sorted_frames = copy.deepcopy(sorted_frames[::-1])
        kepted_idx =  copy.deepcopy(kepted_idx[::-1 ])

    return sorted_frames


def stablization_second_pass(input_path = "directional_corrected.mp4", output_path = "tmp.mp4"):

    stabilizer = VidStab(kp_method='FAST', threshold=32, nonmaxSuppression=False)
    processed_frames = stabilizer.stabilize(input_path=input_path,
                                            smoothing_window=5,
                                            output_path=output_path,
                                            show_progress=True)

    scene_changes = stabilizer.smoothed_trajectory

    stabilizer.plot_trajectory()

    import matplotlib.pyplot as plt
    plt.show()



# get the frame width and height
def get_frame_wh(frames):
    if len(frames)>0:
        h = frames[0].shape[0]
        w = frames[0].shape[1]
        return w, h
    else:
        return None
