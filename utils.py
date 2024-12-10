from VideoStabilization import VidStab
import cv2

def video_rendering(frames, filename = "tmp.mp4"):
    h, w = frames[0].shape[0], frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
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
            frame = cv2.resize(img, (720, 1080))
            images.append(frame)
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



def stabilization(file_path, output_path):

    stabilizer = VidStab(kp_method='FAST', threshold=32, nonmaxSuppression=False)
    processed_frames = stabilizer.stabilize(input_path=file_path, smoothing_window=3,output_path=output_path, show_progress=True)
    scene_changes = stabilizer.smoothed_trajectory
    return processed_frames, scene_changes


def frame_sampling(frames, scene_changes):
    dx = []
    for change in scene_changes:
        dx.append(change[0])

    indices_sorted_by_x = sorted(range(len(dx)), key=lambda z: dx[z], reverse=True)

    max_x_idx = indices_sorted_by_x[0]
    min_x_idx = indices_sorted_by_x[-1]

    sampled_frames = []
    sampled_idx = []

    if max_x_idx < min_x_idx:
        sampled_frames.append(frames[max_x_idx])
        curr_idx = max_x_idx
        sampled_idx.append(curr_idx)
        for idx in range(max_x_idx, min_x_idx+1):
            if dx[idx] < dx[curr_idx] - 40:
                sampled_frames.append(frames[idx])
                curr_idx = idx
                sampled_idx.append(curr_idx)

    else:
        sampled_frames.append(frames[min_x_idx])
        curr_idx = min_x_idx
        sampled_idx.append(curr_idx)
        for idx in range(min_x_idx, max_x_idx+1):
            if dx[idx] > dx[curr_idx] + 40:
                sampled_frames.append(frames[idx])
                curr_idx = idx
                sampled_idx.append(curr_idx)

    return sampled_frames

