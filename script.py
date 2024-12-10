
import argparse
from utils import *
from PanoramaGenerator import generator
from ColorCorrection import correction, vibrancyscore
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename")
parser.add_argument("--output", type=str, default='pano.png', help="name of output file")
parser.add_argument("--no_ce", type=bool, default=False, help="color enhance" )
parser.add_argument("--no_stab", type=bool, default=False, help="set True if resuming just stitching over previously stabilized video.")

args = parser.parse_args()
filename = args.filename
output = args.output
no_ce = args.no_ce
no_stab = args.no_stab

input_file = './videos/'+filename

if not no_stab:
    frames = video_reader(input_file)
    processed_frames,scene_changes = stabilization(input_file, "./processing/tmp.mp4")
    sampled_frames = frame_sampling(processed_frames,scene_changes)
    video_rendering(sampled_frames,"./processing/tmp.mp4")
    processed_frames, _ = stabilization("./processing/tmp.mp4", "./processing/output.mp4")

if not no_ce:
    reference_frame, _ = vibrancyscore.select_vibrant_frame("./processing/output.mp4")
    ce_frames = correction.color_enhancement_pipeline("./processing/output.mp4")
    video_rendering(ce_frames,"./processing/output.mp4")

pano = generator.generate_pano("./processing/output.mp4", output)