
import argparse
from utils import *
from PanoramaGenerator import generator
from ColorCorrection import correction, vibrancyscore
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename")
parser.add_argument("--output", type=str, default='pano.png', help="name of output file")

args = parser.parse_args()
filename = args.filename
output = args.output

input_file = './videos/'+filename

frames = video_reader(input_file)
processed_frames,scene_changes =stablization_first_pass(input_file, "./processing/pre_processing_tmp.mp4")
sorted_frames =directional_correction(processed_frames,scene_changes)
video_rendering(sorted_frames,"./processing/pre_processing_tmp.mp4")
stablization_second_pass("./processing/pre_processing_tmp.mp4", "./processing/pre_processing_output.mp4")

#reference_frame, _ = vibrancyscore.select_vibrant_frame("./processing/pre_processing_output.mp4")
# ce_frames = correction.color_enhancement_pipeline("./processing/pre_processing_output.mp4")
# video_rendering(ce_frames,"./processing/pre_processing_ce.mp4")

pano = generator.generate_pano("./processing/pre_processing_output.mp4", output)
#pano = generator.generate_pano_images(frames, 'test_mid.png')