import sys
import os
import argparse
import numpy as np
from skimage import io, transform
from skimage.filters import frangi
import matplotlib.pyplot as plt


def normalize_image(img, low_pct=1, high_pct=90):
	"""
	Normalize a single-channel image to [0, 1], 
	with percentile-based clamping to remove extreme values.

	Parameters:
	- img: 2D numpy array (image)
	- low_pct: lower percentile to clamp (e.g. 1)
	- high_pct: upper percentile to clamp (e.g. 99)

	Returns:
	- normalized image as float32 in range [0, 1]
	"""
	low_val = np.percentile(img, low_pct)
	high_val = np.percentile(img, high_pct)

	img_clamped = np.clip(img, low_val, high_val)
	norm_img = (img_clamped - low_val) / (high_val - low_val)
	
	return norm_img.astype(np.float32)

def rescale_image(img, output_size=(512,512)):
	resized = transform.resize(img, output_size[::-1], anti_aliasing=True)
	return resized

def frangi_filter(img):
	return frangi(img, sigmas=[20,30], alpha=0.6, beta=0.5, black_ridges=False)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Process image and label directories.")
	parser.add_argument(
		"file_path",
		type = str,
		help = "Path to source tif image"
	)
	parser.add_argument(
		"frame",
		type = int,
		help="The frame index to extract (starting at 1)"
	)
	parser.add_argument(
		"--enhance",
		action = "store_true",
		help="Flag for image enhancement"
	)
	args = parser.parse_args()
	
	file_path = args.file_path
	filename = os.path.basename(file_path)
	directory = os.path.dirname(file_path)
	frame_idx = args.frame-1
	enhance = args.enhance
	
	img = io.imread(file_path)
	print("Shape of image:", img.shape)
	frame = img[frame_idx,:,:]

	if enhance:
		normalized = normalize_image(frame)
		filtered = frangi_filter(normalized)
		frame = normalize_image(filtered+normalized)
		
	out = rescale_image(frame)
	img_uint8 = (out * 255).astype(np.uint8)
	io.imsave(os.path.join(directory, "frame_%d_%s"%(frame_idx,filename)), img_uint8)