import os
import argparse
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm

# Edit this transformation construction to change the augmentation behavior
transform = A.Compose([
	A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_REFLECT), # full rotation
	A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),], p=0.5), # H or V flip
	A.RandomScale(scale_limit=0.1, p=0.5),		# stretch/squeeze
	A.GaussianBlur(blur_limit=(3, 5), p=0.3),	# blur occasionally
	A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), # Adjust contrast 
	A.Resize(512, 512),							# resize to standard size
], additional_targets={'mask': 'mask'})

def augment_n_times(image, mask, transform, n=10):	
	augmented_images = []
	augmented_masks = []
	
	for _ in tqdm(range(n), desc="Augmenting data"):
		augmented = transform(image=image, mask=mask)
		augmented_images.append(augmented['image'])
		augmented_masks.append(augmented['mask'])
	
	return augmented_images, augmented_masks

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Process image and label directories.")
	parser.add_argument(
		"directory",
		type = str,
		help = "Base directory containing 'images' and 'labels' subdirectories"
	)
	parser.add_argument(
		"augment_n",
		type = int,
		nargs = "?",
		default = 10,
		help = "Number of times an image should be augmented"
	)
	args = parser.parse_args()
	
	base_dir = args.directory
	aug_n = args.augment_n

	images_dir = os.path.join(base_dir, "images")
	labels_dir = os.path.join(base_dir, "labels")

	# Get all PNG files in the images and labels directories
	image_files = [f for f in os.listdir(images_dir) if f.endswith(".png") and os.path.isfile(os.path.join(images_dir, f))]
	label_files = [f for f in os.listdir(labels_dir) if f.endswith(".png") and os.path.isfile(os.path.join(labels_dir, f))]
	
	out_img_dir = os.path.join(base_dir, "AugmentedData/images")
	out_label_dir = os.path.join(base_dir, "AugmentedData/labels")
	os.makdirs(out_img_dir, exist_ok=True)
	os.makedirs(out_label_dir, exist_ok=True)
	
	counter = 0
	for img_path, mask_path in zip(image_files, label_files):
		# This treats images as gayscale, but you can do RGB
		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
		
		# Change n to adjust number of augments per image
		a_img, a_masks = augment_n_times(img, mask, transform, n=aug_n)
	
		for ai, am in zip(a_img, a_masks):
			cv2.imwrite(os.path.join(out_img_dir, "img_%d.png"%counter), ai)
			cv2.imwrite(os.path.join(out_label_dir, "mask_%d.png"%counter), am)
			counter += 1