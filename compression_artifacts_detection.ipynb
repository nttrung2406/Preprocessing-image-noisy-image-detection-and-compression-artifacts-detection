import time
import functools
import cv2
import glob
import os
import re
import numpy as np
import pyiqa  # pip isntall first
import torch
from matplotlib import pyplot as plt
from skimage import io
from google.colab.patches import cv2_imshow
import time

def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group())

img_paths = sorted(glob.glob("normal/*.jpg"), key=extract_number)
images = [(os.path.basename(img_path), img_path) for img_path in img_paths]

def jpegblock(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Divide image into 8x8 blocks
    blocks = [image[x:x + 8, y:y + 8] for x in range(0, image.shape[0], 8) for y in range(0, image.shape[1], 8)]

    # Calculate variance of each block
    variances = np.array([np.var(block) for block in blocks])

    # Threshold variances to detect blockiness artifacts
    threshold = np.mean(variances) + 2 * np.std(variances)
    blockiness_map = (variances > threshold).astype(np.uint8) * 255

    return blockiness_map

def jpegstd(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate standard deviation of each pixel
    stds = np.std(image, axis=(0, 1))

    # Threshold standard deviations to detect blocking effect
    threshold = np.mean(stds) + 2 * np.std(stds)
    blocking_map = (stds > threshold).astype(np.uint8) * 255

    return blocking_map

def jpeggrad(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate gradient magnitude (Sobel edge detector)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Threshold gradient magnitude to detect ringing artifacts
    threshold = np.mean(gradient_magnitude) + 2 * np.std(gradient_magnitude)
    ringing_map = (gradient_magnitude > threshold).astype(np.uint8) * 255

    return ringing_map


for img_path in img_paths:
    start_pnt = time.time()
    blockiness_map = jpegblock(img_path)
    blocking_map = jpegstd(img_path)
    ringing_map = jpeggrad(img_path)
    image_name = os.path.basename(img_path)
    if np.any(blockiness_map):
        end_pnt = time.time()
        print(f"The image {image_name}: Blockiness artifacts detected")
        print("Time: ", end_pnt - start_pnt)

    elif np.any(blocking_map):
        end_pnt = time.time()
        print(f"The image {image_name}: Blocking effect detected")
        print("Time: ", end_pnt - start_pnt)

    if np.any(ringing_map):
        end_pnt = time.time()
        print(f"The image {image_name}: Ringing artifacts detected")
        print("Time: ", end_pnt - start_pnt)
