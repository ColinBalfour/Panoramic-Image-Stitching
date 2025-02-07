#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""
import numpy as np
import cv2
import torch
import logging
from typing import List, Optional, Tuple
import networkx as nx

# Add any python libraries here
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max
from scipy.sparse import csr_matrix
import os
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Network.Network import *


# # Check GPU availability
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"GPU Device: {torch.cuda.get_device_name(0)}")
#
# if cv2.cuda.getCudaEnabledDeviceCount():
#     cv2.cuda.setDevice(0)
#     cv2.ocl.setUseOpenCL(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(path=None, im_set=None, model=None):
    # print(path)
    if path is None:
        path = "Set1"
    os.makedirs(f'outputs/{path}', exist_ok=True)

    # new_width = 600
    # new_height = 450

    if im_set is None:
        DIR_PATH = f'D:/Computer vision/Homeworks/Project Phase1/YourDirectoryID_p1/YourDirectoryID_p1/Phase1/Data/Train/'
        #DIR_PATH = f'Phase1/Data/Train/'
        im_set = [cv2.imread(f'{DIR_PATH}/{path}/{i + 1}.jpg') for i in
                  range(len(os.listdir(f'{DIR_PATH}/{path}')))]

    for im in im_set:
        cv2.imshow('image', im)
        cv2.waitKey(2000)

    def generate_patch(im_1, im_2,  patchsize = 128):
        im1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)
        im1 = cv2.resize(im1, (320,240))
        im2 = cv2.resize(im2, (320,240))
        x_offset = (320 - patchsize) // 2
        y_offset = (240 - patchsize) // 2
        patch1 = im1[y_offset:y_offset + patchsize, x_offset:x_offset + patchsize]
        patch2 = im2[y_offset:y_offset + patchsize, x_offset:x_offset + patchsize]
        patches = np.dstack([patch1, patch2])
        patches = (patches.astype(np.float32) - 127.5) / 127.5
        patches = torch.from_numpy(patches).permute(2, 0, 1).unsqueeze(0).to(device)
        return patches

    def get_homography(im1, im2, model):
        """
        Get homography matrix from two images using the deep learning model
        Args:
            im1: First image
            im2: Second image
            model: Trained HomographyNet model
        Returns:
            H: 3x3 homography matrix as numpy array
        """
        # Generate patches and prepare input
        input_patches = generate_patch(im1, im2)

        # Ensure model is in evaluation mode
        model.eval()

        with torch.no_grad():
            # Get 4-point prediction from model
            h4_prediction = model(input_patches)

            # Convert to homography matrix
            H = homographymatrix(h4_prediction)

            # Convert from torch tensor to numpy array and get first matrix (remove batch dimension)
           # H = H[0].cpu().numpy()

        return H

    def homographymatrix(Hfourpoints):
        """
        Convert 4-point parameterization to 3x3 homography matrix
        """
        # Convert tensor to numpy array
        Hfourpoints = Hfourpoints.cpu().detach().numpy()
        print("Hfourpoints", Hfourpoints)

        # Define source points
        pts1 = [
            [96.0, 56.0],  # top-left
            [224.0, 56.0],  # top-right
            [224.0, 184.0],  # bottom-right
            [96.0, 184.0]  # bottom-left
        ]
        # target = Hfourpoints.view(-1, 8).float()
        # print("target", target)
        deltas = Hfourpoints.reshape(-1) *32  # Scale back by 32
        print("deltas",deltas)

        # Calculate destination points
        pts2 = []
        for i in range(4):
            x = pts1[i][0] + deltas[i * 2]  # Get x coordinate
            y = pts1[i][1] + deltas[i * 2 + 1]  # Get y coordinate
            pts2.append([x, y])

        # Construct the A matrix for DLT
        A = []
        for i in range(4):
            x, y = pts1[i]
            xp, yp = pts2[i]

            A.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp])
            A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp])

        A = np.array(A)

        # Solve using SVD
        _, _, V = np.linalg.svd(A)

        # Get the last row of V (corresponding to smallest singular value)
        h = V[-1]

        # Reshape into 3x3 matrix
        H = h.reshape(3, 3)

        # Normalize so that H[2,2] = 1
        H = H / H[2, 2]

        return H

    homography_set = {}
    cumulative_homographies = {0: (0, np.eye(3))}
    # for i, im1 in enumerate(im_set):
    #     for j,im2 in enumerate(im_set, start =1+1):
    #         H = get_homography(im1, im2, model)
    #         if i == 0:  # If we're at the first pair, just store the homography H
    #             cumulative_homographies[j] = (0, H)
    #         else:
    #             # For subsequent pairs, multiply by the previous cumulative homography
    #             H_cumulative = H @ cumulative_homographies[i][1]
    #             cumulative_homographies[j] = (i, H_cumulative)

    for i, im1 in enumerate(im_set):
        for j, im2 in enumerate(im_set[i + 1:], start=i + 1):
            H = get_homography(im1, im2, model)
            homography_set[(i, j)] = H
            if i == 0:
                cumulative_homographies[j] = (0, H)
            else:
                H_cumulative = H @ cumulative_homographies[i][1]
                cumulative_homographies[j] = (i, H_cumulative)

    warped_images = []
    warped_images = []
    translation_offset = {}
    corners_list = []

    homography_idx = sorted(list(cumulative_homographies.keys()))
    for i in homography_idx:
        image = im_set[i]
        from_index, H = cumulative_homographies[i]
        h, w = image.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        warped_corners = np.linalg.inv(H) @ corners
        warped_corners = warped_corners / warped_corners[2]
        corners_list.append(warped_corners[:2].T)

    # size of canvas
    all_corners = np.vstack(corners_list)
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y
    MAX_CANVAS_SIZE = 30000
    canvas_width = min(canvas_width, MAX_CANVAS_SIZE)
    canvas_height = min(canvas_height, MAX_CANVAS_SIZE)
    f = 750

    for i in homography_idx:
        image = im_set[i]
        from_index, H = cumulative_homographies[i]
        h, w = image.shape[:2]
        T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        H_combined = T @ np.linalg.inv(H)
        # H_combined = translation_matrix @ np.linalg.inv(H)
        # Apply homography
        # Limit canvas size
        warped = cv2.warpPerspective(image, H_combined, (canvas_width, canvas_height))
        # create mask
        mask = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY)[1]
        # # Create erosion kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)  # Erode mask to remove border artifacts
        # # Zero out border pixels
        warped[mask == 0] = 0
        warped_images.append(warped)
        # cv2.imwrite(f'outputs/{path}/warped_{from_index}_{i}.png', warped)
        # Get maximum dimensions
        # max_height, max_width = get_panorama_size(warped_images)
        # # Resize all warped images to the same size
        # warped_images = [cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_LINEAR)
        #                 if img.shape[:2] != (max_height, max_width) else img
        #                  for img in warped_images]
        cv2.imwrite(
            f'D:/Computer vision/Homeworks/Project Phase1/YourDirectoryID_p1/YourDirectoryID_p1/Phase1/Code/outputs/{path}/warped_{from_index}_{i}.png',
            warped)

    def get_panorama_size(warped_images):
        max_height = max(img.shape[0] for img in warped_images)
        max_width = max(img.shape[1] for img in warped_images)
        return max_height, max_width

    def create_overlap_mask(img1, img2):
        """
        Create a mask where both images have non-zero values
        """
        mask1 = np.any(img1 > 0, axis=2).astype(np.float32)
        mask2 = np.any(img2 > 0, axis=2).astype(np.float32)
        overlap_mask = mask1 * mask2
        return overlap_mask

    def create_distance_weights(overlap_mask):
        """
        Create weight maps based on distance transform from boundaries
        """
        # Get distance from boundaries for both images
        dist1 = cv2.distanceTransform((overlap_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

        # Normalize the distance transforms
        dist1 = cv2.normalize(dist1, None, 0, 1, cv2.NORM_MINMAX)

        # Create complementary weights
        weights1 = dist1
        weights2 = 1 - dist1

        return weights1, weights2


    def adjust_exposure(source, target, overlap_mask):
        source_float = source.astype(np.float32)
        target_float = target.astype(np.float32)

        adjusted = source.copy().astype(np.float32)
        overlap_pixels = overlap_mask > 0

        if not np.any(overlap_pixels):
            return source

        for i in range(3):
            source_vals = source_float[:, :, i][overlap_pixels]
            target_vals = target_float[:, :, i][overlap_pixels]

            if len(source_vals) > 0 and source_vals.mean() > 0:
                ratio = target_vals.mean() / source_vals.mean()
                adjusted[:, :, i] = source_float[:, :, i] * ratio

        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def blend_images_complete(img1, img2):
        """
        Blend two images using their complete overlap area
        """
        # Create overlap mask
        overlap_mask = create_overlap_mask(img1, img2)

        # Adjust exposure of img1 to match img2 in overlap region
        img1_adjusted = adjust_exposure(img1, img2, overlap_mask)

        # Get weight maps for blending
        weights1, weights2 = create_distance_weights(overlap_mask)

        # Expand weights to 3 channels
        weights1 = np.expand_dims(weights1, axis=2)
        weights2 = np.expand_dims(weights2, axis=2)

        # Perform weighted blending
        blended = np.zeros_like(img1, dtype=np.float32)

        # Where both images have content, blend using weights
        overlap = (overlap_mask > 0)
        blended[overlap] = (
                img1_adjusted[overlap] * weights1[overlap] +
                img2[overlap] * weights2[overlap]
        )

        # Where only img1 has content
        img1_only = np.logical_and(np.any(img1 > 0, axis=2), ~overlap)
        blended[img1_only] = img1_adjusted[img1_only]

        # Where only img2 has content
        img2_only = np.logical_and(np.any(img2 > 0, axis=2), ~overlap)
        blended[img2_only] = img2[img2_only]

        return np.clip(blended, 0, 255).astype(np.uint8)

    def blend_image_sequence(warped_images):
        """
        Blend a sequence of warped images using complete image blending
        """
        if not warped_images:
            return None

        result = warped_images[0]

        for i in range(1, len(warped_images)):
            result = blend_images_complete(result, warped_images[i])
            cv2.imwrite(f'outputs/{path}/blended_{i}.png', result)

        return result

    print("finished stitching, blending...")
    final_result = blend_image_sequence(warped_images)
    # cv2.imwrite('final_blended_result.jpg', final_result)

    cv2.imwrite(f'D:/Computer vision/Homeworks/Project Phase1/YourDirectoryID_p1/YourDirectoryID_p1/Phase1/Code/outputs/{path}/mypano.png', final_result)

    # cv2.imwrite(f'outputs/{path}/mypano.png', panorama)

    return final_result


if __name__ == "__main__":
    PAIRWISE = False
    path = "Set1"
    DIR_PATH = f'D:/Computer vision/Homeworks/Project Phase1/YourDirectoryID_p1/YourDirectoryID_p1/Phase1/Data/Train/'

    # Load model first
    model = HomographyNet(InputSize=2 * 128 * 128, OutputSize=8).to(device)
    checkpoint = torch.load('D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Code/TxtFiles/Checkpointsfinal/49model.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load images
    im_set = [cv2.imread(f'{DIR_PATH}/{path}/{i + 1}.jpg') for i in
              range(len(os.listdir(f'{DIR_PATH}/{path}')))]

    if not PAIRWISE:
        # Process all images at once
        result = main(path,im_set, model)
    else:
        # Process pairs sequentially
        for i in range(len(im_set) - 1):
            result = main(path, im_set[i:i + 2], model)
            im_set[i + 1] = result

    # Save final result
    cv2.imwrite(f'outputs/{path}/mypano.png', result)

