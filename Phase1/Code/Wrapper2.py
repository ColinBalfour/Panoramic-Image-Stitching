# !/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2

# Add any python libraries here
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max
import os


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    path = "Set3"
    os.makedirs(f'outputs/{path}', exist_ok=True)

    """
    Read a set of images for Panorama stitching
    """
    # im_set = [cv2.imread(f'Phase1/Data/Train/{path}/{i + 1}.jpg') for i in range(len(os.listdir(f'Phase1/Data/Train/{path}')))]
    im_set = [cv2.imread(f'D:/Computer vision/Homeworks/Project Phase1/YourDirectoryID_p1/YourDirectoryID_p1/Phase1/Data/Train/{path}/{i + 1}.jpg') for i in
              range(len(os.listdir(f'D:/Computer vision/Homeworks/Project Phase1/YourDirectoryID_p1/YourDirectoryID_p1/Phase1/Data/Train/{path}')))]

    for im in im_set:
        cv2.imshow('image', im)
        cv2.waitKey(2000)

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    corner_im = []
    for i, im in enumerate(im_set):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        harris_corners = cv2.cornerHarris(gray, 2, 3, 0.06)
        corner_im.append(harris_corners)
        # print(harris_corners.shape, im.shape)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(harris_corners, None)


        # Threshold for an optimal value, it may vary depending on the image.
        harris_im = deepcopy(im)
        harris_im[dst > 0.01* dst.max()] = [0, 0, 255]

        """
            harris_response = cv2.cornerHarris(image, block_size, ksize, k)
            harris_response = cv2.dilate(harris_response, None)  # Dilate to mark corners

            # Threshold to retain significant corners
            threshold = threshold_ratio * harris_response.max()
            keypoints = np.argwhere(harris_response > threshold)
            return keypoints"""

        # plt.imshow(harris_corners, cmap='gray')
        # plt.show()
        cv2.imwrite(f'outputs/{path}/corners{i}.png', harris_im)

        ### Subpixel accuracy ###
        # ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        # dst = np.uint8(dst)

        # # find centroids
        # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # # define the criteria to stop and refine the corners
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        # # Now draw them
        # res = np.hstack((centroids,corners))
        # res = np.int0(res)
        # im[res[:,1],res[:,0]]=[0,0,255]
        # im[res[:,3],res[:,2]] = [0,255,0]

        # cv2.imwrite(f'corners_subpix{i}.png', im)

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    USE_SUBPIX = False

    features_set = []
    for i, im in enumerate(corner_im):
        # lm = ndimage.maximum_filter(im, 20)
        # mask = (im == lm)
        # im_set[i][mask] = [0,0,255]

        cpy = deepcopy(im_set[i])

        lm = peak_local_max(im, min_distance= 4, threshold_rel=0.01)
        dst = np.zeros(im.shape)
        features = []
        for x, y in lm:
            features.append((y, x))
            dst[x, y] = 1
            im_set[i][x, y] = [0, 0, 255]
            # cv2.circle(im_set[i], (y, x), 1, (0, 0, 255))

        if not USE_SUBPIX:
            features_set.append(features)

        cv2.imwrite(f'outputs/{path}/anms{i}.png', im_set[i])

        # NOTE: might try to use later
        ### Subpixel accuracy ###
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        # print(corners)
        if USE_SUBPIX:
            features_set.append([np.int0([x, y]) for x, y in corners])

        # Now draw them
        res = np.hstack((centroids, corners))
        res = np.int0(res)
        cpy[res[:, 1], res[:, 0]] = [0, 0, 255]
        cpy[res[:, 3], res[:, 2]] = [0, 255, 0]

        # cv2.imwrite(f'corners_subpix{i}.png', cpy)
        cv2.imwrite(f'outputs/{path}/corners_subpix{i}.png', cpy)

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    def get_patch(image, center, patch_size=(41, 41)):
        """
        Extracts a patch around a center pixel in an image.

        Args:
            image (numpy.ndarray): Input image.
            center (tuple): The (x, y) pixel coordinate to center the patch around.
            patch_size (tuple): Size of the patch, (height, width).

        Returns:
            numpy.ndarray: The patch extracted around the given pixel.
        """
        # Get patch size (height, width)
        height, width = patch_size

        # Calculate the top-left and bottom-right corners of the patch
        x, y = center
        top_left_x = max(0, x - width // 2)
        top_left_y = max(0, y - height // 2)
        bottom_right_x = min(image.shape[1], x + width // 2 + 1)
        bottom_right_y = min(image.shape[0], y + height // 2 + 1)

        # Extract the patch from the image
        patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        return patch

    discriptors_set = []
    keypoints_set = []
    for i, im in enumerate(im_set):
        feature_descriptors = []
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        for x, y in features_set[i]:
            patch = get_patch(gray, (x, y), patch_size=(41, 41))
            blur = cv2.GaussianBlur(patch, (5, 5), 0)

            subsampled = cv2.resize(blur, (8, 8))

            feature_vector = subsampled.flatten()
            normalized = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)

            feature_descriptors.append(normalized)

        discriptors_set.append(feature_descriptors)

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""
    match_set = []

    def match_features(descriptors1, descriptors2, threshold=0.75):
        matches = []
        for i, descriptor in enumerate(descriptors1):
            # Sort descriptors in keypoints2 based on their distance to the current descriptor from keypoints1
            distances = [np.linalg.norm(descriptor - d) for d in descriptors2]
            sorted_indices = np.argsort(distances)

            # Apply the ratio test (lowe's ratio test)
            if distances[sorted_indices[0]] < threshold * distances[sorted_indices[1]]:
                match = cv2.DMatch(i, sorted_indices[0], distances[sorted_indices[0]])
                matches.append(match)

        return matches

    matching_set = []
    # Match between all consecutive pairs
    for i, (im1, features1, desc1) in enumerate(zip(im_set[:-1], features_set[:-1], discriptors_set[:-1])):
        for j, (im2, features2, desc2) in enumerate(zip(im_set[i + 1:], features_set[i + 1:], discriptors_set[i + 1:]),
                                                    start=i + 1):
            # Get matches
            matches = match_features(desc1, desc2)
            matches = np.array(matches)
            matching_set.append(matches)

            #  KeyPoints
            key1 = [cv2.KeyPoint(np.float32(x), np.float32(y), 1) for (x, y) in features1]
            key2 = [cv2.KeyPoint(np.float32(x), np.float32(y), 1) for (x, y) in features2]

            # Draw matches
            out = cv2.drawMatches(im1, key1, im2, key2, matches, None)
            cv2.imwrite(f'outputs/{path}/matching_{i}_{j}.png', out)

            print(f"Matched images {i} and {j}: Found {len(matches)} matches")
            # print(match_set)
    # matches = matching_set[0]

    """
    Refine: RANSAC, Estimate Homography
    """

    def get_homography(pairs):
        n_pairs = len(pairs)
        A_matrix = np.zeros((2 * n_pairs, 9))
        for i, pair in enumerate(pairs):
            x1, y1 = pair[0]
            x2, y2 = pair[1]

            row1 = 2 * i
            row2 = 2 * i + 1

            A_matrix[row1] = [-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2]
            A_matrix[row2] = [0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2]

            # Use SVD to solve for homography
        U, S, Vt = np.linalg.svd(A_matrix)
        h = Vt[-1]
        H = h.reshape(3, 3)

        # Normalize homography matrix
        H = H / H[2, 2]

        return H

    def RANSAC(keypoints1, keypoints2, matches, tau=10, N=10000):
        max_inliers = []
        best_homography = None
        for i in range(N):
            # Randomly select 4 pairs
            random_pairs = np.random.choice(matches, 4, replace=False)
            # print("randompairs",random_pairs)
            pairs = [[keypoints1[pair.queryIdx], keypoints2[pair.trainIdx]] for pair in random_pairs]
            # print("pairs", pairs)

            # Compute homography
            H = get_homography(pairs)

            inliers = []
            for pair in matches:
                p1 = np.array(keypoints1[pair.queryIdx] + (1,))  # Adding 1 here
                p2 = np.array(keypoints2[pair.trainIdx] + (1,))
                # p1 = keypoints1[pair.queryIdx] + (1,))
                # p2 = keypoints2[pair.trainIdx] + (1,))

                p2_pred = H @ p1
                p2_pred = p2_pred / p2_pred[2]

                # Check if it's an inlier
                error = np.linalg.norm(p2[:2] - p2_pred[:2])
                if error < tau:
                    inliers.append(pair)

                # p1_prime = H.dot(np.append(p1, 1))[:2]
                #
                # if np.linalg.norm(p2 - p1_prime) < tau:
                #     inliers.append(pair)

            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                best_homography = H

            if len(max_inliers) > len(matches) * 0.95:
                break

        print(f"Found homography with {len(max_inliers)} inliers after {i+1} iterations")
        if len(max_inliers) < 8:
            return False, None, None

        pairs = [[keypoints1[pair.queryIdx], keypoints2[pair.trainIdx]] for pair in max_inliers]
        H = get_homography(pairs)
        # print("H", H)
        return True, H, max_inliers

    homography_set = {}
    inliers_set = {}

    # Initialize a list to store the cumulative homographies (key: (index, H)) encodes the homography H from 0 to key,
    # using index as the last image in the chain
    cumulative_homographies = {0: (0, np.eye(3))}  # Homography from 0 to 0 is the identity matrix

    for i, (im1, features1, desc1) in enumerate(zip(im_set[:-1], features_set[:-1], discriptors_set[:-1])):
        for j, (im2, features2, desc2) in enumerate(zip(im_set[i + 1:], features_set[i + 1:], discriptors_set[i + 1:]),
                                                    start=i + 1):

            matches = match_features(desc1, desc2)
            if len(matches) < 8:
                continue

            matches = np.array(matches)
            ret, H, inliers = RANSAC(features1, features2, matches)
            if not ret:
                continue
            
            print(f"Found homography from image {i} to {j}")

            homography_set[(i, j)] = H
            inliers_set[(i, j)] = inliers

            # Compute cumulative homography from 0 to j (i.e., compose from 0 to i and i to j)
            if i == 0:  # If we're at the first pair, just store the homography H
                cumulative_homographies[j] = (0, H)
            else:  # For subsequent pairs, multiply by the previous cumulative homography
                cumulative_homographies[j] = (i, cumulative_homographies[i][1] @ H)

            #  KeyPoints
            key1 = [cv2.KeyPoint(np.float32(x), np.float32(y), 1) for (x, y) in features1]
            key2 = [cv2.KeyPoint(np.float32(x), np.float32(y), 1) for (x, y) in features2]
            out = cv2.drawMatches(im1, key1, im2, key2, inliers, None)
            cv2.imwrite(f'outputs/{path}/ransac_{i}_{j}.png', out)


    def get_panorama_size(warped_images):
        max_height = max(img.shape[0] for img in warped_images)
        max_width = max(img.shape[1] for img in warped_images)
        return max_height, max_width

    def blend_with_mask(img1, img2, mask):
        """
        Blend images with transition mask
        """
        # Create distance transform for smooth transition
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)

        # Create mask
        blend_width = 50  # pixels
        alpha = np.clip(dist / blend_width, 0, 1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=10)

        # Blend images
        result = img1.copy()
        for c in range(3):  # For each color channel
            result[:, :, c] = img2[:, :, c] * alpha + img1[:, :, c] * (1 - alpha)

        return result

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    def panorama_blend(warped_images):
        result = warped_images[0].astype(np.float32) / 255.0

        for i in range(1, len(warped_images)):
            current = warped_images[i].astype(np.float32) / 255.0

            # Create masks for overlap region
            mask1 = (result.sum(axis=2) > 0).astype(np.uint8)
            mask2 = (current.sum(axis=2) > 0).astype(np.uint8)
            overlap = cv2.bitwise_and(mask1, mask2)

            # Blend in overlap region
            if overlap.sum() > 0:
                result = blend_with_mask(result, current, overlap)

            # Add non-overlapping regions
            non_overlap = cv2.bitwise_xor(mask2, overlap)
            result = np.where(non_overlap[:, :, np.newaxis] > 0, current, result)

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    print(list(cumulative_homographies.keys()))
    print(list(range(len(im_set))))
    if sorted(list(cumulative_homographies.keys())) != list(range(len(im_set))):
        raise Exception("Could not find homographies for all pairs. Exiting...")


    warped_images = []
    for i, image in enumerate(im_set):
        from_index, H = cumulative_homographies[i]
        h, w = image.shape[:2]
        new_h= h*3
        new_width= w*3

        # Translation matrix to center the image on the canvas
        translation_matrix = np.array([[1, 0, w], [0, 1, h], [0, 0, 1]])
        H_combined = translation_matrix @ np.linalg.inv(H)

        # Apply homography
        warped = cv2.warpPerspective(image, H_combined, (new_width, new_h))
        warped_images.append(warped)
        #cv2.imwrite(f'outputs/{path}/warped_{from_index}_{i}.png', warped)

        # Get maximum dimensions
        max_height, max_width = get_panorama_size(warped_images)

        # Resize all warped images to the same size
        warped_images = [cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_LINEAR)
                         if img.shape[:2] != (max_height, max_width) else img
                         for img in warped_images]
        cv2.imwrite(f'outputs/{path}/warped_{from_index}_{i}.png', warped)

    # # combine all the warped images
    # panorama = np.zeros_like(warped_images[0])
    # for warped in warped_images:
    #     panorama = cv2.add(panorama, warped)
    panorama = panorama_blend(warped_images)
    cv2.imwrite(f'outputs/{path}/mypano.png', panorama)
    #cv2.imwrite(f'outputs/{path}/mypano.png', panorama)





if __name__ == "__main__":
    main()

