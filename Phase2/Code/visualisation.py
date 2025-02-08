import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from Network import HomographyNet
import random
import os
from datetime import datetime
from skimage.feature import peak_local_max


def classical_homography_estimation(img1, img2):
    """
    Estimate homography using classical feature-based method
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Harris Corner Detection
    harris1 = cv2.cornerHarris(gray1, 5, 3, 0.04)
    harris2 = cv2.cornerHarris(gray2, 5, 3, 0.04)

    # Find local maxima
    features1 = peak_local_max(harris1, min_distance=2, threshold_rel=0.01)
    features2 = peak_local_max(harris2, min_distance=2, threshold_rel=0.01)

    # Convert to (x,y) format
    features1 = [(y, x) for x, y in features1]
    features2 = [(y, x) for x, y in features2]

    # Feature Description
    def get_patch(image, center, patch_size=(41, 41)):
        height, width = patch_size
        x, y = center
        top_left_x = max(0, x - width // 2)
        top_left_y = max(0, y - height // 2)
        bottom_right_x = min(image.shape[1], x + width // 2 + 1)
        bottom_right_y = min(image.shape[0], y + height // 2 + 1)
        patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        return patch

    # Get descriptors for both images
    desc1 = []
    desc2 = []

    for x, y in features1:
        patch = get_patch(gray1, (x, y))
        blur = cv2.GaussianBlur(patch, (5, 5), 0)
        subsampled = cv2.resize(blur, (8, 8))
        feature_vector = subsampled.flatten()
        normalized = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
        desc1.append(normalized)

    for x, y in features2:
        patch = get_patch(gray2, (x, y))
        blur = cv2.GaussianBlur(patch, (5, 5), 0)
        subsampled = cv2.resize(blur, (8, 8))
        feature_vector = subsampled.flatten()
        normalized = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
        desc2.append(normalized)

    # Feature Matching
    def match_features(descriptors1, descriptors2, threshold=0.75):
        matches = []
        for i, descriptor in enumerate(descriptors1):
            distances = [np.linalg.norm(descriptor - d) for d in descriptors2]
            sorted_indices = np.argsort(distances)
            if distances[sorted_indices[0]] < threshold * distances[sorted_indices[1]]:
                match = cv2.DMatch(i, sorted_indices[0], distances[sorted_indices[0]])
                matches.append(match)
        return matches

    # Get matches
    matches = match_features(desc1, desc2)

    # RANSAC
    def get_homography(pairs):
        n_pairs = len(pairs)
        A_matrix = np.zeros((2 * n_pairs, 9))
        for i, pair in enumerate(pairs):
            x1, y1 = pair[0]
            x2, y2 = pair[1]
            row1 = 2 * i
            row2 = 2 * i + 1
            A_matrix[row1] = [x1, y1, 1, 0, 0, 0, -(x2 * x1), -(x2 * y1), -x2]
            A_matrix[row2] = [0, 0, 0, x1, y1, 1, -(y2 * x1), -(y2 * y1), -y2]
        _, _, Vt = np.linalg.svd(A_matrix)
        h = Vt[-1]
        H = h.reshape(3, 3)
        H = (1 / H.item(8)) * H
        return H

    def RANSAC(keypoints1, keypoints2, matches, tau= 2, N=10000):
        max_inliers = []
        best_homography = None
        for i in range(N):
            random_pairs = np.random.choice(matches, 4, replace=True)
            pairs = [[keypoints1[pair.queryIdx], keypoints2[pair.trainIdx]] for pair in random_pairs]
            H = get_homography(pairs)

            inliers = []
            for pair in matches:
                p1 = np.array([keypoints1[pair.queryIdx][0], keypoints1[pair.queryIdx][1], 1])
                p2 = np.array([keypoints2[pair.trainIdx][0], keypoints2[pair.trainIdx][1], 1])
                p2_pred = H @ p1
                p2_pred = p2_pred / p2_pred[2]
                ssd = np.sum((p2[:2] - p2_pred[:2]) ** 2)
                if ssd < tau:
                    inliers.append(pair)

            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                best_homography = H

            if len(max_inliers) > len(matches) * 0.95:
                break

        if len(max_inliers) < 8:
            return False, None, None

        pairs = [[keypoints1[pair.queryIdx], keypoints2[pair.trainIdx]] for pair in max_inliers]
        H = get_homography(pairs)
        return True, H, max_inliers

    ret, H_classical, inliers = RANSAC(features1, features2, matches)
    if not ret:
        return None

    return H_classical


def visualize_homography_comparison(test_dir, model_path, output_dir, num_images=4):
    """
    Visualize and compare classical and deep learning homography estimates
    against ground truth for given images and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f'homography_comparison_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    def draw_corners(img, corners, color, thickness=2):
        corners = corners.astype(np.int32)
        for i in range(4):
            cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, thickness)
        return img

    # Load the trained model
    model = HomographyNet(InputSize=2 * 128 * 128, OutputSize=8)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get list of image files
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    selected_images = image_files[:num_images] if len(image_files) > num_images else image_files

    # Create results summary file
    summary_file = os.path.join(results_dir, 'comparison_summary.txt')

    with open(summary_file, 'w') as f:
        f.write(f"Homography Comparison Results\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model used: {model_path}\n\n")

        for idx, img_name in enumerate(selected_images):
            img_path = os.path.join(test_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (320, 240))
            original = img.copy()

            # Define original corners (same as in training)
            top_left = (96, 56)
            top_right = (96 + 128, 56)
            bottom_right = (96 + 128, 56 + 128)
            bottom_left = (96, 56 + 128)
            corners_A = np.float32([top_left, top_right, bottom_right, bottom_left])

            # Generate random perturbation for ground truth
            rho = 32
            corners_B = []
            for corner in corners_A:
                new_corner = (
                    corner[0] + random.randint(-rho, rho),
                    corner[1] + random.randint(-rho, rho)
                )
                corners_B.append(new_corner)
            corners_B = np.float32(corners_B)

            # Get ground truth homography
            H_gt = cv2.getPerspectiveTransform(corners_B, corners_A)

            # Get classical estimation
            warped = cv2.warpPerspective(img, np.linalg.inv(H_gt), (320, 240))
            H_classical = classical_homography_estimation(img, warped)

            # Deep learning prediction
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            patch1 = gray[top_left[1]:bottom_left[1], top_left[0]:top_right[0]]
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            patch2 = warped_gray[top_left[1]:bottom_left[1], top_left[0]:top_right[0]]
            stacked_patches = np.dstack((patch1, patch2))
            stacked_patches = (stacked_patches.astype(np.float32) - 127.5) / 127.5
            x = torch.from_numpy(stacked_patches).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                pred_h = model(x).squeeze().cpu().numpy()
           # pred_h = pred_h * 32.0  # Denormalize
            deltas = pred_h.reshape(-1) * 32  # Scale back by 32
            print("deltas", deltas)
            corners_pred = corners_A + deltas.reshape(4, 2)

            # Convert predicted 4-point parameterization to homography matrix
            #corners_pred = corners_A + pred_h.reshape(-1, 2)
            H_deep = cv2.getPerspectiveTransform(corners_pred, corners_A)

            # Visualize results
            vis_img = original.copy()

            # Draw ground truth (red)
            draw_corners(vis_img, corners_B, (0, 0, 255))

            # Draw deep learning prediction (yellow)
            if H_deep is not None:
                draw_corners(vis_img, corners_pred, (0, 255, 255))

            # Draw classical prediction (green)
            if H_classical is not None:
                warped_corners = cv2.perspectiveTransform(corners_A.reshape(-1, 1, 2), H_classical)
                draw_corners(vis_img, warped_corners.reshape(-1, 2), (0, 255, 0))

            # Save visualization
            output_path = os.path.join(results_dir, f'comparison_{img_name}')
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Image {idx + 1} Homography Comparison\n'
                      'Red: Ground Truth, Yellow: Deep Learning, Green: Classical')
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Save visualization as OpenCV image too
            cv2.imwrite(output_path.replace('.jpg', '_cv.jpg'), vis_img)

            # Write results to summary file
            f.write(f"\nResults for image: {img_name}\n")
            f.write(f"Ground Truth Homography:\n{H_gt}\n")
            if H_classical is not None:
                f.write(f"Classical Estimation:\n{H_classical}\n")
            f.write(f"Deep Learning Prediction:\n{H_deep}\n")
            f.write("-" * 50 + "\n")

    print(f"Results saved to: {results_dir}")
    return results_dir


# Example usage
if __name__ == "__main__":
    # test_dir = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Val/Val"
    # model_path = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Code/TxtFiles/CheckpointsTrainwithouttransfinal/49model.ckpt"
    # output_dir = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Results/HomographyComparison"
    
    test_dir = "Phase2/Data/Val"
    model_path = "Phase2/Code/TxtFiles/CheckpointsTRANS/99model.ckpt"
    output_dir = "Phase2/Results/HomographyComparison"

    results_dir = visualize_homography_comparison(test_dir, model_path, output_dir)