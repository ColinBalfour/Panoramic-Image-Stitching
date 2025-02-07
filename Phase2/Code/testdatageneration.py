from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from numpy.linalg import inv
import os


def patch_pairs_generation(image, path, is_train=True):
    im = cv2.imread(path + '/%s' % image, cv2.IMREAD_GRAYSCALE)
    print('path', path + '/%s' % image)

    if is_train:
        im = cv2.resize(im, (320, 240))
        patch_size = 128
        rho = 32
        p = 8
        dx = random.uniform(-p, p)

    else:
        im = cv2.resize(im, (640, 480))
        patch_size = 256
        rho = 64
        p = 8
        dx = random.uniform(-p, p)


    # Calculate offsets based on image size
    if is_train:
        x_offset = 96
        y_offset = 56
    else:
        x_offset = (im.shape[1] - patch_size) // 2
        y_offset = (im.shape[0] - patch_size) // 2

    top_left_corner = (x_offset, y_offset)
    top_right_corner = (patch_size + x_offset, y_offset)
    bottom_right_corner = (patch_size + x_offset, patch_size + y_offset)
    bottom_left_corner = (x_offset, patch_size + y_offset)

    test_image = im.copy()
    C_A = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
    C_B = []

    for corners in C_A:
        C_B.append((corners[0] + random.randint(-rho, rho) + dx,
                    corners[1] + random.randint(-rho, rho) + dx))

    H_AB = cv2.getPerspectiveTransform(np.float32(C_B), np.float32(C_A))
    H_AB_inv = inv(H_AB)
    warped_image = cv2.warpPerspective(im, H_AB_inv, im.shape[::-1])

    Ip1 = test_image[top_left_corner[1]:bottom_left_corner[1],
          top_left_corner[0]:top_right_corner[0]]
    Ip2 = warped_image[top_left_corner[1]:bottom_left_corner[1],
          top_left_corner[0]:top_right_corner[0]]

    # For test data, resize patches to 128x128
    if not is_train:
        Ip1 = cv2.resize(Ip1, (128, 128))
        Ip2 = cv2.resize(Ip2, (128, 128))

    cv2.imshow("patch1", Ip1)
    cv2.imshow("patch2", Ip2)
    cv2.waitKey(1)

    train_image = np.dstack((Ip1, Ip2))
    train_image = train_image.astype(np.float32)
    H_four_points = np.subtract(np.array(C_B), np.array(C_A))

    # Scale homography points for test data
    if not is_train:
        H_four_points = H_four_points * (128 / 256)

    H_four_points = H_four_points.astype(np.float32)
    return train_image, H_four_points


def savedata(path, dir, is_train=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    lst = os.listdir(path)

    if is_train:
        save_dir = os.path.join("D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Train",
                                'NO')
        save_txt = os.path.join(dir, 'train_homography.txt')
    else:
        save_dir = os.path.join("D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Val",
                                'POLK')

        save_txt = os.path.join(dir, 'TestDataHomography.txt')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    coordinates = []
    with open(save_txt, 'w') as f:
        for i, img in enumerate(lst):
            if img != 'synthetic_data' and os.path.isfile(os.path.join(path, img)):
                try:
                    patch_pairs, homography = patch_pairs_generation(img, path, is_train)
                    data = {'img': patch_pairs, 'homography': homography}
                    coordinates.append(homography.tolist())
                    f.write(str(coordinates) + '\n')
                    np.save(os.path.join(save_dir, f'img{i + 1}'), data, allow_pickle=True)
                except Exception as e:
                    print(f"Error processing image {img}: {str(e)}")
                    continue


# Paths
train_path = 'D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Train/Train'
test_path = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Val/Val"
dir = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Code/TxtFiles"

# Generate train data
#savedata(train_path, dir, is_train=True)

# Generate test data
savedata(test_path, dir, is_train=False)