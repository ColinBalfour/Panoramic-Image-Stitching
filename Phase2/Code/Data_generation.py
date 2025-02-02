from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from numpy.linalg import inv
import os
train_path  = 'D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Train/Train'
test_path  = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Test/Test"
dir = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Code/TxtFiles"
def patch_pairs_generation(image, path):
    #full_path  = os.path.join(path, image)
    #print(full_path)
    im = cv2.imread(path+'/%s'%image,cv2.IMREAD_GRAYSCALE)
    print('path',path+'/%s'%image)
    #im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (320, 240))
    patch_size = 128
    h, w = im.shape
    rho = 32
    top_left_corner = (32,32) #top_point
    top_right_corner = (patch_size + 32, 32) #left_point
    bottom_right_corner = (patch_size + 32, patch_size + 32) #bottom_point
    bottom_left_corner = (32, patch_size + 32)  #right_point

    # Below with offset for translating better
    # Add a random translation amount (fixed for all 4 points) such that your network would work for translated images as well.
    # top_left_corner = (64, 64) #top_point
    # top_right_corner = (patch_size + 64, 64) #left_point
    # bottom_right_corner = (patch_size + 64, patch_size + 64) #bottom_point
    # bottom_left_corner = (64, patch_size + 64)  #right_point

    # fixed point
    # Calculate safe translation bounds
    # dx = 24 # ~48 pixels
    # dy = 24  # ~28 pixels

    # Apply translation to base coordinates
    top_left_corner = (int(top_left_corner[0] + dx), int(top_left_corner[1] + dy))
    top_right_corner = (int(top_right_corner[0] + dx), int(top_right_corner[1] + dy))
    bottom_right_corner = (int(bottom_right_corner[0] + dx), int(bottom_right_corner[1] + dy))
    bottom_left_corner = (int(bottom_left_corner[0] + dx), int(bottom_left_corner[1] + dy))

    test_image = im.copy()


    C_A = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
    C_B =[]
    for corners in C_A:
        C_B.append((corners[0]+ random.randint(-rho, rho), corners[1]+ random.randint(-rho, rho)))
    H_AB = cv2.getPerspectiveTransform(np.float32(C_B), np.float32(C_A))
    H_AB_inv = inv(H_AB)
    warped_image = cv2.warpPerspective(im, H_AB_inv, (320, 240))
    warped_copy = warped_image.copy()
    # Ip1 = test_image[top_left_corner[0]:top_right_corner[0], top_left_corner[1]:top_right_corner[1]]
    # Ip2 = warped_image[top_left_corner[0]:top_right_corner[0], top_left_corner[1]:top_right_corner[1]]
    # Ip1 = test_image[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[1]]
    #
    #
    # Ip2 = warped_image[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]
    # Fixed patch extraction - corrected coordinates
    Ip1 = test_image[top_left_corner[1]:bottom_left_corner[1],
          top_left_corner[0]:top_right_corner[0]]

    Ip2 = warped_image[top_left_corner[1]:bottom_left_corner[1],
          top_left_corner[0]:top_right_corner[0]]
    cv2.imshow("patch1", Ip1)
    cv2.imshow("patch2", Ip1)
    cv2.waitKey(1)

    train_image = np.dstack((Ip1, Ip2))
    train_image = train_image.astype(np.float32)
    H_four_points =  np.subtract(np.array(C_B), np.array(C_A))
    H_four_points = H_four_points.astype(np.float32)
   # data = (train_image,H_four_points)
    return train_image, H_four_points


def savedata(path ,dir):
    lst = os.listdir(path + '/')
    save_dir = path + '/' + 'data/'
    save_txt = dir + '/' + 'train_homography.txt'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_coordinate =[]
    with open(save_txt, 'w') as f:
        for i, img in enumerate(lst):
            if img != 'data':  # Skip the output directory
                patch_pairs, homography = patch_pairs_generation(img, path)
                data = {'img': patch_pairs, 'homography': homography}
                train_coordinate.append(homography.tolist())
                f.write(str(train_coordinate) + '\n')
                # Save with allow_pickle=True
                np.save(save_dir + f'img{i+1}', data, allow_pickle=True)


savedata(train_path, dir)
