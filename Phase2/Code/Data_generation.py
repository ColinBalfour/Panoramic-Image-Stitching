from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from numpy.linalg import inv
import os
train_path  = 'D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Train/Train'
test_path  = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Val/Val"
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
    rho = 16
    # top_left_corner = (32,32) #top_point
    # top_right_corner = (patch_size + 32, 32) #left_point
    # bottom_right_corner = (patch_size + 32, patch_size + 32) #bottom_point
    # bottom_left_corner = (32, patch_size + 32)  #right_point

    #Below with offset for translating better
   # Add a random translation amount (fixed for all 4 points) such that your network would work for translated images as well
    top_left_corner = (96, 56) #top_point
    top_right_corner = (patch_size + 96, 56) #left_point
    bottom_right_corner = (patch_size + 96, patch_size + 56) #bottom_point
    bottom_left_corner = (96, patch_size + 56)  #right_point

    # fixed point
    # Calculate safe translation bounds
    p = 8
    dx = random.uniform(-p,p)
    test_image = im.copy()


    C_A = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
    print("\nOriginal Corner Points (C_A):")
    print(f"Top Left: {top_left_corner}")
    print(f"Top Right: {top_right_corner}")
    print(f"Bottom Right: {bottom_right_corner}")
    print(f"Bottom Left: {bottom_left_corner}")
    C_B =[]
    for corners in C_A:
        perturbed = (corners[0]+ random.randint(-rho, rho) + dx  , corners[1]+ random.randint(-rho, rho) + dx )
        C_B.append(perturbed)
        print(f"Point {corners}: {perturbed}")
    print("\nPerturbed Points (C_B):", C_B)
    H_AB = cv2.getPerspectiveTransform(np.float32(C_A), np.float32(C_B))
    print("\nHomography Matrix H_AB:", H_AB)
    H_AB_inv = inv(H_AB)
    print("\nInverse Homography Matrix H_AB_inv:", H_AB_inv)
    # Save debugging visualizations
    debug_img = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR)

    # Draw original points in blue
    for pt in C_A:
        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

    # Draw perturbed points in red
    for pt in C_B:
        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    cv2.imwrite('debug_points.png', debug_img)
    warped_image = cv2.warpPerspective(im, H_AB_inv, (320, 240))
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
    print("\nH_four_points (C_B - C_A):")
    print(H_four_points)
    return train_image, H_four_points


def savedata(path, dir):
    # Verify directories exist
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training path not found: {path}")

    lst = os.listdir(path)
    save_dir = os.path.join("D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Test", 'TESTTRANS')
    save_txt = os.path.join(dir, 'Te.txt')
    #save_txt = os.path.join(dir, 'test_homography_trans.txt')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_coordinate = []
    #test_coordinate = []
    with open(save_txt, 'w') as f:
        for i, img in enumerate(lst):
            if img != 'synthetic_data' and os.path.isfile(os.path.join(path, img)):  # Check if it's a file
                try:
                    patch_pairs, homography = patch_pairs_generation(img, path)
                    data = {'img': patch_pairs, 'homography': homography}
                    train_coordinate.append(homography.tolist())
                    #test_coordinate.append(homography.tolist())
                    f.write(str(train_coordinate) + '\n')
                    #f.write(str(test_coordinate) + '\n')
                    np.save(os.path.join(save_dir, f'img{i + 1}'), data, allow_pickle=True)
                except Exception as e:
                    print(f"Error processing image {img}: {str(e)}")
                    continue


#savedata(train_path, dir)
savedata(test_path, dir)
