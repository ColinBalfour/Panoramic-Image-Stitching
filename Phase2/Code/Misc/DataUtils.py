"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import cv2
import numpy as np
import random
import skimage
import PIL
from numpy.linalg import inv
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

# train_path  = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Train/Train"
# test_path  = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Test/Test"
# def patch_pairs_generation(image, path):
#     im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     im = cv2.resize(im, (320, 240))
#     patch_size = 128
#     h, w = im.shape
#     rho = 32
#     top_left_corner = (32,32) #top_point
#     top_right_corner = (patch_size + 32, 32) #left_point
#     bottom_right_corner = (patch_size + 32, patch_size + 32) #bottom_point
#     bottom_left_corner = (32, patch_size + 32)  #right_point
#     test_image = im.copy()
#     C_A = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
#     C_B =[]
#     for corners in C_A:
#         C_B.append((corners[0]+ random.randint(-rho, rho), corners[1]+ random.randint(-rho, rho)))
#     H_AB= cv2.getPerspectiveTransform(np.float32(C_B), np.float32(C_A))
#     H_AB_inv = inv(H_AB)
#     warped_image = cv2.warpPerspective(image, H_AB_inv, (320, 240))
#     warped_copy = warped_image.copy()
#     Ip1 = test_image[top_left_corner[0]:top_right_corner[0], top_left_corner[1]:top_right_corner[1]]
#     Ip2 = warped_image[top_left_corner[0]:top_right_corner[0], top_left_corner[1]:top_right_corner[1]]
#     train_image = np.dstack((Ip1, Ip2))
#     H_four_points =  np.subtract(np.array(C_B), np.array(C_A))
#     data = (train_image,H_four_points)
#
#
#     return data
#
#
# def savedata(path):
#     lst = os.listdir(path + '/')
#     os.makedirs(path + 'processed/')
#     new_path = path + 'processed/'
#     for i in lst:
#         np.save(new_path + '%s' % i[0:12],patch_pairs_generation(i, path))


#savedata(train_path)
#savedata(validation_path)
#savedata(test_path)

def SetupAll(CheckPointPath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # # Setup all needed parameters including file reading
    # (
    #     DirNamesTrain,
    #     SaveCheckPoint,
    #     ImageSize,
    #     NumTrainSamples,
    #     TrainCoordinates,
    #     NumClasses,
    # ) = SetupAll(BasePath, CheckPointPath)
    Path = "Phase2/Code/TxtFiles"
    # Path = "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Code/TxtFiles"
    # Setup DirNames
    DirNamesTrain = SetupDirNames(Path)

    # Read and Setup Labels
    # LabelsPathTrain = './TxtFiles/LabelsTrain.txt'
    # TrainLabels = ReadLabels(LabelsPathTrain)

    # If CheckPointPath doesn't exist make the path
    if not (os.path.isdir(CheckPointPath)):
        os.makedirs(CheckPointPath)

    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100
    # Number of passes of Val data with MiniBatchSize
    NumTestRunsPerEpoch = 5

    # Image Input Shape
    ImageSize = [128, 128, 2]
    NumTrainSamples = len(DirNamesTrain)

    # Number of classes
    NumClasses = 8

    return (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        NumClasses,
    )


def ReadLabels(LabelsPathTrain):
    if not (os.path.isfile(LabelsPathTrain)):
        print("ERROR: Train Labels do not exist in " + LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = open(LabelsPathTrain, "r")
        TrainLabels = TrainLabels.read()
        TrainLabels = list(map(float, TrainLabels.split()))

    return TrainLabels


def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesTrain = ReadDirNames(BasePath + "/DirNamesTrain.txt")

    return DirNamesTrain


def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, "r")
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames
