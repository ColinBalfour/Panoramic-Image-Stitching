#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
#from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
#from Network.Network import HomographyModel
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Network.Network import *


# Don't generate pyc codes
sys.dont_write_bytecode = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class COCOCustom(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform = None ): #, labels_path, transform=None, dataset_type="train"):
        """
        Args:
            root_dir: Directory with all images (1 to 50000)
            labels_path: Path to LabelsTrain.txt
            transform: Optional transform to be applied
        """
        self.root_dir = root_dir
        self.transform = transform
        # X = ()
        # Y = ()
        X =[]
        Y =[]
        # Get list of all .npy files
        self.data = sorted([f for f in os.listdir(root_dir) if f.endswith('.npy')])
        i = 0
        train_coordinates = []

        # Load each file
        for im in self.data:
            file_path= os.path.join(self.root_dir, im)
            output = np.load(file_path, allow_pickle=True).item()
            # Get image patches and normalize
            x = torch.from_numpy((output['img'].astype(np.float32) - 127.5) / 127.5)
            x = x.permute(2,0,1).float().to(device)
            #print("xshape", x.shape)
            #X= X + (x,)
            X.append(x)

            # Get homography and normalize
            y = torch.from_numpy(output['homography'].astype(np.float32) / 32.0)
            y = y.float().to(device)

            # Y = Y + (y,)
            Y.append(y)

            i = i + 1
        self.len  = i
        self.X_data = X
        self.Y_data = Y

    def __len__(self):
        """Return total number of samples"""
        #return len(self.X_data)
        return self.len

    def __getitem__(self, index):
        """Get a single data pair"""
        x = self.X_data[index]
        print("xshape2", x.shape)
        y = self.Y_data[index]
        if self.transform:
            x = self.transform(x)

        return x, y

def GenerateBatch(TestSet,  MiniBatchSize, istest = False):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TestSet) - 1)

        #RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"


        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        # I1 = np.float32(cv2.imread(RandImageName))
        # Coordinates = TrainCoordinates[RandIdx]
        I1 , Coordinates = TestSet[RandIdx]
        print("shapeI1", I1.shape)
        # Append All Images and Mask
        I1Batch.append(I1)
        CoordinatesBatch.append(Coordinates)
        ImageNum += 1

    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device)




def TestOperation(NumEpochs,
        MiniBatchSize,
        DivTest,
        LogsPath,
        ModelType, TestSet, ModelPath):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = HomographyNet(InputSize=2 * 128 * 128, OutputSize=8).to(device)
    print("Model Device:", next(model.parameters()).device)

    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )

    model.eval()
    StartEpoch = 0
    # NumEpochs = 50
    # MiniBatchSize = 64
    # DivTest = 1
    epoch_losses_test = []
    epoch_epe_test = []
    forward_pass_times = []
    NumTestSamples = len(TestSet)
    Testing_Start = tic()
    all_Test_Prediction = []
    all_Test_Label = []
    model.eval()
    with torch.no_grad():
        for Epochs in tqdm(range(NumEpochs)):
            NumIterationsPerEpoch = int(NumTestSamples / MiniBatchSize / DivTest)
            test_loss = 0.0
            total_epe_test = 0.0

            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                # Generate and process batch
                I1Batch, CoordinatesBatch = GenerateBatch(TestSet, MiniBatchSize)
                start_time = time.time()
                PredicatedCoordinatesBatch = model(I1Batch)
                forward_time = time.time() - start_time
                forward_pass_times.append(forward_time)
                print("PredicatedCoordinatesBatch: ", PredicatedCoordinatesBatch)

                # EPE Calculation
                pred_denorm = PredicatedCoordinatesBatch * 32.0
                gt_denorm = CoordinatesBatch * 32.0
                epe_test = EPE(pred_denorm, gt_denorm)
                total_epe_test += epe_test.item()

                # Forward pass
                result = model.validation_step((I1Batch, CoordinatesBatch))
                test_loss += result["loss"].item()

            # Calculate average loss for the epoch
            avg_loss = test_loss / NumIterationsPerEpoch
            avg_epe_test = total_epe_test / NumIterationsPerEpoch
            epoch_epe_test.append(avg_epe_test)
            avg_forward_time = np.mean(forward_pass_times) * 1000
            epoch_losses_test.append(avg_loss)
            print(f"Epoch {Epochs + 1}, Average TestLoss: {avg_loss:.4f}")
            print(f"Average TESTING EPE: {avg_epe_test:.4f} pixels")
            print(f"Average Forward Pass Time: {avg_forward_time:.2f} ms")
    # Plot the loss over epochs after training
    plt.plot(range(NumEpochs), epoch_losses_test, label='Testloss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.show()

    # Plot EPE curves
    plt.plot(range(len(epoch_epe_test)), epoch_epe_test, label='Test EPE')
    plt.xlabel('Epochs')
    plt.ylabel('EPE (pixels)')
    plt.title('EPE Over Testing Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LogsPath, 'epe_curves_test.png'))
    plt.close()


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Code/TxtFiles/CheckpointsTRANS/99model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="/home/chahatdeep/Downloads/aa/CMSC733HW0/CIFAR10/Test/",
        help="Path to load images from, Default:BasePath",
    )

    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=100,
        help="Number of Epochs to Train for, Default:50",
    )

    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",)

    Parser.add_argument(
        "--DivTest",
        type=int,
        default=1,
        help="Factor to reduce Test data by per epoch, Default:1",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )

    Parser.add_argument(
        "--LogsPath",
        default="D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Code/TxtFiles/Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    MiniBatchSize = Args.MiniBatchSize
    NumEpochs = Args.NumEpochs
    DivTest = Args.DivTest
    ModelType = Args.ModelType
    LogsPath = Args.LogsPath


    TestSet = COCOCustom(root_dir="D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Val/TESTTRANS")
    TestOperation(
        NumEpochs,
        MiniBatchSize,
        DivTest,
        LogsPath,
        ModelType, TestSet, ModelPath
    )

if __name__ == "__main__":
    main()
