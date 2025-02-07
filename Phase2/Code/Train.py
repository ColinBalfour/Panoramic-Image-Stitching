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
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyNet
from torch.utils.data import DataLoader
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
#from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Network.Network import *
#from Network.Network import HomographyNet, LossFn
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
#from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import torch.optim as optim
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


def GenerateBatch(TrainSet,  MiniBatchSize, istest = False):
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
        RandIdx = random.randint(0, len(TrainSet) - 1)

        #RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"


        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        # I1 = np.float32(cv2.imread(RandImageName))
        # Coordinates = TrainCoordinates[RandIdx]
        I1 , Coordinates = TrainSet[RandIdx]
        print("shapeI1", I1.shape)
        # Append All Images and Mask
        I1Batch.append(I1)
        CoordinatesBatch.append(Coordinates)
        ImageNum += 1

    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(DirNamesTrain, NumTrainSamples, ImageSize,NumEpochs,MiniBatchSize,SaveCheckPoint,CheckPointPath,
    DivTrain,
    LatestFile,
    LogsPath,
    ModelType, TrainSet, TestSet
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyNet(InputSize=2*128*128, OutputSize=8).to(device)
    print("Model Device:", next(model.parameters()).device)
    # Parameters
    print("\nModel Parameter Summary:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} - {param.data.shape}")
            total_params += param.numel()
    print(f"\nTotal trainable parameters: {total_params:,}\n")

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=15, gamma=0.1)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")
    epoch_losses = []
    epoch_accuracies = []
    epoch_losses_test = []
    all_Train_Prediction =[]
    all_Train_Label =[]
    all_Test_Prediction =[]
    all_Test_Label =[]
    Training_time = tic()
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        total_loss = 0.0
        total_accuracy = 0.0
        #Train_Prediction = []
        # training loop
        model.train()
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            Batch = GenerateBatch(TrainSet, MiniBatchSize)
            I1Batch, CoordinatesBatch = Batch
            I1Batch = I1Batch.to(device)
            CoordinatesBatch = CoordinatesBatch.to(device)
            print("CoordinatesBatch", CoordinatesBatch)
            #I1Batch = I1Batch.permute(0, 3, 1, 2).float()
            print("I1Batch Shape: ", I1Batch.shape)
            print("Input Data Device:", I1Batch.device)
            #CoordinatesBatch = CoordinatesBatch.float()
            # Predict output with forward pass
            PredicatedCoordinatesBatch = model(I1Batch)
            print("PredicatedCoordinatesBatch: ", PredicatedCoordinatesBatch)
            LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)

            # Prediction of labels
            if Epochs == NumEpochs - 1:
                Pred = model(I1Batch)
                _, Predicted = torch.max(Pred.data, 1)

                # Storing Predictions and true labels
                all_Train_Prediction.extend(Predicted.cpu().numpy())
                all_Train_Label.extend(CoordinatesBatch.cpu().numpy())


            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Accumulate the loss for this epoch
            total_loss += LossThisBatch.item()
            print(f"LossThisBatch {LossThisBatch}, NumIterationsPerEpoch {NumIterationsPerEpoch}")

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            result = model.validation_step(Batch)
            total_accuracy += result["acc"]
            # Tensorboard
            Writer.add_scalar(
                "TrainingLoss", LossThisBatch.item(),
                #result["val_loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        # Save model every epoch

        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


        # Checking Testing Model Accuracy
        # Evaluate test accuracy
        test_loss = 0.0
        Test_Label = []
        Test_Prediction = []
        model.eval()
        with torch.no_grad(): # disable gradient
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                #Generate a batch
                TestBatch = GenerateBatch(TestSet, MiniBatchSize)
                I1Batch, CoordinatesBatch = TestBatch

                #GPU
                I1Batch = I1Batch.to(device)
                CoordinatesBatch = CoordinatesBatch.to(device)
                # Prediction of labels
                if Epochs == NumEpochs - 1:
                    Pred = model(I1Batch)
                    _, Predicted = torch.max(Pred.data, 1)

                    # Storing Predictions and true labels
                    all_Test_Prediction.extend(Predicted.cpu().numpy())
                    all_Test_Label.extend(CoordinatesBatch.cpu().numpy())

                result = model.validation_step(TestBatch)
                test_loss += result["loss"].item()


        # calculating training loss
        avg_loss = total_loss / NumIterationsPerEpoch
        avg_test_loss = test_loss / NumIterationsPerEpoch
        avg_accuracy = total_accuracy / NumIterationsPerEpoch
        epoch_losses_test.append(avg_test_loss)
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(avg_accuracy)
        print(f"Epoch {Epochs + 1}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
        print(f"Epoch {Epochs + 1}, Average Loss: {avg_test_loss:.4f}")
        scheduler.step()

    Training_Stop = toc(Training_time)
    print("Training Complete and training time: ", Training_Stop- Training_time)

    # Plot both testing and training loss
    plt.plot(range(NumEpochs), epoch_losses, label='Training loss')
    #plt.plot(range(NumEpochs), epoch_testing_losses, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Training loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(range(NumEpochs), epoch_losses_test, label='Testing loss')
    # plt.plot(range(NumEpochs), epoch_testing_losses, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Testing loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot both testing and training loss
    plt.plot(range(NumEpochs), epoch_losses, label='Training loss')
    plt.plot(range(NumEpochs), epoch_losses_test, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Training and Test loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(range(NumEpochs), epoch_accuracies, label='Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracies')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Train/TrainData",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Code/TxtFiles/Checkpointsfinal/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Code/TxtFiles/LogsFinalTrain/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        NumClasses,
    ) = SetupAll(CheckPointPath)
    # # Setup all needed parameters including file reading
    # (
    #     DirNamesTrain,
    #     SaveCheckPoint,
    #     ImageSize,
    #     NumTrainSamples,
    #     TrainCoordinates,
    #     NumClasses,
    # ) = SetupAll(CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    TrainSet = COCOCustom(root_dir=BasePath)
    TestSet = COCOCustom(root_dir= "D:/Computer vision/Homeworks/PH1_phase2/YourDirectoryID_p1/Phase2/Data/Val/TestDatafinal")

    #Train_dataloader = DataLoader(TrainSet, batch_size=64, shuffle=True)
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        LogsPath,
        ModelType, TrainSet, TestSet
    )


if __name__ == "__main__":
    main()
