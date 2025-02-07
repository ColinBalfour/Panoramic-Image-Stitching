"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
#import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True

def accuracy(out, target, threshold =0.1):
    target = target.view(-1,8)
    abs_diff = torch.abs(out - target)
    correct = (abs_diff <= threshold).all(dim=1)
    accuracy = (correct.float().mean() * 100).item()
    # Calculate relative error
    # relative_error = torch.norm(out - target, dim=1) / (torch.norm(target, dim=1) + 1e-8)
    #
    # # Consider prediction "correct" if error below threshold
    # accuracy = torch.mean((relative_error < threshold).float()) * 100

    return accuracy


def EPE(predicted_points, ground_truth_points):
    """Calculate EPE between predicted and ground truth points"""
    # Reshape to (batch_size, 4, 2) for point-wise comparison
    pred_points = predicted_points.reshape(-1, 4, 2)
    gt_points = ground_truth_points.reshape(-1, 4, 2)

    # Calculate L2 distance for each point
    point_distances = torch.sqrt(torch.sum((pred_points - gt_points) ** 2, dim=2))

    # Average over points and batch
    return torch.mean(point_distances)

def LossFn(out, target):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    # loss = torch.norm(out - target, p=2, dim=1)
    criterion = nn.MSELoss()
    loss = criterion(out, target.view(-1, 8))
    return loss



# class HomographyModel(pl.LightningModule):
#     def __init__(self, hparams):
#         super(HomographyModel, self).__init__()
#         self.hparams = hparams
#         self.model = HomographyNet()
#
#     def forward(self, a, b):
#         return self.model(a, b)
#
#     def training_step(self, batch, batch_idx):
#         img_a, patch_a, patch_b, corners, gt = batch
#         delta = self.model(patch_a, patch_b)
#         loss = LossFn(delta, img_a, patch_b, corners)
#         logs = {"loss": loss}
#         return {"loss": loss, "log": logs}
#
#     def validation_step(self, batch, batch_idx):
#         img_a, patch_a, patch_b, corners, gt = batch
#         delta = self.model(patch_a, patch_b)
#         loss = LossFn(delta, img_a, patch_b, corners)
#         return {"val_loss": loss}
#
#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
#         logs = {"val_loss": avg_loss}
#         return {"avg_val_loss": avg_loss, "log": logs}
class HomographyBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = LossFn(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        I1Batch, CoordinatesBatch = batch
        out = self(I1Batch)                    # Generate predictions
        loss = LossFn(out, CoordinatesBatch)   # Calculate loss
        acc = accuracy(out, CoordinatesBatch)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))




class HomographyNet(HomographyBase):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input - 128 * 128 * 2
        OutputSize - Size of the Output - 8
        """
        super(HomographyNet,self).__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.layer1 = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())#, nn.Dropout(p=0.5))
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 8)

        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
        # self.localization = nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=5),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        # )
        #
        # # Regressor for the 3 * 2 affine matrix
        # self.fc_loc = nn.Sequential(
        #     nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        # )
        #
        # # Initialize the weights/bias with identity transformation
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(
        #     torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        #)

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    # def stn(self, x):
    #     "Spatial transformer network forward function"
    #     xs = self.localization(x)
    #     xs = xs.view(-1, 10 * 3 * 3)
    #     theta = self.fc_loc(xs)
    #     theta = theta.view(-1, 2, 3)
    #
    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)
    #
    #     return x

    #def forward(self, xa, xb):
    def forward(self, x):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(out.size(0), -1)
        #out = out.view(128 * 16 * 16, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        # Fill your network structure of choice here!
        #############################
        return out

