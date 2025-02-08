import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DSACPatchLayer(nn.Module):
    def __init__(self, patch_size=128, num_hypotheses=256, inlier_threshold=10.0):
        super(DSACPatchLayer, self).__init__()
        self.patch_size = patch_size
        self.num_hypotheses = num_hypotheses
        self.inlier_threshold = inlier_threshold

        # Generate better distributed base points - corners and interior
        self.base_points = torch.tensor([
            [0, 0],  # Top-left
            [patch_size, 0],  # Top-right
            [patch_size, patch_size],  # Bottom-right
            [0, patch_size],  # Bottom-left
            [patch_size // 2, patch_size // 2]  # Center point
        ], dtype=torch.float32)

    def generate_hypotheses(self, batch_size, device):
        """Generate multiple homography hypotheses using sampling"""
        all_hypotheses = []

        for b in range(batch_size):
            batch_hypotheses = []
            for _ in range(self.num_hypotheses):
                # Randomly select 4 points from the base points
                indices = torch.randperm(len(self.base_points))[:4]
                src_points = self.base_points[indices].to(device)

                # Add random perturbations with learnable noise scale
                noise = torch.randn_like(src_points) * 8.0  # Reduced noise scale
                dst_points = src_points + noise

                # Calculate homography using DLT
                H = self.compute_homography(src_points, dst_points)

                # Normalize homography
                H = H / (H[2, 2] + 1e-8)
                batch_hypotheses.append(H)

            all_hypotheses.append(torch.stack(batch_hypotheses))

        return torch.stack(all_hypotheses)

    # Add this method to your DSACPatchLayer class
    def compute_homography(self, src_pts, dst_pts):
        """Compute homography matrix from point correspondences using DLT"""
        if src_pts.shape[0] != 4 or dst_pts.shape[0] != 4:
            raise ValueError("Exactly 4 point correspondences required")

        # Create the A matrix for DLT
        A = torch.zeros((8, 9), device=src_pts.device)

        for i in range(4):
            x, y = src_pts[i]
            u, v = dst_pts[i]
            A[i * 2] = torch.tensor([x, y, 1, 0, 0, 0, -u * x, -u * y, -u], device=src_pts.device)
            A[i * 2 + 1] = torch.tensor([0, 0, 0, x, y, 1, -v * x, -v * y, -v], device=src_pts.device)

        # Solve for homography using SVD
        try:
            U, S, V = torch.svd(A)
            # Get the last column of V as the solution
            H = V[:, -1].reshape(3, 3)
            # Normalize the homography matrix
            H = H / (H[2, 2] + 1e-8)
            return H
        except Exception as e:
            print(f"SVD computation failed: {e}")
            # Return identity matrix as fallback
            return torch.eye(3, device=src_pts.device)

    def score_hypothesis(self, H, patch1, patch2):
        """Score hypothesis based on patch similarity after warping"""
        # Convert H to affine transformation matrix for grid_sample
        theta = torch.zeros(2, 3, device=H.device)
        theta[:2, :2] = H[:2, :2]
        theta[:2, 2] = H[:2, 2]

        # Create sampling grid
        grid = F.affine_grid(theta.unsqueeze(0),
                             patch2.unsqueeze(0).size(),
                             align_corners=True)

        # Warp patch1 using the homography
        warped = F.grid_sample(patch1.unsqueeze(0),
                               grid,
                               align_corners=True)[0]

        # Calculate score using structural similarity
        # Negative because higher similarity should mean higher score
        diff = torch.abs(warped - patch2)
        score = -torch.mean(diff)

        return score

    def forward(self, patch_pairs):
        """Forward pass with probabilistic hypothesis selection"""
        batch_size = patch_pairs.size(0)
        device = patch_pairs.device

        # Split input patches
        patch1 = patch_pairs[:, 0].unsqueeze(1)  # Add channel dim
        patch2 = patch_pairs[:, 1].unsqueeze(1)

        # Generate and score hypotheses
        hypotheses = self.generate_hypotheses(batch_size, device)
        scores = []

        for b in range(batch_size):
            batch_scores = []
            for h in range(self.num_hypotheses):
                score = self.score_hypothesis(
                    hypotheses[b, h],
                    patch1[b],
                    patch2[b]
                )
                batch_scores.append(score)
            scores.append(torch.stack(batch_scores))

        scores = torch.stack(scores)

        # Convert scores to probabilities
        probabilities = F.softmax(scores, dim=1)

        if self.training:
            # During training: probabilistic selection
            selected_idx = torch.multinomial(probabilities, 1).squeeze()
            selected_H = hypotheses[torch.arange(batch_size), selected_idx]
        else:
            # During inference: select highest probability
            selected_idx = torch.argmax(probabilities, dim=1)
            selected_H = hypotheses[torch.arange(batch_size), selected_idx]

        return selected_H, probabilities


def dsac_homography_loss(predicted_H, gt_H, probabilities, hypotheses):
    """
    DSAC loss function for homography estimation

    Args:
        predicted_H: Selected homography matrix
        gt_H: Ground truth homography matrix
        probabilities: Selection probabilities for hypotheses
        hypotheses: All generated hypotheses
    Returns:
        Combined loss value
    """
    batch_size = predicted_H.size(0)

    # Calculate error for each hypothesis
    errors = []
    for i in range(hypotheses.size(1)):
        # Compute Frobenius norm between hypothesis and ground truth
        error = torch.norm(hypotheses[:, i] - gt_H.view(-1, 3, 3), dim=(1, 2))
        errors.append(error)
    errors = torch.stack(errors, dim=1)

    # Expected loss
    expected_loss = torch.sum(probabilities * errors, dim=1).mean()

    # Direct loss for selected hypothesis
    direct_loss = torch.norm(predicted_H - gt_H.view(-1, 3, 3), dim=(1, 2)).mean()

    # Symmetric transfer error (optional but recommended)
    def transfer_error(H1, H2):
        # Sample points to measure transfer error
        points = torch.tensor([[0, 0], [128, 0], [128, 128], [0, 128]],
                              device=H1.device, dtype=H1.dtype)

        def transform_points(H, pts):
            # Convert to homogeneous coordinates
            ones = torch.ones(pts.size(0), 1, device=pts.device)
            homog_pts = torch.cat([pts, ones], dim=1)

            # Transform points
            transformed = torch.matmul(H, homog_pts.t())
            # Convert back to inhomogeneous coordinates
            transformed = transformed[:2] / (transformed[2] + 1e-8)
            return transformed.t()

        # Compute bi-directional transfer error
        forward_error = torch.norm(transform_points(H1, points) -
                                   transform_points(H2, points), dim=1).mean()
        backward_error = torch.norm(transform_points(torch.inverse(H1), points) -
                                    transform_points(torch.inverse(H2), points), dim=1).mean()
        return (forward_error + backward_error) / 2.0

    transfer_loss = torch.stack([transfer_error(predicted_H[i], gt_H[i])
                                 for i in range(batch_size)]).mean()

    # Combine losses with weights
    total_loss = (direct_loss +
                  0.5 * expected_loss +
                  0.1 * transfer_loss)

    return total_loss

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


def LossFn(out, target, hypotheses=None, probabilities=None):
    """
    Loss function supporting both standard and DSAC approaches

    Args:
        out: Model predictions
        target: Ground truth homography
        hypotheses: Optional set of generated hypotheses (for DSAC)
        probabilities: Optional probabilities for hypotheses (for DSAC)
    """
    # If not using DSAC, fall back to standard MSE loss
    if hypotheses is None or probabilities is None:
        criterion = nn.MSELoss()
        loss = criterion(out, target.view(-1, 8))
        return loss

    # DSAC loss calculation

    errors = []
    for i in range(hypotheses.size(1)):
        error = torch.norm(hypotheses[:, i] - target.view(-1, 3, 3), dim=(1, 2))
        errors.append(error)
    errors = torch.stack(errors, dim=1)

    # Expected loss
    expected_loss = torch.sum(probabilities * errors, dim=1).mean()

    # Direct loss for selected hypothesis
    direct_loss = torch.norm(out - target.view(-1, 3, 3), dim=(1, 2)).mean()

    # Combine losses
    total_loss = direct_loss + 0.5 * expected_loss
    return total_loss

class HomographyBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        cnn_out, dsac_H, probabilities = self(images)

        # Calculate combined loss using the modified LossFn
        loss = LossFn(
            out=dsac_H,  # Use DSAC hypothesis
            target=labels,
            hypotheses=self.dsac.generate_hypotheses(images.size(0), images.device),
            probabilities=probabilities
        )
        return loss

    def validation_step(self, batch):
        images, labels = batch
        cnn_out, dsac_H, probabilities = self(images)

        # Calculate combined loss using the modified LossFn
        loss = LossFn(
            out=dsac_H,  # Use DSAC hypothesis
            target=labels,
            hypotheses=self.dsac.generate_hypotheses(images.size(0), images.device),
            probabilities=probabilities
        )
        return loss

    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))


class HomographyNetWithDSAC(HomographyBase):
    def __init__(self, InputSize, OutputSize):
        super(HomographyNetWithDSAC, self).__init__()

        # Keep your original feature extraction layers
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
                                    nn.ReLU())

        # Add DSAC layer for patch-based homography estimation
        self.dsac = DSACPatchLayer(patch_size=128, num_hypotheses=256)

        # Keep original fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        # Original CNN feature extraction
        features = self.layer1(x)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.layer5(features)
        features = self.layer6(features)
        features = self.layer7(features)
        features = self.layer8(features)

        # CNN prediction path
        cnn_out = features.view(features.size(0), -1)
        cnn_out = self.fc1(cnn_out)
        cnn_out = self.fc2(cnn_out)

        # DSAC path
        dsac_H, probabilities = self.dsac(x)

        if self.training:
            return cnn_out, dsac_H, probabilities
        else:
            # During inference, combine CNN and DSAC predictions
            cnn_H = self.params_to_homography(cnn_out)
            # Take weighted average of both predictions
            final_H = 0.5 * (cnn_H + dsac_H)
            return final_H

    def params_to_homography(self, params):
        """Convert 8-parameter output to 3x3 homography matrix"""
        batch_size = params.size(0)
        H = torch.zeros(batch_size, 3, 3, device=params.device)
        H[:, :2, :] = params.view(-1, 2, 4)
        H[:, 2, 2] = 1
        return H

    def training_step(self, batch):
        images, labels = batch
        cnn_out, dsac_H, probabilities = self(images)

        # Calculate combined loss
        loss = dsac_patch_loss(
            predicted_H=dsac_H,
            gt_H=labels,
            probabilities=probabilities,
            hypotheses=self.dsac.generate_hypotheses(images.size(0), images.device)
        )
        return loss

    def validation_step(self, batch):
        images, labels = batch
        H_pred = self(images)

        # Convert predictions and labels to consistent format
        if isinstance(H_pred, tuple):
            H_pred = H_pred[0]  # Take CNN prediction during validation

        # Calculate validation metrics
        loss = torch.norm(H_pred - labels.view(-1, 3, 3), dim=(1, 2)).mean()
        acc = accuracy(H_pred.view(-1, 8), labels)

        return {'loss': loss.detach(), 'acc': acc}