import gc
import glob
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from numpy import random
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, TensorDataset
from torchvision.utils import save_image

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the ViT-based segmentation model
class ViT_Segmentation(nn.Module):
    def __init__(self, num_classes=1, img_size=224, in_chans=1):
        super(ViT_Segmentation, self).__init__()
        # Initialize the ViT model from timm
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            img_size=img_size,
            num_classes=0,  # Remove the classification head
            in_chans=in_chans,
        )
        # Define the decoder to upsample the features to the original image size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2),
            nn.Sigmoid(),  # Assuming output is between 0 and 1
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        # Exclude the class token and reshape
        x = x[:, 1:]
        H = W = int(x.shape[1] ** 0.5)
        x = x.transpose(1, 2).reshape(batch_size, -1, H, W)
        feature_map = x  # Save feature map for visualization
        x = self.decoder(x)
        return x, feature_map


class FaceMapDataset(Dataset):
    def __init__(
        self,
        data_file="data/facemap_softlabels.pt",
        transform=None,
        rotation_degrees=15,
        blur_radius=(1, 2),  # Tuple for Gaussian blur radius range
        img_size=224,  # Added img_size parameter
    ):
        super().__init__()
        self.transform = transform
        self.rotation_degrees = rotation_degrees
        self.blur_radius = blur_radius
        self.img_size = img_size  # Save img_size for resizing
        self.data, _, self.targets = torch.load(data_file)

    def __len__(self):
        return len(self.targets) * 5  # For five versions per image

    def __getitem__(self, index):
        base_index = index // 5
        aug_type = index % 5

        image, label = self.data[base_index].clone(), self.targets[base_index].clone()

        # Resize images to match ViT input size
        resize_transform = transforms.Resize((self.img_size, self.img_size))
        image = resize_transform(image)
        label = resize_transform(label)

        # Apply augmentations
        if self.transform is not None:
            if aug_type == 1:  # Flipping
                image = image.flip([2])
                label = label.flip([2])
            elif aug_type == 2:  # Rotation
                angle = (torch.rand(1).item() * 2 - 1) * self.rotation_degrees
                image = TF.rotate(image, angle)
                label = TF.rotate(label, angle)
            elif aug_type == 3:  # Zooming
                scale_factor = 1.1 if torch.rand(1).item() < 0.5 else 0.9
                image = self.zoom(image, scale_factor)
                label = self.zoom(label, scale_factor)
            elif aug_type == 4:  # Gaussian Blur
                radius = (
                    torch.rand(1).item() * (self.blur_radius[1] - self.blur_radius[0])
                    + self.blur_radius[0]
                )
                image = TF.gaussian_blur(image, kernel_size=int(radius))
                label = TF.gaussian_blur(label, kernel_size=int(radius))

        return image, label

    def zoom(self, img, scale_factor):
        _, h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        img = TF.resize(img, [new_h, new_w])
        img = TF.center_crop(img, [h, w])
        return img


# Set the image size for ViT
img_size = 224

# Create dataset with the specified img_size
dataset = FaceMapDataset(transform=None, img_size=img_size)

x = dataset[0][0]
dim = x.shape[-1]
print("Using %d size of images" % dim)
N = len(dataset)
train_sampler = SubsetRandomSampler(np.arange(int(0.6 * N)))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6 * N), int(0.8 * N)))
test_sampler = SubsetRandomSampler(np.arange(int(0.8 * N), N))
batch_size = 4

# Initialize loss and metrics
loss_fun = torch.nn.MSELoss(reduction="sum")

# Initialize input dimensions
num_train = len(train_sampler)
num_valid = len(valid_sampler)
num_test = len(test_sampler)
print(
    "Num. train = %d, Num. val = %d, Num. test = %d" % (num_train, num_valid, num_test)
)

# Initialize dataloaders
loader_train = DataLoader(
    dataset=dataset,
    drop_last=False,
    num_workers=0,
    batch_size=batch_size,
    pin_memory=True,
    sampler=train_sampler,
)
loader_valid = DataLoader(
    dataset=dataset,
    drop_last=True,
    num_workers=0,
    batch_size=batch_size,
    pin_memory=True,
    sampler=valid_sampler,
)
loader_test = DataLoader(
    dataset=dataset,
    drop_last=True,
    num_workers=0,
    batch_size=1,
    pin_memory=True,
    sampler=test_sampler,
)

nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

# Hyperparameters
lr = 5e-4
num_epochs = 1000

# Initialize the ViT-based segmentation model
model = ViT_Segmentation(num_classes=1, img_size=img_size, in_chans=1)
model = model.to(device)
nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters: %.2f M" % (nParam / 1e6))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
minLoss = 1e6
convIter = 0
patience = 1000
train_loss = []
valid_loss = []

for epoch in range(num_epochs):
    tr_loss = 0
    model.train()
    for i, (inputs, labels) in enumerate(loader_train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, _ = model(inputs)
        loss = loss_fun(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, i + 1, nTrain, loss.item()
            )
        )
        tr_loss += loss.item()
    train_loss.append(tr_loss / (i + 1))

    with torch.no_grad():
        model.eval()
        val_loss = 0
        for i, (inputs, labels) in enumerate(loader_valid):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores, fmap = model(inputs)
            loss = loss_fun(scores, labels)
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

        valid_loss.append(val_loss)

        print("Val. loss :%.4f" % val_loss)

        labels = labels.squeeze().detach().cpu().numpy()
        scores = scores.squeeze().detach().cpu().numpy()
        img = inputs.squeeze().detach().cpu().numpy()
        fmap_mean = fmap.mean(1).squeeze().detach().cpu().numpy()

        plt.clf()
        plt.figure(figsize=(16, 12))
        for i in range(batch_size):
            plt.subplot(batch_size, 3, 3 * i + 1)
            plt.imshow(labels[i], cmap="gray")
            plt.title("Ground Truth")
            plt.subplot(batch_size, 3, 3 * i + 2)
            plt.imshow(scores[i], cmap="gray")
            plt.title("Prediction")
            plt.subplot(batch_size, 3, 3 * i + 3)
            plt.imshow(fmap_mean[i], cmap="gray")
            plt.title("Feature Map Mean")

        plt.tight_layout()
        plt.savefig("logs/epoch_%03d.jpg" % epoch)
        plt.close()
        gc.collect()

        if minLoss > val_loss:
            convEpoch = epoch
            minLoss = val_loss
            convIter = 0
            torch.save(model.state_dict(), "models/best_model.pt")
        else:
            convIter += 1

        if convIter == patience:
            print(
                "Converged at epoch %d with val. loss %.4f" % (convEpoch + 1, minLoss)
            )
            break

plt.clf()
plt.plot(train_loss, label="Training")
plt.plot(valid_loss, label="Validation")
plt.plot(convEpoch, valid_loss[convEpoch], "x", label="Best Model")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.pdf")

# Load the best model for inference
model.load_state_dict(torch.load("models/best_model.pt"))
model.eval()

with torch.no_grad():
    val_loss = 0

    # Compute global min and max of feature maps across the test set
    global_min = float("inf")
    global_max = float("-inf")
    for i, (inputs, labels) in enumerate(loader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, fmap = model(inputs)
        fmap_min = fmap.min().item()
        fmap_max = fmap.max().item()
        if fmap_min < global_min:
            global_min = fmap_min
        if fmap_max > global_max:
            global_max = fmap_max

    fmap_range = global_max - global_min if global_max != global_min else 1e-6

    print("Global min of feature maps:", global_min)
    print("Global max of feature maps:", global_max)

    # Compute loss and plot images with normalized feature maps
    val_loss = 0
    for i, (inputs, labels) in enumerate(loader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, fmap = model(inputs)
        loss = loss_fun(scores, labels)
        val_loss += loss.item()

        img = inputs.squeeze().detach().cpu().numpy()
        pred = scores.squeeze().detach().cpu().numpy()
        labels_np = labels.squeeze().cpu().numpy()
        fmap_mean = fmap.mean(1).squeeze().cpu().numpy()
        fmap_each = fmap.squeeze().cpu().numpy()

        # Normalize the feature maps using global min and max
        fmap_mean_norm = (fmap_mean - global_min) / fmap_range
        fmap_each_norm = (fmap_each - global_min) / fmap_range

        # Extract individual normalized feature maps
        fmap_list = [fmap_each_norm[i] for i in range(fmap_each_norm.shape[0])]

        # Plotting code
        plt.clf()
        plt.figure(figsize=(12, 9))

        # Display the main images
        plt.subplot(3, 4, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Input Image")

        plt.subplot(3, 4, 2)
        plt.imshow(labels_np, cmap="gray")
        plt.title("Ground Truth")

        plt.subplot(3, 4, 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Prediction")

        plt.subplot(3, 4, 4)
        plt.imshow(fmap_mean_norm, cmap="gray")
        plt.title("Normalized Feature Map Mean")

        # Display individual normalized feature maps
        for idx in range(min(8, len(fmap_list))):
            plt.subplot(3, 4, 5 + idx)
            plt.imshow(fmap_list[idx], cmap="gray")
            plt.title(f"Feature Map {idx + 1}")

        plt.tight_layout()
        plt.savefig("preds/test_{:03d}.jpg".format(i))
        plt.close()
        gc.collect()

    val_loss = val_loss / (i + 1)
    print("Test loss: {:.4f}".format(val_loss))
