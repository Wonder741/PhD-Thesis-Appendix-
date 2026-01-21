"""
Description
-----------
This script trains and evaluates a late-fusion U-Net model with CBAM
(Convolutional Block Attention Modules) to predict high-resolution plantar
pressure maps from multi-modal grayscale inputs.

The model takes two input images per sample:
  1) Input A (e.g., foot mask or structural prior)
  2) Input B (e.g., sparse sensor or feature image)

These two inputs are processed by separate encoder branches and fused at
multiple scales (skip connections and bottleneck) using 1×1 convolutions
and attention mechanisms. Channel and spatial attention (CBAM) are applied
to emphasize informative features and spatial regions.

Data Organization
-----------------
The script assumes a k-fold cross-validation setup (default: 5 folds).
For each fold, data are organized into Training and Validation subsets:

  Data/
    ├── MASK/
    ├── F{sensor_number}/
    └── FP/

Each sample consists of aligned grayscale PNG images:
  - MASK image (Input A)
  - Sensor image (Input B)
  - Ground-truth pressure image (Target)

All images are resized to 64×64 and normalized to the range [-1, 1].

Training
--------
The generator network is trained using:
  - L1 loss (primary objective)
  - L2 (RMSE) loss for stability
  - Loss SUM
  - Total variation (TV) loss for spatial smoothness

A spatial mask is applied to ignore boundary regions during optimization.
Training uses mixed precision, gradient clipping, and a step learning-rate
scheduler.

Evaluation and Output
---------------------
After training each fold:
  - The trained model weights are saved
  - Validation images are predicted and saved as grayscale PNG files
  - Training and validation metrics are logged to CSV files

The predicted outputs are rescaled from [-1, 1] to [0, 255] for visualization.

Intended Use
------------
This script is designed for research experiments on plantar pressure
prediction, multi-modal sensor fusion, and attention-based deep learning
models. It can be adapted to other image-to-image regression tasks with
multiple input modalities.

Author / Notes
--------------
- Feature-level-fusion U-Net with CBAM attention
- Input resolution: 64×64
- Output: single-channel grayscale pressure map
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
import glob
import cv2
import random
import math
import torchvision.utils as vutils

# Directories

FOLD_LIST = ["1", "2", "3", "4", "5"] #["1", "2", "3", "4", "5"]
sensor_number = "2"
save_step = 20
DIR_ROOT = "Data"
DIR_A = "MASK"
DIR_B = fr"F{sensor_number}"
DIR_C = "FP"
DIR_TRAIN = "Training"
DIR_VALID = "Validation"

# Hyperparameters
BATCH_SIZE = 64
LR_GEN = 2e-4
EPOCHS = 51
L2_LAMBDA = 0.01
SUM_LAMBDA = 1
DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EDGE = 4
mask = torch.ones(1,1,64,64, device=DEVICE)
mask[:,:,:EDGE,:] = 0
mask[:,:,-EDGE:,:] = 0
mask[:,:,:,:EDGE] = 0
mask[:,:,:,-EDGE:] = 0

def tv_loss(x):
    return torch.mean(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])) + torch.mean(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]))

# Data loading
def load_dataset(a_dir, b_dir, c_dir, shuffle=True):
    dataset = CustomDataset(a_dir, b_dir, c_dir)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle), dataset

class CustomDataset(Dataset):
    def __init__(self, a_dir, b_dir, c_dir, transform=None):
        self.a_files = sorted(glob.glob(os.path.join(a_dir, "*.png")))
        self.b_files = sorted(glob.glob(os.path.join(b_dir, "*.png")))
        self.c_files = sorted(glob.glob(os.path.join(c_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.a_files)

    def __getitem__(self, idx):
        img_A = cv2.imread(self.a_files[idx], cv2.IMREAD_GRAYSCALE)
        img_B = cv2.imread(self.b_files[idx], cv2.IMREAD_GRAYSCALE)
        img_C = cv2.imread(self.c_files[idx], cv2.IMREAD_GRAYSCALE)

        img_A = cv2.resize(img_A, (64, 64))
        img_B = cv2.resize(img_B, (64, 64))
        img_C = cv2.resize(img_C, (64, 64))

        img_A = torch.tensor(img_A, dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0
        img_B = torch.tensor(img_B, dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0
        img_C = torch.tensor(img_C, dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0

        return torch.cat([img_A, img_B], dim=0), img_C  # Input: (A, B), Target: C

# ---------------------------
# Channel Attention Module
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# ---------------------------
# Spatial Attention Module
# ---------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Apply average and max pooling along channel dimensions
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

# ---------------------------
# CBAM Module: Sequential Channel & Spatial Attention
# ---------------------------
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x_out = x * self.channel_attention(x)
        x_out = x_out * self.spatial_attention(x_out)
        return x_out

# ---------------------------
# U-Net Generator with CBAM (Late Fusion, 2 encoders)
# ---------------------------
class UNetGeneratorLateFusionCBAM(nn.Module):
    """
    Late-fusion U-Net:
      - Encoder A processes xa (e.g., mask)
      - Encoder B processes xb (e.g., sparse landmarks)
      - Fuse at bottleneck and at each skip scale via 1x1 conv
      - Apply CBAM at fusion points + final high-res refinement

    Default inputs:
      xa: (B, 1, 64, 64)
      xb: (B, 1, 64, 64)
    Output:
      y : (B, 1, 64, 64) in [-1,1] (tanh)
    """
    def __init__(self, in_ch_a: int = 1, in_ch_b: int = 1):
        super().__init__()

        def down(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up(in_channels, out_channels):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            return nn.Sequential(*layers)

        # -------------------------
        # Two separate encoders
        # -------------------------
        # Encoder A
        self.a_down1 = down(in_ch_a, 64, normalize=False)  # 64x64 -> 32x32
        self.a_down2 = down(64, 128)                       # 32x32 -> 16x16
        self.a_down3 = down(128, 256)                      # 16x16 -> 8x8
        self.a_down4 = down(256, 512)                      # 8x8 -> 4x4
        self.a_down5 = down(512, 512)                      # 4x4 -> 2x2
        self.a_down6 = down(512, 512)                      # 2x2 -> 1x1 (for 64 input)

        # Encoder B
        self.b_down1 = down(in_ch_b, 64, normalize=False)
        self.b_down2 = down(64, 128)
        self.b_down3 = down(128, 256)
        self.b_down4 = down(256, 512)
        self.b_down5 = down(512, 512)
        self.b_down6 = down(512, 512)

        # -------------------------
        # Fusion layers (1x1 conv) + CBAM at fusion points
        # -------------------------
        # Skip fusions: (A + B) -> same channel count
        self.fuse_s1 = nn.Conv2d(64 + 64,   64,  kernel_size=1, bias=False)
        self.fuse_s2 = nn.Conv2d(128 + 128, 128, kernel_size=1, bias=False)
        self.fuse_s3 = nn.Conv2d(256 + 256, 256, kernel_size=1, bias=False)
        self.fuse_s4 = nn.Conv2d(512 + 512, 512, kernel_size=1, bias=False)
        self.fuse_s5 = nn.Conv2d(512 + 512, 512, kernel_size=1, bias=False)

        self.cbam_s1 = CBAM(64,  reduction=8, kernel_size=7)
        self.cbam_s2 = CBAM(128, reduction=8, kernel_size=7)
        self.cbam_s3 = CBAM(256, reduction=8, kernel_size=7)
        self.cbam_s4 = CBAM(512, reduction=8, kernel_size=7)
        self.cbam_s5 = CBAM(512, reduction=8, kernel_size=7)

        # Bottleneck fusion: (512 + 512) -> 512
        self.fuse_bottleneck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.cbam_z = CBAM(512, reduction=8, kernel_size=7)

        # -------------------------
        # Shared decoder
        # -------------------------
        self.up1 = up(512, 512)      # 1x1 -> 2x2
        self.up2 = up(512 + 512, 512)  # cat with s5
        self.up3 = up(512 + 512, 256)  # cat with s4
        self.up4 = up(256 + 256, 128)  # cat with s3
        self.up5 = up(128 + 128, 64)   # cat with s2

        # Final refinement: cat(u5, s1) -> 128 channels, then CBAM, then final up to 64x64
        self.cbam_final = CBAM(64 + 64, reduction=8, kernel_size=7)
        self.final = nn.ConvTranspose2d(64 + 64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        # -------------------------
        # Encoder A
        # -------------------------
        a1 = self.a_down1(xa)
        a2 = self.a_down2(a1)
        a3 = self.a_down3(a2)
        a4 = self.a_down4(a3)
        a5 = self.a_down5(a4)
        a6 = self.a_down6(a5)

        # -------------------------
        # Encoder B
        # -------------------------
        b1 = self.b_down1(xb)
        b2 = self.b_down2(b1)
        b3 = self.b_down3(b2)
        b4 = self.b_down4(b3)
        b5 = self.b_down5(b4)
        b6 = self.b_down6(b5)

        # -------------------------
        # Fuse skip connections (late-ish fusion per scale) + CBAM
        # -------------------------
        s1 = self.cbam_s1(self.fuse_s1(torch.cat([a1, b1], dim=1)))
        s2 = self.cbam_s2(self.fuse_s2(torch.cat([a2, b2], dim=1)))
        s3 = self.cbam_s3(self.fuse_s3(torch.cat([a3, b3], dim=1)))
        s4 = self.cbam_s4(self.fuse_s4(torch.cat([a4, b4], dim=1)))
        s5 = self.cbam_s5(self.fuse_s5(torch.cat([a5, b5], dim=1)))

        # -------------------------
        # Bottleneck fusion (true late fusion) + CBAM
        # -------------------------
        z = self.fuse_bottleneck(torch.cat([a6, b6], dim=1))
        z = self.cbam_z(z)

        # -------------------------
        # Decoder
        # -------------------------
        u1 = self.up1(z)
        u2 = self.up2(torch.cat([u1, s5], dim=1))
        u3 = self.up3(torch.cat([u2, s4], dim=1))
        u4 = self.up4(torch.cat([u3, s3], dim=1))
        u5 = self.up5(torch.cat([u4, s2], dim=1))

        cat = torch.cat([u5, s1], dim=1)     # (B, 128, 32, 32)
        cat = self.cbam_final(cat)
        out = torch.tanh(self.final(cat))    # (B, 1, 64, 64)
        return out

for folder_number in FOLD_LIST:

    print(f"\n==============================")
    print(f" Running Fold {folder_number}")
    print(f"==============================")

    # ---------------------------
    # Directories (per fold)
    # ---------------------------
    DIR_FOLDER = f"Folder{folder_number}"

    DATA_DIR_A = os.path.join(DIR_ROOT, DIR_A, DIR_FOLDER, DIR_TRAIN)
    DATA_DIR_B = os.path.join(DIR_ROOT, DIR_B, DIR_FOLDER, DIR_TRAIN)
    DATA_DIR_C = os.path.join(DIR_ROOT, DIR_C, DIR_FOLDER, DIR_TRAIN)

    VALID_DIR_A = os.path.join(DIR_ROOT, DIR_A, DIR_FOLDER, DIR_VALID)
    VALID_DIR_B = os.path.join(DIR_ROOT, DIR_B, DIR_FOLDER, DIR_VALID)
    VALID_DIR_C = os.path.join(DIR_ROOT, DIR_C, DIR_FOLDER, DIR_VALID)

    RESULTS_DIR = os.path.join(
        f"Result_{DIR_A}{sensor_number}",
        f"UC{sensor_number}_E15_folder{folder_number}"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    LOG_FILE = os.path.join(RESULTS_DIR, f"log_{sensor_number}_{folder_number}.csv")

    # ---------------------------
    # Initialize model and optimizer
    # ---------------------------
    generator = UNetGeneratorLateFusionCBAM().to(DEVICE)
    optim_G = optim.Adam(generator.parameters(), lr=LR_GEN, betas=(0.5, 0.999), weight_decay=DECAY)
    # Schaduled LR
    scheduler_G = lr_scheduler.StepLR(optim_G, step_size=save_step, gamma=0.5)


    # Define loss functions: L1 and L2 (MSE)
    criterion_L1 = nn.L1Loss()
    criterion_L2 = nn.MSELoss()
    scaler = torch.amp.GradScaler()
    print("Model and optimizer are set up.")

    # Data loading
    #transform = transforms.Compose([transforms.ToTensor()])
    dataloader, train_dataset = load_dataset(DATA_DIR_A, DATA_DIR_B, DATA_DIR_C, shuffle=True)
    valid_dataloader, valid_dataset = load_dataset(VALID_DIR_A, VALID_DIR_B, VALID_DIR_C, shuffle=False)
    print("Data is set up.")

    # Open CSV and write header (only once)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "L1_Loss", "L1_Valid", "L2_Loss", "L2_Valid",  "SUM_loss", "Sum_Valid"])
    print("Log file initialized.")

    # Training loop (only L1 loss is used)
    for epoch in range(EPOCHS):
        total_loss_L1 = 0
        total_loss_L2 = 0
        total_loss_SUM = 0
        num_batches = len(dataloader)

        # Training
        generator.train()
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            with torch.amp.autocast(DEVICE):
                xa = inputs[:, 0:1, :, :]
                xb = inputs[:, 1:2, :, :]
                fake = generator(xa, xb)

                fake = fake * mask + (-1.0)*(1.0-mask)
                loss_L1 = criterion_L1(fake, targets)
                loss_L2 = torch.sqrt(criterion_L2(fake, targets))
                loss_SUM = torch.mean(torch.abs(fake.sum(dim=[2, 3]) - targets.sum(dim=[2, 3]))) 
                loss_TV = tv_loss(fake)
                loss = loss_L1 + (loss_SUM * SUM_LAMBDA) + 1e-5 * loss_TV

            optim_G.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler.step(optim_G)
            scaler.update()

            total_loss_L1 += loss_L1.item()
            total_loss_L2 += loss_L2.item()
            total_loss_SUM += loss_SUM.item()

        avg_loss_L1 = total_loss_L1 / num_batches
        avg_loss_L2 = total_loss_L2 / num_batches
        avg_loss_SUM = total_loss_SUM / num_batches * SUM_LAMBDA
        
        # Validation
        generator.eval()
        total_val_rmse = 0
        total_val_sum = 0
        total_val_mae = 0
        with torch.no_grad():
            for val_inputs, val_targets in valid_dataloader:
                val_inputs, val_targets = val_inputs.to(DEVICE), val_targets.to(DEVICE)
                xa = val_inputs[:, 0:1, :, :]
                xb = val_inputs[:, 1:2, :, :]
                val_pred = generator(xa, xb) * mask + (-1.0)*(1.0-mask)
                mae_batch = torch.mean(torch.abs(val_pred - val_targets), dim=[1, 2, 3])
                rmse_batch = torch.sqrt(torch.mean((val_pred - val_targets) ** 2, dim=[1, 2, 3]))
                sum_batch = torch.abs((val_pred.sum() - val_targets.sum()))
                total_val_mae += mae_batch.sum().item()
                total_val_rmse += rmse_batch.sum().item()
                total_val_sum += sum_batch.item()
        avg_val_mae = total_val_mae / len(valid_dataset)
        avg_val_rmse = total_val_rmse / len(valid_dataset)
        avg_val_sum = total_val_sum / len(valid_dataset) * SUM_LAMBDA
        # adjust the LR
        scheduler_G.step()

        # Log and print
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss_L1, avg_val_mae, avg_loss_L2, avg_val_rmse, avg_loss_SUM, avg_val_sum])
        print(f"Epoch [{epoch+1}/{EPOCHS}] - L1: {(avg_loss_L1*127.5):.2f}, V1: {(avg_val_mae*127.5):.2f}, L2: {(avg_loss_L2*127.5):.2f}, V2: {avg_val_rmse*127.5:.2f}, TS: {(avg_loss_SUM*127.5):.1f}, VS: {(avg_val_sum*127.5):.1f}")

    # Final save
    torch.save(generator.state_dict(), os.path.join(RESULTS_DIR, "generator_final.pth"))
    print("Final trained generator saved!")

    # ---------------------------
    # Inference on validation set + save predictions
    # ---------------------------
    result_dir = os.path.join(RESULTS_DIR, "result")
    os.makedirs(result_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        for i in range(len(valid_dataset)):
            # valid_dataset[i] returns: (inputs, targets)
            inputs, _ = valid_dataset[i]  # inputs: (2, 64, 64)
            inputs = inputs.unsqueeze(0).to(DEVICE)  # -> (1, 2, 64, 64)
            xa = inputs[:, 0:1, :, :]
            xb = inputs[:, 1:2, :, :]

            pred = generator(xa, xb)  # -> (1, 1, 64, 64), range [-1, 1] due to tanh
            pred = pred * mask + (-1.0)*(1.0-mask)

            # Convert to uint8 image [0,255] for saving as PNG
            pred_img = pred.squeeze(0).squeeze(0).detach().cpu().numpy()  # (64, 64)
            pred_img = ((pred_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            # Use the same filename as the validation ground-truth FP image (C)
            filename = os.path.basename(valid_dataset.c_files[i])
            save_path = os.path.join(result_dir, filename)

            cv2.imwrite(save_path, pred_img)

    print(f"Validation predictions saved to: {result_dir}")