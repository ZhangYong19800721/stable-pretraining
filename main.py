
import torch
import torch.nn as nn
import torchvision
import torchmetrics
import lightning as pl

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.forward import simclr

print(f"stable-pretraining {spt.__version__}")
print(f"PyTorch {torch.__version__}  |  Lightning {pl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# step1

# spt.data.static.CIFAR10 is {"mean": [...], "std": [...]}
CIFAR10_STATS = spt.data.static.CIFAR10

# Two augmented views for SimCLR on 32x32 CIFAR images
simclr_transform = transforms.MultiViewTransform([
    # View 1: standard crop + color jitter
    transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToImage(**CIFAR10_STATS),
    ),
    # View 2: tighter crop + solarize
    transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomSolarize(threshold=0.5, p=0.2),
        transforms.ToImage(**CIFAR10_STATS),
    ),
])

# Deterministic transform for validation (no augmentation)
val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**CIFAR10_STATS),
)

# Wrap CIFAR-10 in dict format
train_dataset = spt.data.FromTorchDataset(
    torchvision.datasets.CIFAR10(root="./data", train=True, download=True),
    names=["image", "label"],
    transform=simclr_transform,
)
val_dataset = spt.data.FromTorchDataset(
    torchvision.datasets.CIFAR10(root="./data", train=False, download=True),
    names=["image", "label"],
    transform=val_transform,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=2, drop_last=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=256, num_workers=2
)

data = spt.data.DataModule(train=train_loader, val=val_loader)
print(f"Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
print(f"Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")

# step2

# ResNet-18 with the classifier head removed → 512-dim embeddings
# low_resolution=True adapts the first conv and removes MaxPool for 32x32 inputs
backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
backbone.fc = nn.Identity()

# 3-layer MLP projector (SimCLR paper architecture)
projector = nn.Sequential(
    nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
    nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
    nn.Linear(2048, 256),
)

module = spt.Module(
    forward=simclr,
    backbone=backbone,
    projector=projector,
    simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
    optim={
        # Adam for this quick demo; use LARS + LinearWarmupCosineAnnealing for full runs
        "optimizer": {"type": "Adam", "lr": 1e-3},
        "scheduler": "CosineAnnealingLR",
        "interval": "epoch",
    },
)

trainable = sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6
print(f"Trainable parameters: {trainable:.1f}M")

# step3

# Linear probe: trains a single Linear(512, 10) on top of frozen backbone embeddings
linear_probe = spt.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",   # key written by simclr
    target="label",      # key written by simclr
    probe=nn.Linear(512, 10),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

# KNN probe: non-parametric; maintains a rolling queue of past embeddings
knn_probe = spt.OnlineKNN(
    name="knn",
    input="embedding",
    target="label",
    queue_length=10000,
    input_dim=512,
    k=10,
    metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
)

# RankMe: tracks representation rank — collapse → rank approaches 1
rankme = spt.RankMe(name="rankme", target="embedding", queue_length=10000, target_shape=512)

# step4

trainer = pl.Trainer(
    max_epochs=5,          # Increase to 1000 for publication-quality results
    accelerator="auto",   # Uses GPU if available, otherwise CPU
    logger=False,
    enable_checkpointing=False,
    callbacks=[linear_probe, knn_probe, rankme],
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()

# step5

results = manager.validate()

if results:
    print(f"{'Metric':<45} {'Value':>8}")
    print("-" * 55)
    for key, val in sorted(results[0].items()):
        print(f"{key:<45} {val:>8.4f}")