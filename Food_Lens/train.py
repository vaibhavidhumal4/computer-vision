"""
Fine-tune MobileNetV3-Large on Food-101 dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    BATCH_SIZE = 64 if torch.cuda.is_available() else 16
    NUM_EPOCHS = 10
    LR = 1e-3
    NUM_WORKERS = 4
    DATA_DIR = "./data"
    MODEL_SAVE_PATH = "./model/food_model.pth"

    os.makedirs("./model", exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Downloading Food-101 dataset (this may take a while on first run ~5GB)...")
    train_dataset = torchvision.datasets.Food101(root=DATA_DIR, split="train", transform=train_transform, download=True)
    val_dataset = torchvision.datasets.Food101(root=DATA_DIR, split="test", transform=val_transform, download=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("Loading MobileNetV3-Large with ImageNet weights...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 101)
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None  # Fixed deprecation warning too

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        if epoch == 4:
            print("\nUnfreezing all layers for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.*train_correct/train_total:.1f}%")

        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        train_acc = 100. * train_correct / train_total
        print(f"\nEpoch {epoch}: Train Acc={train_acc:.2f}%  Val Acc={val_acc:.2f}%  LR={scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Saved best model (val_acc={val_acc:.2f}%) -> {MODEL_SAVE_PATH}")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()