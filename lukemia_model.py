import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random
import copy

DATA_DIR    = "dataset/lukemia_subtype"
BATCH_SIZE  = 32
EPOCHS      = 80
LR          = 3e-4
PATIENCE    = 15
SAVE_PATH   = "best_lukemia_model.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def predict(image_path: str, model_path: str = SAVE_PATH, confidence_threshold: float = 0.5):
    from PIL import Image

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint  = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']

    net = models.efficientnet_b3(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    net.load_state_dict(checkpoint['model_state'])
    net = net.to(DEVICE)
    net.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    image  = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs    = net(tensor)
        probs      = torch.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)
        confidence = confidence.item()
        pred_idx   = pred_idx.item()

    predicted_class = class_names[pred_idx] if confidence >= confidence_threshold else "Uncertain (low confidence)"

    all_probs = {class_names[i]: round(probs[i].item(), 4) for i in range(num_classes)}
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

    print(f"\nImage     : {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print("Top probabilities:")
    for cls, prob in list(all_probs.items())[:5]:
        bar = "█" * int(prob * 40)
        print(f"  {cls:<30} {prob:.4f}  {bar}")

    return {'class': predicted_class, 'confidence': confidence, 'all_probs': all_probs}

if __name__ == '__main__':

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    full_dataset = datasets.ImageFolder(DATA_DIR)
    targets      = np.array(full_dataset.targets)
    NUM_CLASSES  = len(full_dataset.classes)
    CLASS_NAMES  = full_dataset.classes

    print(f"\nClasses ({NUM_CLASSES}): {CLASS_NAMES}")
    class_dist = {CLASS_NAMES[i]: int((targets == i).sum()) for i in range(NUM_CLASSES)}
    print("Class Distribution:", class_dist)

    class_counts  = np.array([class_dist[c] for c in CLASS_NAMES], dtype=np.float32)
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32)

    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        stratify=targets,
        random_state=42
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(DATA_DIR, transform=train_transform), train_idx
    )
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(DATA_DIR, transform=val_transform), val_idx
    )

    train_targets  = targets[train_idx]
    sample_weights = class_weights[train_targets]
    sampler        = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,     
        pin_memory=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,      
        pin_memory=False
    )

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    model = model.to(DEVICE)
    print(f"\nModel: EfficientNet-B3 | Classes: {NUM_CLASSES} | Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE),
        label_smoothing=0.1
    )
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(EPOCHS - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc     = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    early_stop_count = 0

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    for epoch in range(EPOCHS):
        model.train()
        train_loss    = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            _, preds      = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_loss    += loss.item()

        train_acc  = train_correct / len(train_dataset)
        train_loss = train_loss / len(train_loader)

        model.eval()
        val_correct = 0
        val_loss    = 0.0
        all_preds   = []
        all_labels  = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs  = model(images)
                loss     = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_correct += (preds == labels).sum().item()
                val_loss    += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc    = val_correct / len(val_dataset)
        val_loss   = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"\nEpoch [{epoch+1:03d}/{EPOCHS}]  LR: {current_lr:.2e}  |  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_count = 0
            torch.save({
                'epoch':       epoch + 1,
                'model_state': best_model_state,
                'val_acc':     best_val_acc,
                'class_names': CLASS_NAMES,
                'num_classes': NUM_CLASSES,
            }, SAVE_PATH)
            print(f" Saved best model  (val acc: {best_val_acc:.4f})")
        else:
            early_stop_count += 1
            print(f"  No improvement ({early_stop_count}/{PATIENCE})")
            if early_stop_count >= PATIENCE:
                print("\nEarly stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    print(f"\n{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))