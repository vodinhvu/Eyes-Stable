import os
import argparse
import time
import logging
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms, datasets, models
import torchmetrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device| str):
    """Train the model for one epoch."""
    model.train()
    process = tqdm.tqdm(dataloader, desc='Train')
    total_loss = 0.0
    
    for inputs, labels in process:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        process.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    logging.info(f'Training Loss: {avg_loss:.4f}')
    return preds, labels

def validate(model, dataloader, criterion, accuracy_fn, conf_mat_fn, device):
    """Validate the model after training."""
    model.eval()
    process = tqdm.tqdm(dataloader, desc='Validation')
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in process:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            accuracy = accuracy_fn(preds, labels)
            process.set_postfix({'accuracy': accuracy.item()})

    all_preds_tensor = torch.tensor(all_preds, device=device)
    all_labels_tensor = torch.tensor(all_labels, device=device)
    all_accuracy = accuracy_fn(all_preds_tensor, all_labels_tensor).item()
    conf_mat = conf_mat_fn(all_preds_tensor, all_labels_tensor)

    return all_accuracy, conf_mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, default='datasets/hymenoptera_data')
    parser.add_argument('--epochs', '-e', type=int, default=3)
    parser.add_argument('--save', '-s', type=str, default='models')
    parser.add_argument('--batch', '-b', type=int, default=4)
    parser.add_argument('--num_workers', '-n', type=int, default=2)
    parser.add_argument('--save_every_epoch', action='store_true', help='Save model after every epoch')
    opt = parser.parse_args()

    # Define the data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(opt.image, x), data_transforms)
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=opt.batch,
                                                  shuffle=True, num_workers=opt.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    logging.info(f'Classes: {class_names}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and modify the model
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_features = model_ft.fc.in_features
    model_ft.fc = torch.nn.Sequential(torch.nn.Linear(num_features, len(class_names)), torch.nn.Softmax(dim=1))
    model_ft = model_ft.to(device)

    # Freeze early layers
    for i, params in enumerate(model_ft.parameters()):
        if i < 50:
            params.requires_grad = False

    total_params = sum(p.numel() for p in model_ft.parameters())
    requires_grad_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    logging.info(f'Total parameters: {total_params}, Trainable parameters: {requires_grad_params}')

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.1)

    best_acc = 0.0
    best_model_params_path = os.path.join(opt.save, 'best_model_params.pt')

    conf_mat_fn = torchmetrics.ConfusionMatrix(task='binary', num_classes=2).to(device)
    accuracy_fn = torchmetrics.Accuracy(task='binary', num_classes=2).to(device)

    since = time.time()
    
    for epoch in range(opt.epochs):
        logging.info(f'Epoch {epoch}/{opt.epochs - 1}')
        logging.info('-' * 10)

        # Train for one epoch
        train(model_ft, dataloaders['train'], criterion, optimizer_ft, device)

        # Validate the model
        all_accuracy, conf_mat = validate(model_ft, dataloaders['val'], criterion, accuracy_fn, conf_mat_fn, device)

        if opt.save_every_epoch:
            torch.save(model_ft.state_dict(), os.path.join(opt.save, f'epoch_{epoch}_acc_{all_accuracy:.4f}.pt'))

        if all_accuracy > best_acc:
            best_acc = all_accuracy
            torch.save(model_ft.state_dict(), best_model_params_path)

        logging.info(f'Confusion Matrix:\n{conf_mat}')
        logging.info(f'Best val Acc: {best_acc:4f}')

        time_elapsed = time.time() - since
        logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        exp_lr_scheduler.step()

    logging.info(f'Final Best val Acc: {best_acc:4f}')

if __name__ == '__main__':
    main()
