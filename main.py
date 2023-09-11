import os
import argparse
import time

import torch
import tqdm
from torchvision import transforms, datasets, models
import torchmetrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, default='datasets/hymenoptera_data')
    parser.add_argument('--epochs', '-e', type=int, default=3)
    parser.add_argument('--save', '-s', type=str, default='models')
    parser.add_argument('--batch', '-b', type=int, default=4)
    parser.add_argument('--num_workers', '-n', type=int, default=2)
    opt = parser.parse_args()

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(opt.image, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch,
                                                  shuffle=True, num_workers=opt.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(weights='IMAGENET1K_V1')

    num_features = model_ft.fc.in_features
    model_ft.fc = torch.nn.Sequential(torch.nn.Linear(num_features, len(class_names)), torch.nn.Softmax(dim=1))
    model_ft = model_ft.to(device)

    i = 0
    for params in model_ft.parameters():
        i += 1
        if i < 50:
            params.requires_grad = False

    total_params = sum(p.numel() for p in model_ft.parameters())
    requires_grad_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print('total_params {}, requires_grad_params {}'.format(total_params, requires_grad_params))

    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.1, verbose=True)

    since = time.time()

    # Create a temporary directory to save training checkpoints
    # with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(opt.save, 'best_model_params.pt')

    conf_mat_fn = torchmetrics.ConfusionMatrix(task='binary', num_classes=2).to(device)
    accuracy_fn = torchmetrics.Accuracy(task='binary', num_classes=2).to(device)

    best_acc = 0.0
    for epoch in range(opt.epochs):
        print(f'Epoch {epoch}/{opt.epochs - 1}')
        print('-' * 10)

        # Train
        model_ft.train()
        process = tqdm.tqdm(dataloaders['train'], desc='Train')
        for inputs, labels in process:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer_ft.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model_ft.forward(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase

                loss.backward()
                optimizer_ft.step()
            process.set_postfix({'loss': loss.item()})

        # Validation
        model_ft.eval()
        process = tqdm.tqdm(dataloaders['val'], desc='Validation')
        all_preds = []
        all_labels = []
        for inputs, labels in process:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model_ft.forward(inputs)
                _, preds = torch.max(outputs, 1)
                accuracy = accuracy_fn(preds, labels)
                all_preds.append(preds.tolist())
                all_labels.append(labels.tolist())
                process.set_postfix({'accuracy': accuracy.item()})
        all_preds_tensors = torch.tensor(all_preds, device=device)
        all_labels_tensors = torch.tensor(all_labels, device=device)
        all_accuracy = accuracy_fn(all_preds_tensors, all_labels_tensors).item()
        conf_mat = conf_mat_fn(all_preds_tensors, all_labels_tensors)

        torch.save(model_ft.state_dict(), os.path.join(opt.save, f'epoch_{epoch}_acc_{all_accuracy:.4f}.pt'))

        if all_accuracy > best_acc:
            best_acc = all_accuracy
            torch.save(model_ft.state_dict(), best_model_params_path)

        print(conf_mat)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        exp_lr_scheduler.step()
        print()

    print(f'Best val Acc: {best_acc:4f}')
