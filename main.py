import os
import argparse
import time

import torch
import tqdm
from torchvision import transforms, datasets, models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, default='datasets/hymenoptera_data')
    parser.add_argument('--epochs', '-e', type=int, default=3)
    parser.add_argument('--save', '-s', type=str, default='models')
    parser.add_argument('--batch', '-b', type=int, default=4)
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
                                                  shuffle=True)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.Inception3(num_classes=len(class_names), aux_logits=False, init_weights=True)

    model_ft = model_ft.to(device)

    total_params = sum(p.numel() for p in model_ft.parameters())
    requires_grad_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print('total_params {}, requires_grad_params {}'.format(total_params, requires_grad_params))

    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    since = time.time()

    # Create a temporary directory to save training checkpoints
    # with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(opt.save, 'best_model_params.pt')

    torch.save(model_ft.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(opt.epochs):
        print(f'Epoch {epoch}/{opt.epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()  # Set model to training mode
            else:
                model_ft.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            process = tqdm.tqdm(dataloaders[phase], desc=phase)
            # Iterate over data.
            for inputs, labels in process:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft.forward(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                process.set_postfix({'loss': loss.item(), 'lr': exp_lr_scheduler.get_last_lr()[0]})
            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val':
                torch.save(model_ft.state_dict(), os.path.join(opt.save, f'epoch_{epoch}_acc_{epoch_acc:.4f}.pt'))
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model_ft.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    # model_ft.load_state_dict(torch.load(best_model_params_path))
