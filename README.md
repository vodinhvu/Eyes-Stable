# Image Classification with Transfer Learning

This repository contains a PyTorch implementation for image classification using transfer learning with a ResNet-18 model. The model is trained to classify images into different categories using data from a specified directory structure.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training Process](#training-process)
- [Results](#results)
- [License](#license)

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.6 or higher
- PyTorch 1.7.0 or higher
- torchvision
- tqdm
- torchmetrics

You can install the required packages using pip:

```bash
pip install torch torchvision tqdm torchmetrics
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vodinhvu/Eyes-Stable.git
   cd Eyes-Stable
   ```

2. Prepare your dataset in the following structure:

   ```
   datasets/hymenoptera_data/
       train/
           class_1/
               image1.jpg
               image2.jpg
               ...
           class_2/
               image1.jpg
               image2.jpg
               ...
       val/
           class_1/
               image1.jpg
               image2.jpg
               ...
           class_2/
               image1.jpg
               image2.jpg
               ...
   ```

## Usage

Run the training script with the following command:

```bash
python train.py --image datasets/hymenoptera_data --epochs 3 --save models --batch 4 --num_workers 2
```

### Arguments:

- `--image` or `-i`: Path to the dataset folder (default: `datasets/hymenoptera_data`)
- `--epochs` or `-e`: Number of training epochs (default: `3`)
- `--save` or `-s`: Directory to save the model checkpoints (default: `models`)
- `--batch` or `-b`: Batch size for training (default: `4`)
- `--num_workers` or `-n`: Number of workers for data loading (default: `2`)

## Training Process

1. The model will be trained on the training set and validated on the validation set.
2. The progress of training and validation will be displayed using tqdm for better visibility.
3. The confusion matrix and accuracy will be computed for the validation set after each epoch.
4. Model checkpoints will be saved after each epoch, and the best model (with the highest validation accuracy) will be saved separately.

## Results

After training, you can find the best model saved in the `models` directory. The best validation accuracy will be printed to the console after the training is complete.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or feedback, feel free to open an issue or contact me.

