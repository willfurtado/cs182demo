import torch
import torchvision
import torchvision.transforms as transforms
from utils import run_training_loop

# Assign variables with model-specific parameters
model = torchvision.models.resnet18(weights=None)
model_path = "models/example_model.pt"
curves_path = "images/loss_example.png"
device = "cpu"

# Compose several data augmentations for the training data
data_aug_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ]
)

# No transform for the training data
base_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

cifar10_train = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

# Break up training data into training and validation sets
cifar10_train, cifar10_valid = torch.utils.data.random_split(
    dataset=cifar10_train,
    lengths=[int(len(cifar10_train) * 0.8), int(len(cifar10_train) * 0.2)],
    generator=torch.Generator().manual_seed(42),
)

# Run training using bsz=128, n_epochs=15
results = run_training_loop(
    model=model,
    train_data=cifar10_train,
    valid_data=cifar10_valid,
    batch_size=128,
    n_epochs=15,
    device=device,
    model_path=model_path,
    curves_path=curves_path,
)
(
    trained_model,
    train_loss_history,
    valid_loss_history,
    valid_accuracy_history,
) = results
