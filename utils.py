######################################################
# #            CS182 Demo: Data Augmentations        ##
# #               Code based on CS189 HW5            ##
# # https://www.eecs189.org/static/homeworks/hw5.pdf ##
# #####################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def run_training_loop(
    model,
    train_data,
    valid_data,
    batch_size=32,
    n_epochs=10,
    lr=1e-3,
    device="cpu",
    model_path=None,
    curves_path=None,
):
    """
    Run a training loop based on the input model and associated parameters

    Parameters:
        model: The input model to be trained
        train_data: The training dataset
        valid_data: The validation dataset
        batch_size: Number of training points to include in batch. Defaults to 32.
        n_epochs: Number of epochs to train the model for. Defaults to 10.
        lr: Learning rate used in Adam optimizer. Defaults to 1e-3.
        device: The device the model is to be trained on. Defaults to 'cpu'.
        model_path: The path where the model will be saved. Defaults to None and
            no model will be saved.
        curves_path: The path where the train and validation loss curves will be
            saved. Defaults to None and no plots will be saved.
    """
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=True
    )

    model.to(device)

    # Choose Adam as the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use the cross entropy loss function
    loss_fn = nn.CrossEntropyLoss()

    # store metrics
    train_loss_history = np.zeros([n_epochs, 1])
    valid_accuracy_history = np.zeros([n_epochs, 1])
    valid_loss_history = np.zeros([n_epochs, 1])

    for epoch in range(n_epochs):

        # Some layers, such as Dropout, behave differently during training
        model.train()

        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)

            # Erase accumulated gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = loss_fn(output, target)
            train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Weight update
            optimizer.step()

        train_loss_history[epoch] = train_loss / len(train_loader.dataset)

        # Track loss each epoch
        print(
            "Train Epoch: %d  Average loss: %.4f"
            % (epoch + 1, train_loss_history[epoch])
        )

        # Putting layers like Dropout into evaluation mode
        model.eval()

        valid_loss = 0
        correct = 0

        # Turning off automatic differentiation
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_loss += loss_fn(
                    output, target
                ).item()  # Sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # Get the index of the max class score
                correct += pred.eq(target.view_as(pred)).sum().item()

        valid_loss_history[epoch] = valid_loss / len(valid_loader.dataset)
        valid_accuracy_history[epoch] = correct / len(valid_loader.dataset)

        print(
            "Valid set: Average loss: %.4f, Accuracy: %d/%d (%.4f)\n"
            % (
                valid_loss_history[epoch],
                correct,
                len(valid_loader.dataset),
                100.0 * valid_accuracy_history[epoch],
            )
        )

    # Save checkpoint of the model after epoch
    if model_path is not None:
        torch.save(model, model_path)

    if curves_path is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_loss_history, label="train")
        ax.plot(valid_loss_history, label="valid")
        ax.set_xlabel("Epoch Number")
        ax.set_ylabel("Cross-Entropy Loss")
        plt.title("Training and Validation Loss Curves")
        plt.legend()
        plt.savefig(curves_path, dpi=300, bbox_inches="tight")

    return model, train_loss_history, valid_loss_history, valid_accuracy_history


def test_performance(
    model,
    test_data,
    batch_size=32,
    device="cpu",
):
    """
    Test model performance on test dataset

    Parameters:
        model: The model to be tested
        test_data: The test dataset
        batch_size: Number of training points to include in batch. Defaults to 32.
        device: The device the model is to be trained on. Defaults to 'cpu'.
    """
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )

    # Putting layers like Dropout into evaluation mode
    model.eval()
    # Use the cross entropy loss function
    loss_fn = nn.CrossEntropyLoss()

    # Send model to appropriate device
    model.to(device)

    test_loss = 0
    correct = 0

    # Turning off automatic differentiation
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # Sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # Get the index of the max class score
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)"
        % (test_loss, correct, len(test_loader.dataset), 100.0 * test_accuracy)
    )
    return test_loss, test_accuracy


def show_data_augmentations(
    original_im,
    transform_f,
    title=None,
):
    """
    Displays a 1 x 4 grid of augmentations applied to the original image

    Parameters:
        original_im: Original image to apply transform to
        transform_f: PyTorch transform to apply to image
        title: Optional title to plot with image grid. Defaults to None, and no
            title is provided.
    """
    # Randomly augment the original image four times
    ims_augmented = [
        transform_f(original_im.permute(2, 0, 1)).permute(1, 2, 0)
        for _ in range(4)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for im, ax in zip(ims_augmented, axes):
        ax.imshow(im)
        ax.set_xticks(ticks=[])
        ax.set_yticks(ticks=[])

    plt.subplots_adjust(wspace=0, hspace=0)

    if title is not None:
        plt.suptitle(title, y=0.95)

    # Show plot
    plt.show()
