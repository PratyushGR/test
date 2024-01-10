import torch
import numpy as np


def faster_dice(x, y, labels, fudge_factor=1e-8):
    """Faster PyTorch implementation of Dice scores.
    :param x: input label map as torch.Tensor
    :param y: input label map as torch.Tensor of the same size as x
    :param labels: list of labels to evaluate on
    :param fudge_factor: an epsilon value to avoid division by zero
    :return: pytorch Tensor with Dice scores in the same order as labels.
    """

    assert x.shape == y.shape, "both inputs should have same size, had {} and {}".format(
        x.shape, y.shape
    )

    if len(labels) > 1:

        dice_score = torch.zeros(len(labels))
        for label in labels:
            x_label = x == label
            y_label = y == label
            xy_label = (x_label & y_label).sum()
            dice_score[label] = (
                2 * xy_label / (x_label.sum() + y_label.sum() + fudge_factor)
            )

    else:
        dice_score = dice(x == labels[0], y == labels[0], fudge_factor=fudge_factor)

    return dice_score


def dice(x, y, fudge_factor=1e-8):
    """Implementation of dice scores ofr 0/1 numy array"""
    return 2 * torch.sum(x * y) / (torch.sum(x) + torch.sum(y) + fudge_factor)

class MeshNetTrainer:
    def __init__(self, model, db_file, loader, cubes=1, label_type='GWlabels', learning_rate=0.0004, num_epochs=10, num_classes=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_file = db_file
        self.cubes = cubes
        self.label_type = label_type
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.loader = loader
        
    def train(self):
        dataset = self.loader.Scanloader(self.db_file, label_type=self.label_type, num_cubes=self.cubes)
        shape = 256 // self.cubes
        train_loader, valid_loader, _ = dataset.get_loaders()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            dice = [0 for i in range(self.num_classes)]
            # Training
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data.reshape(-1, 1, shape, shape, shape))
                loss = criterion(output, target.reshape(-1, shape, shape, shape).long() * 2)
                loss.requires_grad_
                loss.backward()
                if self.cubes == 1:
                    dice_loss = faster_dice(torch.argmax(torch.squeeze(output), 0), target.reshape(shape, shape, shape) * 2, labels=[0, 1, 2])
                else:
                    dice_loss = faster_dice(torch.argmax(torch.squeeze(output), 1), target.reshape(-1, shape, shape, shape) * 2, labels=[0, 1, 2])

                for i in range(self.num_classes):
                    dice[i] += dice_loss[i]
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], "
                  f"Train Loss: {running_loss / len(train_loader):.4f}")
            for i in range(self.num_classes):
                print(f"Dice for class {i}: {dice[i] / len(train_loader):.4f}")
            running_loss = 0.0
            dice = [0,0,0]
            
            # Validation
            self.model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for val_data, val_target in valid_loader:
                    val_output = self.model(val_data.reshape(-1, 1, shape, shape, shape))
                    val_loss = criterion(val_output, val_target.reshape(-1, shape, shape, shape).long() * 2)
                    val_running_loss += val_loss.item()
                
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], "
                  f"Validation Loss: {val_running_loss / len(valid_loader):.4f}")
