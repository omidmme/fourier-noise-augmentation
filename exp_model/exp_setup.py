import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from exp_model.resnet import ResNet18
from fourier_noise.data import DatasetBuilder
from fourier_noise.fourier_heatmap import create_fourier_heatmap

criterion = nn.CrossEntropyLoss()
mean = [0.49139968, 0.48215841, 0.44653091]
std = [0.24703223, 0.24348513, 0.26158784]


# Training step
def train_model(model, device, epochs, patience, train_dataset, validation_dataset, optimizer, scheduler, suffix=""):
    best_loss = 1000
    stop_counter = 0
    for epoch in range(epochs):
        print("Epoch: ", epoch + 1)
        model.train()
        torch.set_grad_enabled(True)
        outs = []
        for train_batch_idx, train_batch in tqdm.tqdm(enumerate(train_dataset)):
            x, y = train_batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            outs.append(loss)
            loss.backward()
            optimizer.step()

        epoch_metric = torch.mean(torch.stack([x for x in outs]))
        print("train loss at epoch ", epoch, ": ", epoch_metric.tolist())
        current_loss = validation(model, device, validation_dataset)
        print("current validation loss: ", current_loss.tolist())

        if current_loss > best_loss:
            stop_counter += 1
            if stop_counter >= patience:
                state = {
                    'exp_model': model.state_dict(),
                    'best_acc': best_loss,
                    'best_epoch': epoch
                }
                torch.save(state, './best_early' + suffix + '.pth')
                return
        else:
            stop_counter = 0
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            state = {
                'exp_model': model.state_dict(),
                'best_acc': best_loss,
                'best_epoch': epoch
            }
        scheduler.step(current_loss)
    torch.save(state, './best_full' + suffix + '.pth')


# Validation step
def validation(model, device, validation_dataset):
    torch.set_grad_enabled(False)
    model.eval()
    with torch.no_grad():
        val_loss = []
        for val_batch_idx, val_batch in enumerate(validation_dataset):
            x, y = val_batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            val_loss.append(loss)
        avg_val_loss = torch.mean(torch.stack([x for x in val_loss]))
    return avg_val_loss


# Testing step
def test_model(model, device, test_dataset):
    correct = 0
    total = 0
    # disable gradients
    torch.set_grad_enabled(False)
    model.eval()
    with torch.no_grad():
        for batch_idx, test_batch in enumerate(test_dataset):
            x, y = test_batch
            y_hat = model(x)
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    simple_accuracy = correct / total
    print("simple accuracy: ", simple_accuracy)
    return str(simple_accuracy)


# Runs an experiment by training a model (ResNet18) with either base transformation or Fourier-Basis noise (fb_noise),
# evaluates the model on the CIFAR-10 test set and stores the result and computes the Fourier heatmap
def run_experiment(fb_noise, exp_name, heat_map_name):
    # set this variable to True to load the saved weights
    load_weights = False
    # set this variable to True to create a base transformation without Fourier-Base noise
    base_transform = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device state:', device)

    # number of max epochs
    epochs = 100

    # name of experiment
    name = exp_name
    # exp_model
    model = ResNet18().to(device)
    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, mode='min', verbose=True)

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if base_transform:
        train_transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
                                                 transforms.Pad(4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomCrop(32), transforms.ToTensor()] + fb_noise + [
                                                 transforms.Normalize(mean, std)])

    train_data = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True,
                                              transform=train_transform)
    test_data = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, download=True,
                                             transform=test_transform)

    # training/validation split
    train_data, val_data = torch.utils.data.random_split(train_data, [45000, 5000])

    # define DataLoader
    train_dataset = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=2)  # num_of_GPUS * 4)
    val_dataset = DataLoader(val_data, batch_size=100, num_workers=2)  # num_of_GPUS * 4)
    test_dataset = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)  # num_of_GPUS * 4)

    if load_weights:
        # define weights
        weights = torch.load("PATH")
        model.load_state_dict(weights["exp_model"])
    else:
        print("Training the exp_model")
        train_model(model, device, epochs, 30, train_dataset, val_dataset, optimizer, scheduler, name)

    print("Testing the exp_model")
    model = model.to("cpu")
    results = test_model(model, device, test_dataset)
    with open("results.txt", 'a') as f:
        f.write("Result of " + name + ": " + results + "\n")

    print("Creating the Fourier heatmap")
    dataset_builder = DatasetBuilder(name='cifar10', root_path='data/cifar10')
    model = model.to(device)
    create_fourier_heatmap(model, dataset_builder, 31, 31, 4.0, 'l2', -1, 100, 4, "", suffix=name,
                           fig_title=heat_map_name)
