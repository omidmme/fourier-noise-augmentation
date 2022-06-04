import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from model import ResNet18
from fourier_noise.data import DatasetBuilder
from fourier_noise.fourier_heatmap import create_fourier_heatmap
from fourier_noise.custom_noise_filter import CustomNoise

criterion = nn.CrossEntropyLoss()
mean = [0.49139968, 0.48215841, 0.44653091]
std  = [0.24703223, 0.24348513, 0.26158784]


# Training step
def train_model(model, device, epochs, patience, train_dataset, validation_dataset, optimizer, scheduler, suffix=""):
    # early loss
    prev_loss = 100
    best_loss = 100
    stop_counter = 0
    for epoch in range(epochs):
        print("Epoch: ", epoch+1)
        # set training mode
        model.train()
        torch.set_grad_enabled(True)
        outs = []
        for train_batch_idx, train_batch in tqdm.tqdm(enumerate(train_dataset)):
            # get the inputs and labels
            x, y = train_batch
            x, y = x.to(device), y.to(device)
            # clear gradients
            optimizer.zero_grad()
            # perform forward pass
            y_hat = model(x)
            # compute loss
            loss = criterion(y_hat, y)
            outs.append(loss)
            # perform backward pass
            loss.backward()
            # update parameters
            optimizer.step()

        epoch_metric = torch.mean(torch.stack([x for x in outs]))
        print("train loss epoch ", epoch, ": ", epoch_metric.tolist())
        # perform early stopping
        current_loss = validation(model, device, validation_dataset)
        print('current validation loss: ', current_loss.tolist())

        # check for improvement of loss to previous loss
        if current_loss > prev_loss:
            stop_counter += 1
            print('stopping counter: ', stop_counter)
            # check times of no improvement greater than or equal to patience
            if stop_counter >= patience:
                print(prev_loss)
                state = {
                    'model': model.state_dict(),
                    'best_acc': prev_loss,
                }
                torch.save(state, './best_early' + suffix + '.pth')
                print('early stopping occurred')
                return
        else:
            stop_counter = 0
        if current_loss <= best_loss:
            best_loss = current_loss
            # store state that has best validation metric
            state = {
                'model': model.state_dict(),
                'best_acc': best_loss,
            }
        prev_loss = current_loss
        scheduler.step(current_loss)
    print(best_loss)
    torch.save(state, './best_full' + suffix + '.pth')


# Validation step
def validation(model, device, validation_dataset):
    # disable grads + batchnorm + dropout
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
        print(type(val_loss))
        avg_val_loss = torch.mean(torch.stack([x for x in val_loss]))
    return avg_val_loss


# Testing step
# TODO: Implement the necessary accuracy metrics,
def test_model(model, device, test_dataset):
    ac = 0
    accuracy_metrics = Accuracy()
    correct = 0
    total = 0
    # disable gradients
    torch.set_grad_enabled(False)
    model.eval()
    with torch.no_grad():
        for batch_idx, test_batch in enumerate(test_dataset):
            # get the input and label
            x, y = test_batch
            #x, y = x.to(device), y.to(device)
            # perform forward pass
            y_hat = model(x)
            # the class with the highest energy is chosen as prediction
            _, predicted = torch.max(y_hat.data, 1)
            #predicted = predicted.to(device)
            ac += accuracy_metrics(predicted, y).item()
            total += y.size(0)
            correct += (predicted == y).sum().item()
    simple_accuracy = correct / total
    print("simple accuracy: ", simple_accuracy, ", accuracy: ", ac/total, " AC", ac)


def main():
    # Set this variable to define the noise function
    train_transform = None
    # Set this variable to TGrue to load the weights saved
    load_weights = True
    transform = 1
    # num_of_GPUS = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('Device state:', device)

    # number of max epochs
    epochs = 100

    # model
    model = ResNet18().to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, mode='min', verbose=True)  # , step_size=2, gamma=0.95)

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if transform == 0:
        train_transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        print("Reached")
        noise = CustomNoise()
        train_transform = noise.single_frequency_n(8, "h", 0.1)

    # get dataset TODO (Change root to be adaptable)
    train_data = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True,
                                              transform=train_transform)
    test_data = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=test_transform)

    # training/validation split
    train_data, val_data = torch.utils.data.random_split(train_data, [45000, 5000])

    # define DataLoader
    train_dataset = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=2)# num_of_GPUS * 4)
    val_dataset = DataLoader(val_data, batch_size=100, num_workers=2)# num_of_GPUS * 4)
    test_dataset = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)# num_of_GPUS * 4)

    # train model
    print("Training the model")
    # train_model(model, device, epochs, 30, train_dataset, val_dataset, optimizer, scheduler, "_EXP4_HIGH8")
    if load_weights is True:
        weights = torch.load('best_full_EXP1_HIGH.pth')
        model.load_state_dict(weights['model'])
    # test model
    # model = model.to("cpu")
    # test_model(model, device, test_dataset)
    torch.set_grad_enabled(False)
    model.eval()
    dataset_builder = DatasetBuilder(name='cifar10', root_path='data/cifar10')
    model = model.to(device)
    create_fourier_heatmap(model, dataset_builder, 31, 31, 4.0, 'l2', -1, 500, 4, "", fig_title="")

if __name__ == '__main__':
    main()


"""
#best_acc = 0  # best test accuracy


def load_checkpoint():
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('../model/checkpoint/best.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    # for epoch in range(start_epoch, start_epoch + 200):
    #    train(epoch, trainloader)
    #    test(epoch, testloader)
    #    scheduler.step()



# Save checkpoint.
global best_acc
acc = 100. * correct / total
if acc > best_acc:
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if acc >= 0.9:
        print(acc)
        torch.save(state, './checkpoint/best.pth')
    else:
        torch.save(state, './checkpoint/ckpt.pth')
    best_acc = acc
    print(best_acc)




TODO
- save the CIFAR10-C, various accuracy, heatmap and mean corruption error to a file

Questions:
- Does putting the val_data in CPU ok to do or does it also have to be in the GPU?
- When I put eval in anopther function is it still valid in main

"""
