import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torch
import numpy as np
from torch import nn
import torch.optim as optim
import random
import time
import matplotlib.pyplot as plt

image_transforms = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

dataset = 'ori_datasets'
data = datasets.ImageFolder(root=dataset, transform=image_transforms)
n = len(data)
n_test = random.sample(range(1,n), int(0.1 * n))
test_set = torch.utils.data.Subset(data, n_test)
train_set = torch.utils.data.Subset(data, list(set(range(1,n)).difference(set(n_test))))
train_data = DataLoader(train_set, batch_size = 192, shuffle=True,num_workers = 8)
valid_data = DataLoader(test_set, batch_size = 64, shuffle=True,num_workers = 8)
train_data_size = len(train_set)
valid_data_size = len(test_set)

def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            #因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            for j in range(predictions.shape[0]):
                if(predictions[j]==labels[j]):
                   train_acc+=1

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(tqdm(valid_data)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                for j in range(predictions.shape[0]):
                    if(predictions[j]==labels[j]):
                        valid_acc+=1

        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size

        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        torch.save(model, 'models/'+dataset+'_model_'+str(epoch+1)+'.pt')
    return model, history

if __name__ == "__main__":
    model = torchvision.models.resnet50(pretrained=True)
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,4),
            nn.LogSoftmax(dim=1)
            ) 
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.NLLLoss()
    epoch = 10
    trained_model, history = train_and_valid(model, loss_func, optimizer, epochs=epoch)
    torch.save(history, 'models/'+dataset+'_history.pt')
    
    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(dataset+'_loss_curve.png')
    plt.show()
    
    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(dataset+'_accuracy_curve.png')
    plt.show()



