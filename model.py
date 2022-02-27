import os
import torch
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from dataset import featureDataset



device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuralNetwork(Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = Sequential(
            BatchNorm1d(48, affine=False),
            Linear(48, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 1)
        )
    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

model = NeuralNetwork().to(device)
dataset = featureDataset()
# train_set, test_set = random_split(dataset, [2400,600])

training_loader = DataLoader(dataset, batch_size=64)
# test_loader = DataLoader(test_set, batch_size=64)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

def train_one_epoch(model):
    running_loss = 0
    last_loss = 0
    total_wrong = 0

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        # print(inputs.size())
        labels = labels.float().to(device)
        optimizer.zero_grad()
        
        

        outputs = model(inputs.float()).float()

        loss = loss_fn(outputs, labels)
        loss.backward(retain_graph=True)
        # print(outputs)

        optimizer.step()

        running_loss += loss.item()
        # if i % 1000 == 999: 
        #     last_loss = running_loss / 1000
        #     # print(' batch: {} loss: {}'.format(i + 1, last_loss))
        #     running_loss = 0
    accuracy = total_wrong / 2400

    print(f'loss is {running_loss}, accuracy is {accuracy*100}%')
    return last_loss

epoch_number = 0
best_test_loss = 1000000 

for epoch in range(100):
    avg_loss = train_one_epoch(model)

    running_test_loss = 0.0
    # for i, test_data in enumerate(test_loader):
    #     test_inputs, test_labels = test_data
    #     test_outputs = model(test_inputs)
    #     test_loss = loss_fn(test_outputs, test_labels)
    #     running_test_loss += test_loss
    # avg_test_loss = running_test_loss / (i+1)

    # if avg_test_loss < best_test_loss:
    #     best_test_loss = avg_test_loss
    #     model_path = 'model_{}'.format(epoch_number)
    #     torch.save(model.state_dict(), model_path)

    # epoch_number += 1


# test_total_loss = 0
# for i, test_data in enumerate(test_loader):
#     test_inputs, test_labels = test_data
#     test_outputs = model(test_inputs)
#     test_loss = loss_fn(test_outputs, test_labels)
#     test_total_loss += test_loss
# model_path = 'model'
# torch.save(model.state_dict(), model_path)














