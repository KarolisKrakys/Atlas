import os
import torch
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d, Sigmoid, Dropout, Conv1d
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from dataset import featureDataset



device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 60
EPOCH = 30
# class NeuralNetwork(Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.linear_relu_stack = Sequential(
#             BatchNorm1d(48, affine=False),
#             Linear(48, 1024),
#             ReLU(),
#             Dropout(0.5),
#             Linear(1024, 1024),
#             ReLU(),
#             Linear(1024, 128),
#             ReLU(),
#             Linear(128, 1)
#         )
#     def forward(self, x):
#         output = self.linear_relu_stack(x)
#         return output
class NeuralNetwork(Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = Sequential(
            Conv1d(1, 16, 4, stride=3),
            ReLU(),
            Conv1d(16, 32, 4, stride=3),
            ReLU()
        )
        self.part2  = Sequential(
            Linear(7680, 128),
            ReLU(),
            Dropout(0.4),
            Linear(128, 64),
            ReLU(),
            Linear(64, BATCH_SIZE)
        )
    
    def forward(self, x):
        x = self.linear_relu_stack(x)
        x = x.flatten()
        output = self.part2(x)
        # print(f'output shape is {output.size()}')
        return output.unsqueeze(1)
    
model = NeuralNetwork().to(device)
dataset = featureDataset()
train_set, test_set = random_split(dataset, [2400,600])

training_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=64)

loss_fn = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=1e-9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)

def train_one_epoch(model):
    running_loss = 0
    last_loss = 0
    total_correct = 0

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()

        outputs = model(inputs.float()).float()
        loss = loss_fn(outputs, labels)
        loss.backward(retain_graph=True)

        optimizer.step()

        with torch.no_grad():
            running_loss += loss.item()
            pred_y= Sigmoid()(outputs)
            pred_y = pred_y >= 0.5
            correct = torch.sum(labels == pred_y)
            total_correct += correct 

    accuracy = total_correct/ 3000

    print(f'loss is {running_loss}, accuracy is {accuracy*100}%')
    return last_loss


for epoch in range(EPOCH):
    avg_loss = train_one_epoch(model)

path = 'weight.pt'
torch.save(model.state_dict(), path)
