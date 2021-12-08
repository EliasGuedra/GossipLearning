import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import MNIST
import copy
from tqdm import tqdm
import random


class Net(nn.Module):
    def __init__(self): 
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



def train(net, trainloader, epochs, name = 'Random net') -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    print(name + ' is training...')
    for _ in range(epochs):
        #print(test(net, testloader))
        tl = tqdm(trainloader)
        i = 0
        losses = [1 for _ in range(1000)]
        for images, labels in tl:
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            losses[i%1000] = loss.item()
            if i % 100 == 0:
                tl.set_postfix({'loss': sum(losses)/1000})
            i+=1




def test(net, testloader, name):
    """Validate the network on the entire test set."""
    print('Evaluating ' + name)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images, labels
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def train2agents(agent1, agent2, data1, data2):
    i = 0
    j = 0
    while i < len(data1)-2000 or j < len(data2)-2000:
        agent1.data = Subset(data1, list(range(i, i+1000)))
        i += 1000
        agent1.train()
        agent2.agregate(agent1.CurrentModel.named_parameters(), id=agent1.id)
        agent2.data = Subset(data2, list(range(j, j+1000)))
        agent2.train()
        agent1.agregate(agent2.CurrentModel.named_parameters(), id=agent2.id)
        j+=1000



def trainListOfAgents(agents, dataset, examples_per_round=200):
    i = 0
    while i*examples_per_round < len(dataset):
        for a in range(len(agents)):
            agents[a].agregate(agents[a-1].CurrentModel.named_parameters(), id=agents[a-1].id)
            agents[a].data = Subset(dataset, range(i*examples_per_round, (i+1)*examples_per_round))
            agents[a].train()
            i+=1



def trainListOfAgents_uniqueLabels(agents, datasets, examples_per_round=200, rounds=100_000):
    i = 0
    trained_on = [0 for _ in range(10)]
    while i < rounds:
        for a in range(len(agents)):
            agents[a].agregate(agents[a-1].CurrentModel.named_parameters(), id=agents[a-1].id)
            agents[a].data = Subset(datasets[a], range(trained_on[a], trained_on[a] + examples_per_round))
            trained_on[a] += examples_per_round            
            agents[a].train()
            i+=1


def trainListOfAgentsRandom(agents, dataset, examples_per_round=200, rounds=100_000):
    i = 0
    while i < rounds:
        i+=1
        nr1 = random.randint(0, 9)
        nr2 = random.randint(0, 9)

        agents[nr2].agregate(agents[nr1].CurrentModel.named_parameters(), id=agents[nr1].id)
        agents[nr2].data =  Subset(dataset, range(i*examples_per_round, (i+1)*examples_per_round))
        print("We have trained " + str(i) + ' times')
        agents[nr2].train()


def trainListOfAgentsRandom_uniqueLabels(agents, datasets, examples_per_round=200, rounds=100_000):
    i = 0
    trained_on = [0 for _ in range(10)]
    while i < rounds:
        i+=1
        nr1 = random.randint(0, 9)
        nr2 = random.randint(0, 9)

        agents[nr2].agregate(agents[nr1].CurrentModel.named_parameters(), id=agents[nr1].id)
        agents[nr2].data = Subset(datasets[nr2], range(trained_on[nr2], trained_on[nr2] + examples_per_round))
        trained_on[nr2] += examples_per_round
        agents[nr2].train()
        print("We have trained " + str(i) + ' times')





