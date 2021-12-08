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



def train(net, trainloader, epochs) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    print('Training...')
    for _ in tqdm(range(epochs)):
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









model1 = Net()
model2 = Net()


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)


trainset = MNIST("./dataset", train=True, download=True, transform=transform)
test = MNIST("./dataset", train=False, download=True, transform=transform)

subset = Subset(trainset, list(range(200)))


loader = DataLoader(trainset, batch_size=1, shuffle=True)
testloader = DataLoader(test, batch_size=1, shuffle=True)

def test(net, testloader):
    """Validate the network on the entire test set."""
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






class agent:
    def __init__(self, model, data):
    #Should maybe be replaced with IP.
        self.id = None
        self.CurrentModel = model
        self.LastModel = self.CurrentModel
        self.data = data
        self.parameters_dict = dict(model.named_parameters())

    def agregate(self, forreignModel, beta=0.5):
        ##Beta says how big of an influence the forreignModel should have in the agregation.
        ##It goes between 0 and 1 where 0.5 gives the average between the models.
        beta = beta

        forreignParameters = forreignModel.named_parameters()

        self.parameters_dict = dict(self.CurrentModel.named_parameters())



        for name, parameter in forreignParameters:
            if name in self.parameters_dict:
                self.parameters_dict[name].data.copy_(beta*parameter.data + (1-beta)*self.parameters_dict[name].data)

        self.CurrentModel.load_state_dict(self.parameters_dict)
        


    def train(self):
        ##Train on your current data
        self.dataloader = DataLoader(self.data, batch_size=1, shuffle=True)
        train(self.CurrentModel, self.dataloader, epochs=1)



    def send(self, id):
    ##ROS-code that sends data to annother agent.
        return 1


a = agent(model1, subset)

b = agent(model2, trainset)

agents = []

for i in range(10):
    agents.append(agent(Net(), trainset))

zero2four = [i for i in range(len(trainset)) if trainset[i][1] < 5]
five2nine = [i for i in range(len(trainset)) if trainset[i][1] >= 5]

subsets = []
for k in range(10):
    subsets.append(Subset(trainset, [i for i in range(len(trainset)) if trainset[i][1] == k]))

zero2four = Subset(trainset, zero2four)
five2nine = Subset(trainset, five2nine)

def train2agents(agent1, agent2):
    i = 0
    j = 0
    while i < len(zero2four)-2000 or j < len(five2nine)-2000:
        agent1.data = Subset(five2nine, list(range(i, i+1000)))
        i += 1000
        agent1.train()
        agent2.agregate(agent1.CurrentModel)
        agent2.data = Subset(zero2four, list(range(j, j+1000)))
        agent2.train()
        agent1.agregate(agent2.CurrentModel)
        j+=1000



def trainListOfAgents(agents):
    i = 0
    while i*200 < len(trainset)-2000:
        i+=1
        for a in range(len(agents)):
            agents[a].agregate(agents[a-1].CurrentModel)
            agents[a].data = Subset(subsets[a], range(i*200, (i+1)*200))
            agents[a].train()



def trainListOfAgentsRandom(agents):
    i = 0
    trained_on = [0 for _ in range(10)]
    while i*200 < len(trainset)-2000:
        i+=1
        nr1 = random.randint(0, 9)
        nr2 = random.randint(0, 9)

        agents[nr2].agregate(agents[nr1].CurrentModel)
        agents[nr2].data = Subset(subsets[nr2], list(range(trained_on[nr2], trained_on[nr2] + 200)))
        trained_on[nr2] += 200
        agents[nr2].train()



