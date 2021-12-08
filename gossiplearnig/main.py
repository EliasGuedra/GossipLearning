import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import MNIST
from tqdm import tqdm


from agent import agent
from strategies import Net, train, test, train2agents, trainListOfAgentsRandom, trainListOfAgents

from copy import deepcopy

a = agent(Net(), id = 'Agent_A')
b = agent(Net(), id = 'Agent_B')

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)


trainset = MNIST("./dataset", train=True, download=True, transform=transform)
testset = MNIST("./dataset", train=False, download=True, transform=transform)


#train2agents(a, b, Subset(trainset, range(4000)), Subset(trainset, range(4000)))

loader = DataLoader(trainset, batch_size=1, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

subset = Subset(trainset, range(30000))


subsets = []
for k in tqdm(range(10)):
    subsets.append(Subset(subset, [i for i in range(len(trainset)) if trainset[i][1] == k]))

net = Net()

agents = []
for i in tqdm(range(10)):
    agents.append(agent(deepcopy(net), trainset, id="Agent"+str(i)))

def create_agents():
    agents = []
    for i in range(10):
        agents.append(agent(deepcopy(net), trainset, id="Agent"+str(i)))
    return agents

#trainListOfAgentsRandom(agents)

def eval_models(models):
    loss = 0
    accuracy = 0
    for model in models:
        l, a = model.evaluate(testloader)
        loss+=l
        accuracy+=a
        print("-"*20)


    print("-"*20)
    print("Total average:\n")
    print("Loss:      "  +  str(loss/len(models)))
    print("Accuracy:  "  +  str(accuracy/len(models)))
