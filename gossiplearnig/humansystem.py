import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import MNIST
from tqdm import tqdm
from random import randint

from agent import agent
from strategies import Net, train, test, train2agents, trainListOfAgentsRandom, trainListOfAgents

from copy import deepcopy

import json

import os

import sys
 
import pygame
from pygame.locals import *




class human(agent):
    def __init__(self, model=Net(), data=None, id="Some Agent", x=500, y=500):
        super(human, self).__init__(model, data, id)
        self.x = x
        self.y = y
        self.connection = 0
        self.vel_x = randint(-2, 2)    
        self.vel_y = randint(-2, 2)
        self.logdata = {}
        self.logdata['x'] = []
        self.logdata['y'] = []
        self.logdata['connections'] = []
        self.trained = 0
        self.trainedon = 0


    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y

        if self.x > 1000 or self.x < 0:
            self.vel_x *= -1
            self.vel_y = randint(-2, 2)
        if self.y > 1000 or self.y < 0:
            self.vel_y *= -1
            self.vel_x = randint(-2, 2)

        self.logdata['x'].append(self.x)
        self.logdata['y'].append(self.y)
        self.logdata['connections'].append(0)
        self.trained -= 1
    def save(self, location = ""):
        with open(location + self.id + '.json', 'w') as outfile:
            json.dump(self.logdata, outfile)









transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)


trainset = MNIST("./dataset", train=True, download=True, transform=transform)
testset = MNIST("./dataset", train=False, download=True, transform=transform)


#train2agents(a, b, Subset(trainset, range(4000)), Subset(trainset, range(4000)))

loader = DataLoader(trainset, batch_size=1, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

subset = Subset(trainset, range(30000))





humans = []
net = Net()


for i in range(10):
    humans.append(human(model = deepcopy(net), id='Human_nr_' + str(i+1), x=randint(0, 1000), y=randint(0, 1000)))



"""
for _ in range(5000):
    for human in humans:
        human.update()
    for i in range(len(humans)):
        for j in range(i, len(humans)):
            if (humans[i].x-humans[j].x)**2 + (humans[j].y-humans[i].y)**2 < 2500:
                humans[i].logdata['connections'][-1] = j



for human in humans:
    human.save(location="C:/Users/elias/OneDrive/Skrivbord/gossiplearnig/data/")
"""






pygame.init()
 
fps = 60
fpsClock = pygame.time.Clock()
 
width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))



def connect(c1, c2, f):
    pygame.draw.line(screen, (255, 30, 30), (c1.x, c1.y), (c2.x, c2.y))


nr_of_trained_examples = 0

# Game loop.
while nr_of_trained_examples < 10_000:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    # Update.
    for h in humans:
        h.update()
    # Draw.
    connections = []
    for j in range(len(humans)):
        pygame.draw.circle(screen, (255, 255, 255, 10), (humans[j].x, humans[j].y), 10)
        for i in range(len(humans)):
            if (humans[i].x-humans[j].x)**2 + (humans[j].y-humans[i].y)**2 < 2500 and humans[i] != humans[j] and humans[j].trained < 0:
                connections.append((i, j))

    
    for i, j in connections:
        print(i, j)
        pygame.draw.line(screen, (255, 30, 30), (humans[j].x, humans[j].y), (humans[i].x, humans[i].y))


        for i, j in connections:
            humans[j].agregate(humans[i].CurrentModel.named_parameters(), id=humans[i].id)
            humans[j].data = (Subset(trainset, range(nr_of_trained_examples, nr_of_trained_examples+200)))
            humans[j].train()
            nr_of_trained_examples += 200
            humans[j].trainedon += 200
            print("We have trained on " + str(nr_of_trained_examples) + ' examples')
            humans[j].trained = 40

    pygame.display.flip()
    fpsClock.tick(fps)