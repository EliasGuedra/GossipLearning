import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import MNIST
import copy
from tqdm import tqdm
import random
from strategies import train, test


class agent:
    def __init__(self, model, data = None, id = "Some Agent"):
    #id should maybe be replaced with IP.
        self.id = id
        self.CurrentModel = model
        self.LastModel = self.CurrentModel
        self.data = data
        self.parameters_dict = dict(model.named_parameters())

    def agregate(self, forreignParameters, beta=0.5, id='some model'):
        ##Beta says how big of an influence the forreignModel should have in the agregation.
        ##It goes between 0 and 1 where 0.5 gives the average between the models.
        beta = beta

        self.parameters_dict = dict(self.CurrentModel.named_parameters())


        for name, parameter in forreignParameters:
            if name in self.parameters_dict:
                self.parameters_dict[name].data.copy_(beta*parameter.data + (1-beta)*self.parameters_dict[name].data)

        self.CurrentModel.load_state_dict(self.parameters_dict)

        print(self.id + " agregated weights from " + id)

    def train(self):
        ##Train on your current data
        self.dataloader = DataLoader(self.data, batch_size=1, shuffle=True)
        train(self.CurrentModel, self.dataloader, epochs=1, name=self.id)

    def evaluate(self, testloader):
        loss, accuracy = test(self.CurrentModel, testloader, self.id)
        print('Loss:     ' + str(loss))
        print('Accuracy: ' + str(accuracy))
        return loss, accuracy


    def send(self, id):
    ##ROS-code that sends data to annother agent.
        return 1


