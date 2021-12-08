import json

import sys
 
import pygame
from pygame.locals import *



frame = 0
class circle:
    def __init__(self, x=[] , y=[], connections=[]):
        self.x = x
        self.y = y
        self.connections = connections



circles = []


for i in range(30):
    f = open('data/Human_nr_' + str(i+1) + '.json')
    data = json.load(f)
    circles.append(circle(data['x'], data['y'], data['connections']))





 
pygame.init()
 
fps = 60
fpsClock = pygame.time.Clock()
 
width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))



def connect(c1, c2, f):
    pygame.draw.line(screen, (255, 30, 30), (c1.x[f%5000], c1.y[f%5000]), (c2.x[f%5000], c2.y[f%5000]))


 
# Game loop.
while True:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # Update.
    frame+=1
    # Draw.
    for c in circles:
        pygame.draw.circle(screen, (255, 255, 255, 10), (c.x[frame%5000], c.y[frame%5000]), 10)
        if c.connections != 0:
            connect(c, circles[c.connections[frame%5000]], frame)

    pygame.display.flip()
    fpsClock.tick(fps)