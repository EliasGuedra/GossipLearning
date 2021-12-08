from processing_py import *
import json

frame = 0
app = App(1000,1000) # create window: width, height

class circle:
    def __init__(self, x=[] , y=[]):
        self.x = x
        self.y = y



circles = []


for i in range(30):
    f = open('data/Human_nr_' + str(i+1) + '.json')
    data = json.load(f)
    circles.append(circle(data['x'], data['y']))





while(True):
    app.background(0,0,0) # set background:  red, green, blue
    for c in circles:
        app.fill(255)
        app.ellipse(c.x[frame], c.y[frame], 10, 10)
    app.redraw()

    frame += 1





