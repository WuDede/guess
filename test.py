import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_color(x, y):
    if ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) < 0.09:
        return 'r'
    else:
        return 'g'


global points
global scat
global colors

points = np.random.rand(10, 2)
colors = []
for _x, _y in points:
    colors.append(get_color(_x, _y))

fig, ax = plt.subplots()

textTemplate = 'Count = %d'
textCount = ax.text(0.8, 0.9, "")


def scat_init():
    global scat
    ax.axis([0, 1, 0, 1])
    textCount.set_text("")
    scat = ax.scatter(points[:, 0], points[:, 1], c=colors)
    return scat


def scat_update(f):
    global scat
    global points
    global colors
    newp = np.random.rand(1, 2)
    points = np.append(points, newp, 0)
    colors.append(get_color(newp[0, 0], newp[0, 1]))
    scat = ax.scatter(points[:, 0], points[:, 1], c=colors)
    textCount.set_text(textTemplate % len(points[:, 0]))
    return scat


ani = FuncAnimation(fig, scat_update, interval=10, init_func=scat_init)
plt.show()
