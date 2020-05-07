import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection
import json
import numpy as np
# f = open('local-data/scalabel/timing_data-1588810424898.json')
# data = json.load(f)
# sample = data[-3]
# diffs = []
# names = []
# for p0, p1 in zip(sample[:-1], sample[1:]):
#     t0, n0 = p0['time'], p0['name']
#     t1, n1 = p1['time'], p1['name']
#     curr_names = [n0, n1]
#     curr_names.sort()
#     n0, n1 = curr_names
#     diffs.append(t1 - t0)
#     if n0 != n1:
#         names.append('{}-{}'.format(n0, n1))
#     else:
#         names.append('{}'.format(n0))

#temporary override to get better viz
diffs = [25, 5, 6, 10, 9, 7, 15]
names = ['hub-synchronizer', 'hub', 'hub-bot', 'bot', 'hub-bot', 'hub', 'hub-synchronizer']

#connector names should be alphabetical like a-b
nodes = ["hub-synchronizer", "hub", "hub-bot", "bot"]
color_options = ["B", "G", "R", "Y"]
name_map = {}
for i, node in enumerate(nodes):
    name_map[node] = (i, color_options[i])

current_pos = 0
vertices = []
colors = []
heightDelta = 0.4
for diff, name in zip(diffs, names):
    height, color = name_map[name]
    next_pos = current_pos + diff
    v = [
        (current_pos, height - heightDelta),
        (current_pos, height + heightDelta),
        (next_pos, height + heightDelta),
        (next_pos, height - heightDelta),
        (current_pos, height - heightDelta)
    ]
    vertices.append(v)
    colors.append(color)
    current_pos = next_pos

bars = PolyCollection(vertices, facecolors=colors)

fig, ax = plt.subplots()
ax.add_collection(bars)
ax.autoscale()
loc = mdates.MinuteLocator(byminute=[0,15,30,45])
ax.xaxis.set_major_locator(loc)
# ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 10))
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(nodes)
plt.show()