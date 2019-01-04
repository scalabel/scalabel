from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--num-cpus', '-c', type=int, default=64)
parser.add_argument('--num-actors', '-a', type=int, default=100)
parser.add_argument('--actor-step-size', '-s', type=int, default=8)
parser.add_argument('--num-operations', '-o', type=int, default=5)
args = parser.parse_args()

print('================================')
print('configurations: ')
print(args)
print('================================')

# set_values
num_cpus = args.num_cpus
num_actors = args.num_actors
actor_step_size = args.actor_step_size
num_operations = args.num_operations

# init ray functions
ray.init(num_cpus=num_cpus, ignore_reinit_error=True,
         object_store_memory=20000000000)


@ray.remote
class TestActor(object):
    def __init__(self):
        self.counter = 0

    def reset(self):
        self.counter = 0

    def increment(self):
        time.sleep(0.5)
        self.counter += 1
        return self.counter


number_curr_actors = [int(i * actor_step_size) for i in
                      range(1, num_actors // actor_step_size)]
ray_actors = [TestActor.remote() for i in range(num_actors)]
time.sleep(15)
times = []
for n in number_curr_actors:
    curr_ray_actors = ray_actors[:n]

    for curr_actor in curr_ray_actors:
        curr_actor.reset.remote()

    results = []
    for _ in range(num_operations):
        for curr_actor in curr_ray_actors:
            results.append(curr_actor.increment.remote())
    start_time = time.time()
    results = ray.get(results)
    duration = time.time() - start_time
    times.append(duration)

plt.scatter(number_curr_actors, times)
plt.xticks(number_curr_actors[::2])
plt.xlabel('Number of ray actors')
plt.ylabel('Time (seconds)')
plt.title('Time to run remote function {} times on each actor'.format(
    num_operations))
plt.savefig('ray_res/ray_plot_ncpu_{}_nactors_{}_nstep_{}_nops_{}.pdf'.format(
    num_cpus, num_actors, actor_step_size, num_operations))

print('===============================')
print('times: ')
print(times)
print('number_curr_actors: ')
print(number_curr_actors)
print('===============================')
print('finished!')
