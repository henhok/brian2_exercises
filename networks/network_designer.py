from brian2 import *
import numpy as np
import matplotlib.pyplot as plt


grid_width = 1*mm
grid_height = 1*mm

n_calpha = 10000
n_magno = 100

magno = NeuronGroup(n_magno, '''x: meter
                                y : meter''')

Calpha = NeuronGroup(n_calpha, '''x : meter
                                  y : meter''')

Calpha.x = 'rand()*grid_width'
Calpha.y = 'rand()*grid_height'
magno.x = 'rand()*grid_width'
magno.y = 'rand()*grid_height'


syns = Synapses(magno, Calpha)
syns.connect('i != j',
             p='exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2 ) / (2*(100*umeter)**2))')

neuron_idx = np.random.randint(0, n_magno)
# plt.figure()
# plt.plot(magno.x[neuron_idx] / umeter, magno.y[neuron_idx] / umeter, 'o')
# plt.plot(Calpha.x[syns.j[neuron_idx,:]] / umeter, Calpha.y[syns.j[neuron_idx,:]] / umeter, '.')
# plt.xlim([0, grid_width/umeter])
# plt.ylim([0, grid_height/umeter])
#
# plt.show()

random_calpha_idx = np.random.randint(0, n_calpha)
efferents = [len(syns.i[:, idx]) for idx in range(0, n_calpha)]

plt.hist(efferents)
plt.show()

