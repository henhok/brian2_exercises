import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

search_logfile = 'adex_fits/L4SS_200gens.csv'

data = pd.read_csv(search_logfile)

a_var = data.a
b_var = data.b

# Bin all (a,b) to some cat
hist, a_edges, b_edges = np.histogram2d(a_var, b_var, bins=[20,40])
a_digi = np.digitize(a_var, a_edges)
b_digi = np.digitize(b_var, b_edges)

# Pick minimum for each cat
fitness_sum = data.fitness_sum

new_data = pd.DataFrame([a_digi, b_digi, fitness_sum]).T
new_data.columns=['a','b','fitness_sum']
a = new_data.pivot_table('fitness_sum', index='b', columns='a', aggfunc=min)

sns.heatmap(a, vmin=25, vmax=100)
plt.show()

# As = np.linspace(min(a_var), max(a_var), 1000)
# Bs = np.linspace(min(b_var), max(b_var), 1000)
# A, B = np.meshgrid(As, Bs)
#
# smooth_fitsums = Rbf(a_var, b_var, z_var, epsilon=2)
# Z = smooth_fitsums(A, B)
#
# plt.pcolor(A, B, Z, cmap=cm.jet)
# plt.scatter(a_var, b_var, 10, z_var, cmap=cm.jet)
# plt.colorbar()
#
# plt.show()