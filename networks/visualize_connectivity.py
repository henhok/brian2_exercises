from __future__ import division
from brian2 import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import bz2
import cPickle as pickle

sns.set_style('whitegrid')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class ConnectivityData(object):
    cell_counts={'PC_L5toL1': 5050,
                 'BC_L5': 558,
                 'MC_L5': 491,
                 'PC_L6toL4': 9825,
                 'PC_L6toL1': 1637,
                 'BC_L6': 813,
                 'MC_L6': 372,
                 'PC_L2toL1': 5877,
                 'BC_L2': 1198,
                 'MC_L2': 425,
                 'PC_L4toL2': 2674,
                 'PC_L4toL1': 1098,
                 'SS_L4': 406,
                 'BC_L4': 329,
                 'MC_L4': 137,
                 'L1i_L1': 338}

    def __init__(self, datafile, keys_to_exclude=['relay_vpm']):
        not_connection_keys = ['Full path', 'positions_all']

        with bz2.BZ2File(datafile, 'rb') as fi:
            data = pickle.load(fi)
            fi.close()

        # Exclude not connection keys and excluded keys
        all_keys_to_exclude = not_connection_keys + keys_to_exclude
        connections_dict = {k: v for k, v in data.items()
                            if not self._startswith_any(k, all_keys_to_exclude)}

        self.connections_df = self._makeConnectivityDataFrame(connections_dict)


    def _startswith_any(self, keyvalue, list):
        for x in list:
            if keyvalue.startswith(x):
                return True
            else:
                pass

        return False

    def _makeConnectivityDataFrame(self, connections_dict_raw):
        # Extract data from raw dictionary
        connections_dict = {k: [v['data'].getnnz(), v['n'], v['data'].getnnz()*v['n']]
                            for k, v in connections_dict_raw.items()}

        connections_df = pd.DataFrame.from_dict(connections_dict, orient='index')
        connections_df.columns = ['n_connections', 'n_syns_per_connection', 'n_synapses']

        # Split "connection name" into presynaptic group, postsynaptic group and target compartment
        pre_and_post = connections_df.index.str.extract('(.*)__to__(.*)_(\w+)', expand=True)
        pre_and_post.index = connections_df.index
        pre_and_post.columns = ['presynaptic_group', 'postsynaptic_group', 'target_compartment']

        # Compute number of synapses per postsynaptic target
        n_syns_per_target = {k: connections_df.loc[k].n_synapses / ConnectivityData.cell_counts[pre_and_post.loc[k].postsynaptic_group]
                             for k in connections_df.index}
        n_syns_per_target = pd.DataFrame.from_dict(n_syns_per_target, orient='index')
        n_syns_per_target.columns = ['n_syns_per_target']

        # Split pre- and postsynaptic group into group type and layer
        # ...



        final_df = pd.concat([connections_df, pre_and_post, n_syns_per_target], axis=1)

        return final_df

    def plotNetwork(self, min_synapses=200):
        # Node size = size of group
        # Edge width = n_synapses_per_target
        c = self.connections_df[self.connections_df.n_syns_per_target >= min_synapses]
        c_edges = [(c.iloc[i].presynaptic_group, c.iloc[i].postsynaptic_group) for i in range(len(c))]

        graph = nx.MultiDiGraph(c_edges)
        agraph = nx.nx_agraph.to_agraph(graph)

        agraph.draw('gugu.png', prog='circo')
        #nx.draw_circular(agraph, with_labels=True)
        #plt.show()


    # def printElements(self):
    #     for k,v in self.connectivity_data.items():
    #         print '%s\t\t\t\t %d\t\t\t\t %d' % (str(k), v['data'].getnnz(), v['n'])


if __name__ == '__main__':
    fixedconn = ConnectivityData('fixed_connections.bz2')
    fixedconn.plotNetwork(min_synapses=100)