from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_style('whitegrid')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class CalciumSynapse(object):

    excitatory_groups = ['PC', 'SS']
    steep_post_inhibitory_groups = ['MC']
    shallow_post_inhibitory_groups = ['BC']
    steep_post = excitatory_groups + steep_post_inhibitory_groups
    shallow_post = shallow_post_inhibitory_groups

    def __init__(self, pre_group_type, post_group_type):
        self.output_synapse = {'pre_group_type': pre_group_type,
                               'post_group_type': post_group_type}
        self._set_calcium_dependency()


    def _set_calcium_dependency(self):
        """
        Sets the dissociation constant for calcium-dependent modulation of synaptic weight
        # NB! It's not only E->E and E->I connections that get scaled. Goes both ways. See Markram suppl p16.
        """

        # excitatory_groups = ['PC', 'SS', 'VPM']
        # steep_inhibitory_groups = ['MC']
        # shallow_inhibitory_groups = ['BC']
        # steep_post = excitatory_groups + steep_inhibitory_groups
        # shallow_post = shallow_inhibitory_groups

        if self.output_synapse['pre_group_type'] in CalciumSynapse.excitatory_groups and self.output_synapse['post_group_type'] in CalciumSynapse.steep_post:
            self._K12 = 2.79
        elif self.output_synapse['pre_group_type'] in CalciumSynapse.steep_post_inhibitory_groups and self.output_synapse['post_group_type'] in CalciumSynapse.excitatory_groups:
            self._K12 = 2.79
        elif self.output_synapse['pre_group_type'] in CalciumSynapse.excitatory_groups and self.output_synapse['post_group_type'] in CalciumSynapse.shallow_post:
            self._K12 = 1.09
        elif self.output_synapse['pre_group_type'] in CalciumSynapse.shallow_post_inhibitory_groups and self.output_synapse['post_group_type'] in CalciumSynapse.excitatory_groups:
            self._K12 = 1.09
        else:
            self._K12 = np.average([2.79, 1.09])


    def scale(self, ca, cw):
        """
        Scales synaptic weight depending on calcium level

        """

        ca0 = 2.0  # Baseline calcium level

        calcium_factor = (pow(ca, 4) / (pow(self._K12, 4) + pow(ca, 4))) / (pow(ca0, 4) / (pow(self._K12, 4) + pow(ca0, 4)))
        final_synapse_strength = cw * calcium_factor

        return final_synapse_strength


    def _set_utilization_factor(self, is_facilitating=False, ca=2.0):
        pass
        # excitatory_groups = ['PC', 'SS', 'in']
        # inhibitory_groups = ['BC', 'MC', 'L1i']
        #
        # if is_facilitating is False:
        #     U_E = 0.5
        #     U_I = 0.25
        #
        #     if self.output_synapse['pre_group_type'] in excitatory_groups:
        #         self.output_namespace['U'] = self._scale_by_calcium(ca, U_E)
        #     elif self.output_synapse['pre_group_type'] in inhibitory_groups:
        #         self.output_namespace['U'] = self._scale_by_calcium(ca, U_I)
        #     else:
        #         print 'Warning! Unrecognized group type %s will have outbound synapses with averaged utilization factor' % \
        #               self.output_synapse['pre_group_type']
        #         U = np.average([U_E, U_I])
        #         self.output_namespace['U'] = self._scale_by_calcium(ca, U)
        #
        # else:
        #     U_f = self.value_extractor(self.physio_config_df, 'U_f')
        #     self.output_namespace['U_f'] = self._scale_by_calcium(ca, U_f)


if __name__ == '__main__':
    EtoBC = CalciumSynapse('PC', 'BC')  # shallow
    EtoE = CalciumSynapse('PC', 'PC')  # steep
    BCtoE = CalciumSynapse('BC', 'PC')
    EtoMC = CalciumSynapse('PC', 'MC')
    MCtoE = CalciumSynapse('MC', 'PC')

    plt.subplots(2,1)

    plt.subplot(211)
    ax1 = plt.gca()
    x_range = np.linspace(1.0, 2.5, 100)
    y_steep = [EtoE.scale(x, 1.0) for x in x_range]
    y_shallow = [EtoBC.scale(x, 1.0) for x in x_range]

    plt.plot(x_range, y_steep, c='0.4', linewidth=2, label='steep')
    plt.plot(x_range, y_shallow, c='0.7', linewidth=2, label='shallow')
    plt.legend()

    plt.subplot(212, sharex=ax1)
    x_range = np.linspace(1.0, 2.5, 100)
    y_EtoE = np.array([EtoE.scale(x, 1.0) for x in x_range])
    y_EtoBC = np.array([EtoBC.scale(x, 1.0) for x in x_range])
    y_BCtoE = np.array([EtoBC.scale(x, 0.5) for x in x_range])

    y_EtoMC = np.array([EtoMC.scale(x, 1.0) for x in x_range])
    y_MCtoE = np.array([MCtoE.scale(x, 0.5) for x in x_range])

    plt.plot(x_range, y_EtoBC/y_EtoE, label='PC-BC/PC-PC')
    plt.plot(x_range, y_BCtoE/y_EtoE, label='BC-PC/PC-PC')
    plt.plot(x_range, y_EtoMC / y_EtoE, label='PC-MC/PC-PC')
    plt.plot(x_range, y_MCtoE / y_EtoE, label='MC-PC/PC-PC')

    plt.plot()

    plt.legend()

    plt.show()