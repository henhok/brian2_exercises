import pandas as pd
from brian2 import *
import matplotlib.pyplot as plt


def get_rheo(neuron_params, show_fi_curve=False):
    num_neurons = 500
    duration = 1 * second
    refr_time = 4 * ms

    # Parameters
    print neuron_params

    g_leak = neuron_params['g_leak'] * nS
    Vr = neuron_params['Vr'] * mV
    DeltaT = neuron_params['DeltaT'] * mV
    VT = neuron_params['VT'] * mV
    C = neuron_params['C'] * 10 * pF
    Vcut = neuron_params['Vcut'] * mV
    V_res = neuron_params['V_res'] * mV

    eqs = '''
	dvm/dt = (g_leak*(Vr-vm) + g_leak * DeltaT * exp((vm-VT) / DeltaT) + I) / C : volt (unless refractory)
	I : amp
	'''
    group = NeuronGroup(num_neurons, eqs,
                        threshold='vm > ' + repr(Vcut),
                        reset='vm = ' + repr(V_res),
                        refractory=refr_time,
                        method='euler')

    group.vm = Vr
    group.I = '0.5*nA * i / num_neurons'

    monitor = SpikeMonitor(group)

    run(duration)

    rheo_idx = min(np.where(monitor.count > 0)[0])

    if show_fi_curve is True:
        plt.figure()
        plt.plot(group.I / nA, monitor.count / duration, '.')
        plt.xlabel('I (nA)')
        plt.ylabel('Firing rate (sp/s)')
        plt.show()

    return group.I[rheo_idx]


if __name__ == '__main__':

    data = pd.read_csv('step1_model_params.csv', index_col='mtype')

    for index, row in data.iterrows():
        print 'Now running ' + str(index)
        print row
        rheo = get_rheo(dict(row))

        print '-> ' + str(rheo)
        data.loc[index, 'rheobase'] = rheo
