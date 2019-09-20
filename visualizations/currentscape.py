# This is a script for plotting currentscapes from CxSystem result files. Only pointlike neurons!
# For computing currents it needs: membrane voltage, conductances, reversal potentials.
# Check that inward currents are downwards and outward currents upwards in the final plot (convention).
#
# Currentscape plotting code adapted (tbh, copy-pasted) from
# https://datadryad.org/resource/doi:10.5061/dryad.d0779mb
#
# For example currentscapes, see Alonso & Marder 2019 eLife.

import bz2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from brian2.units import *


def unpack_results(filename):
    fi = bz2.BZ2File(filename, 'rb')
    data = pd.read_pickle(fi, compression=None)  # Somehow data is with Brian units! Hooray!

    # return spike_monitors, state_monitors  TODO? :)
    return data


def compute_currents(data, cell_type, cell_ix, leak_params=None, visualize=False):
    # Set some constants
    E = {'ge_soma': 0 * mV, 'gi_soma': -75 * mV, 'g_nmda_soma': 0 * mV, 'g_gabab_soma': -93 * mV}

    # These have to be added as well due to technical reasons related to CxSystem
    scaling = {'ge_soma': 1, 'gi_soma': 1, 'g_nmda_soma': 0.4, 'g_gabab_soma': 0.2}
    B = lambda x: (1 / (1 + exp(-62 * (x / volt)) * (1 / 3.57)))

    # Extract data and vm
    all_synaptic_conductances = ['ge_soma', 'gi_soma', 'g_nmda_soma', 'g_gabab_soma']
    vm = data['vm_all'][cell_type]['vm'][:, cell_ix]
    common_time = data['vm_all'][cell_type]['t']
    N_datapoints = len(vm)

    # Extract synaptic conductances
    conductance_traces = dict()
    for cond_type in all_synaptic_conductances:
        try:
            conductance_traces[cond_type] = data[cond_type + '_all'][cell_type][cond_type][:, cell_ix]
            print('Extracted %s conductance' % cond_type)
        except KeyError:
            conductance_traces[cond_type] = np.zeros(N_datapoints)

    # Add leak as well
    if leak_params is not None:
        conductance_traces['g_leak_soma'] = np.ones(N_datapoints) * leak_params['g_leak']
        E['g_leak_soma'] = leak_params['E_leak']

    # Compute currents
    current_traces = dict()

    for cond_type in all_synaptic_conductances:
        current_traces[cond_type] = conductance_traces[cond_type] * scaling[cond_type] * (E[cond_type] - vm)

        if cond_type == 'ge_nmda_soma':  # For some reason, the ge_nmda in CxSystem is not scaled by the Mg-block fn B
            current_traces[cond_type] = current_traces[cond_type] * B(vm)


    current_traces['g_leak_soma'] = conductance_traces['g_leak_soma'] * (E['g_leak_soma'] - vm)



    if visualize:
        plt.figure()
        plt.plot(common_time, vm / mV)
        plt.show(block=False)

        plt.subplots(2, 4)
        for i, cond_type in enumerate(all_synaptic_conductances, start=1):
            plt.subplot(2, 4, i)
            plt.plot(common_time, conductance_traces[cond_type] / nS)
            plt.title(cond_type)

        for i, cond_type in enumerate(all_synaptic_conductances, start=5):
            plt.subplot(2, 4, i)
            plt.plot(common_time, current_traces[cond_type] / pA)

        plt.show(block=True)

    currents_array = np.array([values/pA for name, values in current_traces.items()])

    return vm/mV, currents_array

def plotCurrentscape(voltage, currents):
    # make a copy of currents
    # CURRENTSCAPE CALCULATION STARTS HERE.
    curr = np.array(currents)
    cpos = curr.copy()
    cpos[curr < 0] = 0
    cneg = curr.copy()
    cneg[curr > 0] = 0

    normapos = np.sum(np.abs(np.array(cpos)), axis=0)
    normaneg = np.sum(np.abs(np.array(cneg)), axis=0)
    npPD = normapos
    nnPD = normaneg
    cnorm = curr.copy()
    cnorm[curr > 0] = (np.abs(curr) / normapos)[curr > 0]
    cnorm[curr < 0] = -(np.abs(curr) / normaneg)[curr < 0]

    resy = 1000
    impos = np.zeros((resy, np.shape(cnorm)[-1]))
    imneg = np.zeros((resy, np.shape(cnorm)[-1]))

    times = arange(0, np.shape(cnorm)[-1])
    for t in times:
        lastpercent = 0
        for numcurr, curr in enumerate(cnorm):
            if (curr[t] > 0):
                percent = int(curr[t] * (resy))
                impos[lastpercent:lastpercent + percent, t] = numcurr
                lastpercent = lastpercent + percent
    for t in times:
        lastpercent = 0
        for numcurr, curr in enumerate(cnorm):
            if (curr[t] < 0):
                percent = int(np.abs(curr[t]) * (resy))
                imneg[lastpercent:lastpercent + percent, t] = numcurr
                lastpercent = lastpercent + percent
    im0 = np.vstack((impos, imneg))
    # CURRENTSCAPE CALCULATION ENDS HERE.

    # PLOT CURRENTSCAPE
    fig = plt.figure(figsize=(3, 4))

    # PLOT VOLTAGE TRACE
    xmax = len(voltage)
    swthres = -70
    ax = plt.subplot2grid((7, 1), (0, 0), rowspan=2)
    t = arange(0, len(voltage))
    plt.plot(t, voltage, color='black', lw=1.)
    plt.plot(t, np.ones(len(t)) * swthres, ls='dashed', color='black', lw=0.75)
    plt.vlines(1, -70, -40, lw=1)
    plt.ylim(-90, 30)
    plt.xlim(0, xmax)
    plt.axis('off')

    # PLOT TOTAL INWARD CURRENT IN LOG SCALE
    ax = plt.subplot2grid((7, 1), (2, 0), rowspan=1)
    plt.fill_between(arange(len((npPD))), (npPD), color='black')
    plt.plot(5. * np.ones(len(nnPD)), color='black', ls=':', lw=1)
    plt.plot(50. * np.ones(len(nnPD)), color='black', ls=':', lw=1)
    plt.plot(500. * np.ones(len(nnPD)), color='black', ls=':', lw=1)
    plt.plot(5000. * np.ones(len(nnPD)), color='black', ls=':', lw=1)
    plt.yscale('log')
    plt.ylim(0.01, 5000)
    plt.xlim(0, xmax)
    plt.axis('off')

    # PLOT CURRENT SHARES
    elcolormap = 'Set1'
    ax = plt.subplot2grid((7, 1), (3, 0), rowspan=3)
    plt.imshow(im0[::1, ::1], interpolation='nearest', aspect='auto', cmap=elcolormap)
    plt.ylim(2 * resy, 0)
    plt.plot(resy * np.ones(len(npPD)), color='black', lw=2)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.xlim(0, xmax)
    plt.clim(0, 8)
    plt.axis('off')

    # PLOT TOTAL OUTWARD CURRENT IN LOG SCALE
    ax = plt.subplot2grid((7, 1), (6, 0), rowspan=1)
    plt.fill_between(arange(len((nnPD))), (nnPD), color='black')
    plt.plot(5. * np.ones(len(nnPD)), color='black', ls=':', lw=1)
    plt.plot(50. * np.ones(len(nnPD)), color='black', ls=':', lw=1)
    plt.plot(500. * np.ones(len(nnPD)), color='black', ls=':', lw=1)
    plt.plot(5000. * np.ones(len(nnPD)), color='black', ls=':', lw=1)
    plt.yscale('log')
    plt.ylim(5000, 0.01)
    plt.xlim(0, xmax)
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    #return fig

    plt.savefig('step1_nmda_ca15.pdf', dpi=600)
    # plt.show()

if __name__ == '__main__':
    # STEP 2
    # filename = './data/step2_gabab_Jeigeneric_fig3params_20190910_12114185_calcium_concentration1.5_python_5000ms.bz2'
    # filename = './data/step2_gabab_Jeigeneric_fig3params_20190910_12114336_calcium_concentration2.0_python_5000ms.bz2'
    # filename = './data/argh_step2_nmda_gabab_Jeigeneric_fig3params_20190910_19510952_calcium_concentration1.5_python_5000ms.bz2'
    # filename = './data/argh_step2_nmda_gabab_Jeigeneric_fig3params_20190910_19511103_calcium_concentration2.0_python_5000ms.bz2'

    # STEP 1
    # filename = './data/argh_step1_gabab_customweights_fig3params_20190910_19442923_calcium_concentration1.5_python_5000ms.bz2'
    # filename = './data/argh_step1_gabab_customweights_fig3params_20190910_19443073_calcium_concentration2.0_python_5000ms.bz2'
    filename = './data/arghh_step1_nmda_gabab_customweights_fig3params_20190911_10272978_calcium_concentration1.5_tonic_depol_level1_python_5000ms.bz2'
    # filename = './data/arghh_step1_nmda_gabab_customweights_fig3params_20190911_10273431_calcium_concentration2.0_tonic_depol_level1_python_5000ms.bz2'



    data = unpack_results(filename)
    leak_params = {'g_leak': 4.77 * nS, 'E_leak': -73.66 * mV}
    vm, currents = compute_currents(data, 'NG19_L4_SS_L4', cell_ix=15, leak_params=leak_params, visualize=False)

    t_start = 30000
    t_end = 50000

    plotCurrentscape(vm[t_start:t_end], (-1)*currents[:, t_start:t_end])
