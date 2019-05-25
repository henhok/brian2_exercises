from brian2 import *
import pandas as pd

num_neurons = 1000
duration = 2*second

refr_time = 4*ms


def simulatePC(dendritic_extent, Area_tot_pyram, EL, V_res, VT, fract_areas, show_plot=True):
    # Parameters
    # dendritic_extent = 4
    Cm = 1*uF*cm**-2
    gl = 4.2e-5*siemens*cm**-2

    # Area_tot_pyram = 25000 * 0.75 * um**2
    # EL = -72.02*mV
    # V_res = -64.76*mV
    # VT = -53.76*mV
    # Area_tot_pyram = 25000 * 0.75 * um**2
    # EL = -70*mV
    # V_res = -55*mV
    # VT = -42*mV

    Vcut = 20*mV
    DeltaT = 2*mV


    # Dendritic parameters
    neuron_namespace = dict()

    #fract_areas = {2: array([0.354, 0.047, 0.599, 0.01, 0.01])}

    # fract_areas = {1: array([0.2,  0.03,  0.15,  0.2]),
    #                 2: array([0.2,  0.03,  0.15,  0.15,  0.2]),
    #                 3: array([0.2,  0.03,  0.15,  0.09,  0.15,  0.2]),
    #                 4: array([0.2,  0.03,  0.15,  0.15,  0.09,  0.15,  0.2])}

    neuron_namespace['Ra'] = [100, 80, 150, 150, 200] * Mohm
    #neuron_namespace['Ra'] = [100, 100, 100, 100, 100] * Mohm

    neuron_namespace['C'] = fract_areas[dendritic_extent] * \
                                 Cm * Area_tot_pyram * 2
    neuron_namespace['gL'] = fract_areas[dendritic_extent] * \
                                  gl * Area_tot_pyram
    neuron_namespace['taum_soma'] = neuron_namespace['C'][1] / neuron_namespace['gL'][1]

    # WITH ADAPTATION
    eq_template_soma = '''
    dvm/dt = ((gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) +I_dendr + I) / C) : volt (unless refractory)
    I : amp
    '''

    #: The template for the dendritic equations used in multi compartmental neurons, the inside values could be replaced later using "Equation" function in brian2.
    eq_template_dend = '''
    dvm/dt = (gL*(EL-vm) +I_dendr) / C : volt
    '''

    neuron_equ = Equations(eq_template_dend, vm="vm_basal", ge="ge_basal",
                                               gealpha="gealpha_basal",
                                               C=neuron_namespace['C'][0],
                                               gL=neuron_namespace['gL'][0],
                                               gi="gi_basal", geX="geX_basal", gialpha="gialpha_basal",
                                               gealphaX="gealphaX_basal", I_dendr="Idendr_basal")

    # ADAPTIVE
    neuron_equ += Equations(eq_template_soma, gL=neuron_namespace['gL'][1],
                                                ge='ge_soma', geX='geX_soma', gi='gi_soma', gealpha='gealpha_soma',
                                                gealphaX='gealphaX_soma',
                                                gialpha='gialpha_soma', C=neuron_namespace['C'][1],
                                                I_dendr='Idendr_soma',
                                                taum_soma=neuron_namespace['taum_soma'])

    for _ii in range(dendritic_extent + 1):  # extra dendritic compartment in the same level of soma
        neuron_equ += Equations(eq_template_dend, vm="vm_a%d" % _ii,
                                                    C=neuron_namespace['C'][_ii],
                                                    gL=neuron_namespace['gL'][_ii],
                                                    ge="ge_a%d" % _ii,
                                                    gi="gi_a%d" % _ii, geX="geX_a%d" % _ii,
                                                    gealpha="gealpha_a%d" % _ii, gialpha="gialpha_a%d" % _ii,
                                                    gealphaX="gealphaX_a%d" % _ii, I_dendr="Idendr_a%d" % _ii, w='w_a%d' % _ii)

    # Defining decay between soma and basal dendrite & apical dendrites
    neuron_equ += Equations('I_dendr = gapre*(vmpre-vmself)  : amp',
                                                gapre=1 / (neuron_namespace['Ra'][0]),
                                                I_dendr="Idendr_basal", vmself="vm_basal", vmpre="vm")
    neuron_equ += Equations('I_dendr = gapre*(vmpre-vmself)  + gapost*(vmpost-vmself) : amp',
                                                gapre=1 / (neuron_namespace['Ra'][1]),
                                                gapost=1 / (neuron_namespace['Ra'][0]),
                                                I_dendr="Idendr_soma", vmself="vm",
                                                vmpre="vm_a0", vmpost="vm_basal")

    if dendritic_extent > 0:
        neuron_equ += Equations('I_dendr = gapre*(vmpre-vmself) + gapost*(vmpost-vmself) : amp',
                                                    gapre=1 / (neuron_namespace['Ra'][2]),
                                                    gapost=1 / (neuron_namespace['Ra'][1]),
                                                    I_dendr="Idendr_a0", vmself="vm_a0", vmpre="vm_a1", vmpost="vm")

        # Defining decay between apical dendrite compartments

        for _ii in arange(1, dendritic_extent):
            neuron_equ += Equations('I_dendr = gapre*(vmpre-vmself) + gapost*(vmpost-vmself) : amp',
                                                        gapre=1 / (neuron_namespace['Ra'][_ii]),
                                                        gapost=1 / (neuron_namespace['Ra'][_ii - 1]),
                                                        I_dendr="Idendr_a%d" % _ii, vmself="vm_a%d" % _ii,
                                                        vmpre="vm_a%d" % (_ii + 1), vmpost="vm_a%d" % (_ii - 1))

        neuron_equ += Equations('I_dendr = gapost*(vmpost-vmself) : amp',
                                                    I_dendr="Idendr_a%d" % dendritic_extent,
                                                    gapost=1 / (neuron_namespace['Ra'][dendritic_extent-1]),
                                                    vmself="vm_a%d" % dendritic_extent,
                                                    vmpost="vm_a%d" % (dendritic_extent - 1))

    # If there's only one apical dendrite compartment
    else:
        neuron_equ += Equations('I_dendr = gapre*(vmpre-vmself)  : amp',
                                gapre=1 / (neuron_namespace['Ra'][1]),
                                I_dendr="Idendr_a0", vmself="vm_a0", vmpre="vm")


    group = NeuronGroup(num_neurons, neuron_equ, threshold='vm > ' + repr(Vcut),
                    reset='vm = ' + repr(V_res),
                    refractory=refr_time, method='euler')


    group.vm = EL
    #print group.Idendr_soma


    monitor = SpikeMonitor(group)
    voltmon = StateMonitor(group, 'vm', record=True)

    group.I = 0*nA
    run(500*ms)

    group.I = '0.5*nA * i / num_neurons'
    run(duration)

    # PC neurons have their I_dendr initialized weirdly and thus create spikes in the beginning
    # Thus this solution does not work
    # rheo_idx = min(np.where(monitor.count > 2)[0])
    # print group.I[rheo_idx]

    spikes_after_init = np.where(monitor.t > 500*ms)[0]
    spikers_idx = monitor.i[spikes_after_init]
    rheo_idx = min(spikers_idx)


    if show_plot is True:
        plt.subplots(1,2)
        plt.subplot(1,2,1)
        plot(group.I/nA, monitor.count / duration, '.')
        xlabel('I (nA)')
        ylabel('Firing rate (sp/s)')

        plt.subplot(1,2,2)
        plot(voltmon.t/ms, voltmon.vm[rheo_idx]/mV)

        show()

    return group.I[rheo_idx]

if __name__ == '__main__':
    # data = pd.read_csv('/home/shohokka/Dropbox/~Tutkimus/ManuRevising/CxSystem/gamma_config/gamma_55mtype_anatconf_template.csv', index_col='mtype')
    data = pd.read_csv(
        '/home/shohokka/Dropbox/~Tutkimus/ManuRevising/CxSystem/gamma_config/step2_physio/step2_physioconf_template.csv',
        index_col='step2a_group')
    for k in range(0,7):
        x = data.iloc[k]
        #simulatePC(dendritic_extent, Area_tot_pyram, EL, V_res, VT, fract_areas, show_plot=True)
        rheo = simulatePC(int(x.dendritic_extent), x.Area_tot_pyram*um**2, EL=x.EL*mV, V_res=x.V_res*mV, VT=x.VT*mV,fract_areas=eval(x.fract_areas),show_plot=False)
        print '%s\t\t%.1f*pA' % (x.name, rheo/pA)