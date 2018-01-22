from brian2 import *

num_neurons = 500
duration = 2*second

refr_time = 4*ms


# Parameters
dendritic_extent = 4
Cm = 1*uF*cm**-2
gl = 4.2e-5*siemens*cm**-2
Area_tot_pyram = 25000 * 0.75 * um**2
Vcut = 20*mV
EL = -73*mV

# Adaptation parameters
a = 0.469*nsiemens
tau_w = 69.9*ms
b = 315*pA
V_res = -51.2*mV
VT = -56*mV
DeltaT = 4*mV

# Dendritic parameters
neuron_namespace = dict()

fract_areas = {1: array([0.2,  0.03,  0.15,  0.2]),
                2: array([0.2,  0.03,  0.15,  0.15,  0.2]),
                3: array([0.2,  0.03,  0.15,  0.09,  0.15,  0.2]),
                4: array([0.2,  0.03,  0.15,  0.15,  0.09,  0.15,  0.2])}

neuron_namespace['Ra'] = [100, 80, 150, 150, 200] * Mohm

neuron_namespace['C'] = fract_areas[dendritic_extent] * \
                             Cm * Area_tot_pyram * 2
neuron_namespace['gL'] = fract_areas[dendritic_extent] * \
                              gl * Area_tot_pyram
neuron_namespace['taum_soma'] = neuron_namespace['C'][1] / neuron_namespace['gL'][1]

# WITH ADAPTATION
eq_template_soma = '''
dvm/dt = ((gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) -w +I_dendr + I) / C) : volt (unless refractory)
dw/dt = (-a*(EL-vm)-w)/tau_w : amp
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
                                            gapost=1 / (neuron_namespace['Ra'][-1]),
                                            vmself="vm_a%d" % dendritic_extent,
                                            vmpost="vm_a%d" % (dendritic_extent - 1))


group = NeuronGroup(num_neurons, neuron_equ, threshold='vm > ' + repr(Vcut),
                reset='vm = ' + repr(V_res) + '; w=w+' + repr(b),
                refractory=refr_time, method='euler')


group.vm = EL
group.I = '0.5*nA * i / num_neurons'

monitor = SpikeMonitor(group)

run(duration)

rheo_idx = min(np.where(monitor.count > 0)[0])
print group.I[rheo_idx]

plot(group.I/nA, monitor.count / duration)
xlabel('I (nA)')
ylabel('Firing rate (sp/s)')
show()