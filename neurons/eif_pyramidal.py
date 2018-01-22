from brian2 import *
import matplotlib.pyplot as plt


dendritic_extent = 1
test_current = 300*pA

refr_time = 4*ms
defaultclock_dt = 0.1*ms  # Just for visualization! Changing this doesn't change the clock.
DeltaT = 2*mV

# PC cell parameters
Cm = 1*uF*cm**-2
gl = 4.2e-5*siemens*cm**-2
Area_tot_pyram = 25000 * 0.75 * um**2
VT = -41.61*mV
Vcut = -25*mV
V_res = -55*mV
EL = -70.11*mV

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


# OBSOLETE - Code for simplified model
# fract_area_fixed = fract_areas[dendritic_extent]
#
# area_basal = 0.2 * area_total
# area_apical = 0*um**2
# for i in range(0, dendritic_extent+1):
#     area_apical += fract_area_fixed[2+i] * area_total
#
# R_basal = Ra[0]
# R_apical = Ra[-1]  # last compartment always with Ra[-1] resistance
# for i in range(0, dendritic_extent):
#     R_apical += Ra[i+1]
#
# gL_basal = gl*area_basal
# gL_apical = gl*area_apical
# C_basal = Cm*area_basal*2  # x2 to account for spine area
# C_apical = Cm*area_apical*2  # x2 to account for spine area

# Synaptic parameters; redundant in this tool as there are no synaptic conductances
tau_e = 3*ms  # Depends on neuron type
tau_i = 8*ms  # Depends on neuron type
Ee = 0*mV
Ei = -75*mV
#tau_m = C/gL


###############################
# EQUATIONS & RUNNING the SIM #
###############################

### BEGIN -- Copy-paste from physiology_reference

eq_template_soma = '''
dvm/dt = ((gL*(EL-vm) + gealpha * (Ee-vm) + gialpha * (Ei-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) +I_dendr + I_injected*(1-exp(-t/(50*msecond)))) / C) : volt (unless refractory)
dge/dt = -ge/tau_e : siemens
dgealpha/dt = (ge-gealpha)/tau_e : siemens
dgi/dt = -gi/tau_i : siemens
dgialpha/dt = (gi-gialpha)/tau_i : siemens
'''
#: The template for the dendritic equations used in multi compartmental neurons, the inside values could be replaced later using "Equation" function in brian2.
eq_template_dend = '''
dvm/dt = (gL*(EL-vm) + gealpha * (Ee-vm) + gialpha * (Ei-vm) +I_dendr) / C : volt
dge/dt = -ge/tau_e : siemens
dgealpha/dt = (ge-gealpha)/tau_e : siemens
dgi/dt = -gi/tau_i : siemens
dgialpha/dt = (gi-gialpha)/tau_i : siemens
'''


neuron_equ = Equations(eq_template_dend, vm="vm_basal", ge="ge_basal",
                                           gealpha="gealpha_basal",
                                           C=neuron_namespace['C'][0],
                                           gL=neuron_namespace['gL'][0],
                                           gi="gi_basal", geX="geX_basal", gialpha="gialpha_basal",
                                           gealphaX="gealphaX_basal", I_dendr="Idendr_basal")
neuron_equ += Equations(eq_template_soma, gL=neuron_namespace['gL'][1],
                                            ge='ge_soma', geX='geX_soma', gi='gi_soma', gealpha='gealpha_soma',
                                            gealphaX='gealphaX_soma',
                                            gialpha='gialpha_soma', C=neuron_namespace['C'][1],
                                            I_dendr='Idendr_soma', I_injected=test_current,
                                            taum_soma=neuron_namespace['taum_soma'])
for _ii in range(dendritic_extent + 1):  # extra dendritic compartment in the same level of soma
    neuron_equ += Equations(eq_template_dend, vm="vm_a%d" % _ii,
                                                C=neuron_namespace['C'][_ii],
                                                gL=neuron_namespace['gL'][_ii],
                                                ge="ge_a%d" % _ii,
                                                gi="gi_a%d" % _ii, geX="geX_a%d" % _ii,
                                                gealpha="gealpha_a%d" % _ii, gialpha="gialpha_a%d" % _ii,
                                                gealphaX="gealphaX_a%d" % _ii, I_dendr="Idendr_a%d" % _ii)

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

### END -- Copy-paste from physiology_reference

# Main
G = NeuronGroup(1, neuron_equ, threshold='vm > '+repr(Vcut), reset='vm = '+repr(V_res), refractory=refr_time, method='euler')
G.vm = EL


# M = StateMonitor(G, ('vm','ge','gi'), record=True)
# M_spikes = SpikeMonitor(G)
M = StateMonitor(G, ('vm'), record=True)
M_spikes = SpikeMonitor(G)


# Constant current fed here for 1000ms
# run(20 * ms)
# G.I = test_current
# run(1000 * ms)
# G.I = 0*nA
# run(50 * ms)

run(1000*ms)


############
# PLOTTING #
############

plt.figure()
plt.title('$V_m$ with spikes')
plt.plot(M.t/ms, M.vm[0])
plt.plot(M_spikes.t/ms, [0*mV] * len(M_spikes.t), '.')
xlabel('Time (ms)')
ylabel('V_m (V)')
ylim([-0.075, 0.02])
plt.show()

