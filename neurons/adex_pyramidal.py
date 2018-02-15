from __future__ import division
from brian2 import *
import matplotlib.pyplot as plt


dendritic_extent = 3
#test_current = 150*pA

refr_time = 4*ms
defaultclock_dt = 0.1*ms  # Just for visualization! Changing this doesn't change the clock.
DeltaT = 2*mV

# PC cell parameters
Cm = 1*uF*cm**-2
gl = 4.2e-5*siemens*cm**-2
Area_tot_pyram = 25000 * 0.75 * um**2
VT = -41.61*mV
Vcut = -25*mV
EL = -70.11*mV

# Adaptation parameters
V_res = -55*mV
tau_w = 200*ms
a = 5*nsiemens
b = 0.1*pA

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



# Synaptic parameters; redundant in this tool as there are no synaptic conductances
tau_e = 3*ms  # Depends on neuron type
tau_i = 8*ms  # Depends on neuron type
Ee = 0*mV
Ei = -75*mV
#tau_m = C/gL


def total_PC_area():
    areas = fract_areas[dendritic_extent] * Area_tot_pyram  * 2
    area_total = np.sum(areas)
    return area_total


def compute_rheobase():
    PC_area = total_PC_area()
    gL = PC_area * gl
    C = PC_area*Cm
    # gL = 0.03*Area_tot_pyram*gl
    # C = 0.03*Area_tot_pyram*Cm

    tau_m = C/gL

    print 'PC capacitance: '+str(C)

    bif_type = (a/gL)*(tau_w/tau_m)

    if bif_type < 1:  # saddle-node bifurcation
        print 'SN type neuron'
        rheobase = (gL+a)*(VT - EL - DeltaT + DeltaT*log(1+a/gL))

    elif bif_type > 1:  # Andronov-Hopf bifurcation
        print 'AH type neuron'
        rheobase = (gL+a)*(VT - EL - DeltaT + DeltaT*log(1+tau_m/tau_w)) + DeltaT*gL*((a/gL) - (tau_m/tau_w))

    else:
        print 'Unable to compute rheobase!'
        rheobase = 0*pA

    return rheobase

# rheobase = compute_rheobase()
# print 'Rheobase: '+str(rheobase)

###############################
# EQUATIONS & RUNNING the SIM #
###############################

### BEGIN -- Copy-paste from physiology_reference

# ORIGINAL
# eq_template_soma = '''
# dvm/dt = ((gL*(EL-vm) + gealpha * (Ee-vm) + gialpha * (Ei-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) +I_dendr + I_injected*(1-exp(-t/(50*msecond)))) / C) : volt (unless refractory)
# dge/dt = -ge/tau_e : siemens
# dgealpha/dt = (ge-gealpha)/tau_e : siemens
# dgi/dt = -gi/tau_i : siemens
# dgialpha/dt = (gi-gialpha)/tau_i : siemens
# '''

# WITH ADAPTATION
eq_template_soma = '''
dvm/dt = ((gL*(EL-vm) + gealpha * (Ee-vm) + gialpha * (Ei-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) -w +I_dendr + I) / C) : volt (unless refractory)
dge/dt = -ge/tau_e : siemens
dgealpha/dt = (ge-gealpha)/tau_e : siemens
dgi/dt = -gi/tau_i : siemens
dgialpha/dt = (gi-gialpha)/tau_i : siemens
dw/dt = (-a*(EL-vm)-w)/tau_w : amp
I : amp
'''

#: The template for the dendritic equations used in multi compartmental neurons, the inside values could be replaced later using "Equation" function in brian2.
# eq_template_dend = '''
# dvm/dt = (gL*(EL-vm) + gealpha * (Ee-vm) + gialpha * (Ei-vm) +I_dendr) / C : volt
# dge/dt = -ge/tau_e : siemens
# dgealpha/dt = (ge-gealpha)/tau_e : siemens
# dgi/dt = -gi/tau_i : siemens
# dgialpha/dt = (gi-gialpha)/tau_i : siemens
# '''

# ADAPT. ALSO IN DENDRITE COMPARTMENTS
eq_template_dend = '''
dvm/dt = (gL*(EL-vm) + gealpha * (Ee-vm) + gialpha * (Ei-vm) +I_dendr -w) / C : volt
dge/dt = -ge/tau_e : siemens
dgealpha/dt = (ge-gealpha)/tau_e : siemens
dgi/dt = -gi/tau_i : siemens
dgialpha/dt = (gi-gialpha)/tau_i : siemens
dw/dt = (-a*(EL-vm)-w)/tau_w : amp
'''

neuron_equ = Equations(eq_template_dend, vm="vm_basal", ge="ge_basal",
                                           gealpha="gealpha_basal",
                                           C=neuron_namespace['C'][0],
                                           gL=neuron_namespace['gL'][0],
                                           gi="gi_basal", geX="geX_basal", gialpha="gialpha_basal",
                                           gealphaX="gealphaX_basal", I_dendr="Idendr_basal")

# ORIGINAL
# neuron_equ += Equations(eq_template_soma, gL=neuron_namespace['gL'][1],
#                                             ge='ge_soma', geX='geX_soma', gi='gi_soma', gealpha='gealpha_soma',
#                                             gealphaX='gealphaX_soma',
#                                             gialpha='gialpha_soma', C=neuron_namespace['C'][1],
#                                             I_dendr='Idendr_soma', I_injected=test_current,
#                                             taum_soma=neuron_namespace['taum_soma'])

# ADAPTIVE
neuron_equ += Equations(eq_template_soma, gL=neuron_namespace['gL'][1],
                                            ge='ge_soma', geX='geX_soma', gi='gi_soma', gealpha='gealpha_soma',
                                            gealphaX='gealphaX_soma',
                                            gialpha='gialpha_soma', C=neuron_namespace['C'][1],
                                            I_dendr='Idendr_soma',
                                            taum_soma=neuron_namespace['taum_soma'], w='w_soma')

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

### END -- Copy-paste from physiology_reference

# Main

# ORIGINAL
# G = NeuronGroup(1, neuron_equ, threshold='vm > '+repr(Vcut),reset='vm = '+repr(V_res), refractory=refr_time, method='euler')

# With adaptation
#w_reset = Function(w_reset_func, arg_units=[amp], return_unit=amp)

G = NeuronGroup(1, neuron_equ, threshold='vm > '+repr(Vcut),
                reset='vm = '+repr(V_res)+'; w=w+'+repr(b),
                refractory=refr_time, method='euler')


G.vm = EL


# M = StateMonitor(G, ('vm','ge','gi'), record=True)
# M_spikes = SpikeMonitor(G)
M = StateMonitor(G, ('vm', 'w'), record=True)
M_spikes = SpikeMonitor(G)


# rheobase = compute_rheobase()
# print 'Rheobase: ' + str(rheobase)
rheobase=115*pA
print 'Rheobase: ' + str(rheobase)
test_currents = np.array([0.95, 1.05, 1.2, 1.3, 1.4])
print 'Stimuli (x rheobase): ' + str(test_currents*rheobase)

# Constant current fed here for 1000ms
G.I = 0
run(500*ms)
for curr in test_currents:
    G.I = curr*rheobase
    run(2000 * ms)
    G.I = 0
    run(500*ms)


############
# PLOTTING #
############

plt.subplots(1,3)

plt.subplot(131)
plt.title('$V_m$ with spikes')
plt.plot(M.t/ms, M.vm[0])
plt.plot(M_spikes.t/ms, [0*mV] * len(M_spikes.t), '.')
xlabel('Time (ms)')
ylabel('V_m (V)')
ylim([-0.075, 0.02])

plt.subplot(132)
plt.plot(M.t/ms, M.w[0]/pA)

plt.subplot(133)
plt.plot(M.vm[0]/mV, M.w[0]/pA)
xlabel('V_m (V)')
ylabel('Adap.var. w (pA)')

plt.show()

