from brian2 import *
import matplotlib.pyplot as plt


test_current = 72*pA

refr_time = 4*ms
defaultclock_dt = 0.1*ms  # Just for visualization! Changing this doesn't change the clock.
DeltaT = 2*mV

######################################################
#  NEURON TYPES -- uncomment appropriate parameters  #
######################################################

# PC cell
# PC_flag = True
# Cm = 1*uF*cm**-2
# gl = 4.2e-5*siemens*cm**-2
# area_total = 25000 * 0.75 * um**2
# C = 5.625*pF  # soma
# gL = 0.24*nS  # soma
# VT = -41.61*mV
# Vcut = -25*mV
# V_res = -55*mV
# EL = -70.11*mV
#
# # Dendritic parameters (simplified 3 compartment model)
# dendritic_extent = 1
# fract_areas = {1: array([0.2,  0.03,  0.15,  0.2]),
#                 2: array([0.2,  0.03,  0.15,  0.15,  0.2]),
#                 3: array([0.2,  0.03,  0.15,  0.09,  0.15,  0.2]),
#                 4: array([0.2,  0.03,  0.15,  0.15,  0.09,  0.15,  0.2])}
#
# Ra = [100, 80, 150, 150, 200] * Mohm
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


# BC cell
# C = 100*pF
# gL = 10*nS
# VT = -38.8*mV
# Vcut = VT + 5*DeltaT
# V_res = VT - 4*mV
# EL = -67.66*mV

# L1i cell
# C = 63.36*pF
# gL =3.2*nS
# VT = -36.8*mV
# Vcut = VT + 5*DeltaT
# V_res = VT - 4*mV
# EL = -67.66*mV

# MC cell
# C = 92.1*pF
# gL = 4.2*nS
# VT = -42.29*mV
# Vcut = VT + 5*DeltaT
# V_res = VT - 4*mV
# EL = -60.38*mV

# SS cell
# C = 35*pF
# gL = 1.0*nS
# VT = -45*mV
# Vcut = -25*mV
# V_res = -70*mV
# EL = -70*mV

# SS cell (alternative; params wanted within physiological range)
C = 110*pF
gL = 3.1*nS
VT = -45*mV
Vcut = -25*mV
V_res = -70*mV
EL = -70*mV

# Synaptic parameters; redundant in this tool as there are no synaptic conductances
tau_e = 3*ms  # Depends on neuron type
tau_i = 8*ms  # Depends on neuron type
Ee = 0*mV
Ei = -75*mV
tau_m = C/gL


###############################
# EQUATIONS & RUNNING the SIM #
###############################

if 'PC_flag' not in locals():
    eq_soma = '''
     dvm/dt = ((gL*(EL-vm) + ge * (Ee-vm) + gi * (Ei-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) +I) / C) : volt
     dge/dt = -ge/tau_e : siemens
     dgi/dt = -gi/tau_i : siemens
     I: amp
     '''
else:
    eq_soma = '''
     dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) + I + (1/R_apical)*(v_apical-vm) + (1/R_basal)*(v_basal-vm) ) / C : volt
     dv_apical/dt = (gL_apical*(EL-v_apical) + (1/R_apical)*(vm-v_apical))/C_apical : volt
     dv_basal/dt = (gL_basal*(EL-v_basal) + (1/R_apical)*(vm-v_basal))/C_basal : volt
     I: amp
     '''

# Main
G = NeuronGroup(1,eq_soma, threshold='vm > '+repr(Vcut), reset = 'vm = '+repr(V_res), refractory = refr_time, method='euler')
G.vm = EL
# G.v_apical = EL
# G.v_basal = EL

# M = StateMonitor(G, ('vm','ge','gi'), record=True)
# M_spikes = SpikeMonitor(G)
M = StateMonitor(G, ('vm'), record=True)
M_spikes = SpikeMonitor(G)


# Constant current fed here for 1000ms
run(20 * ms)
G.I = test_current
run(1000 * ms)
G.I = 0*nA
run(50 * ms)


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

