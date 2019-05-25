# Test suite for spike frequency adaptation
# According to Fuhrmann et al. J Neurophysiol 2002 vol 87.
#
# Author Henri Hokkanen <henri.hokkanen@helsinki.fi> 14 Nov 2017

from brian2 import *
import matplotlib.pyplot as plt


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

# SS cell (variant no2)
# C = 35*pF
# gL = 1.0*nS
# VT = -45*mV
# Vcut = -25*mV
# V_res = -70*mV
# EL = -70*mV

# SS cell (variant no3; params wanted within physiological range)
C = 110*pF
gL = 3.1*nS
VT = -45*mV
Vcut = -25*mV
V_res = -70*mV
EL = -70*mV

# Synaptic parameters
tau_e = 3*ms  # Depends on neuron type
tau_i = 8*ms  # Depends on neuron type
Ee = 0*mV
Ei = -75*mV
tau_m = C/gL

noise_sigma = 0*mV
tonic_current = 0.0*72*pA

eq_soma = '''
 dvm/dt = ((gL*(EL-vm) + ge*(Ee-vm) + gi*(Ei-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) + tonic_current) / C) + noise_sigma*xi*tau_m**-0.5: volt
 dge/dt = -ge/tau_e : siemens
 dgi/dt = -gi/tau_i : siemens
 '''



# Main
G = NeuronGroup(4, eq_soma, threshold='vm > '+repr(Vcut), reset='vm = '+repr(V_res), refractory=refr_time, method='euler')
G.vm = EL
G.gi = 0
G.ge = 0

M = StateMonitor(G, ('vm', 'ge', 'gi'), record=True)
M_spikes = SpikeMonitor(G)


### STP parameters ###
taurec1 = 450*ms
taurec2 = 450*ms
taufacil = 100*ms
U = 0.5
U1 = 0.02
w = 0.80*nS

### SYNAPSE 1 Depr. ###
H = PoissonGroup(1, 50*Hz)

# Clock-driven depressing
# R - resources ie. presynaptic vesicles
# Whether to have ge_post += U*R*w or R*w is a question of taste (what "w" represents; release of U*all_vesicles or all_vesicles)
S = Synapses(H, G,
             model='dR/dt = (1-R)/taurec1 : 1 (clock-driven)',  # <-- UPDATE 5/2019 // works with (event-driven) from Brian2 2.1.3(?) onwards
             on_pre=''' ge_post += R * w
                        R = (1-U)*R ''')

# Event-driven depressing
# (because for some reason Brian2 refuses to solve the eq above -- UPDATE 5/2019: diff eq works as event-driven, no need for lastupdate hassle)
# Think: R_new = whatever resources we had + what was recovered after previous spike
S_alt = Synapses(H, G,
             model='''  R : 1
                        lastupdate : second''',   # Brian 2.1.3 needs this line
             on_pre=''' R = R + (1-R)*(1 - exp(-(t-lastupdate)/taurec2))
                        ge_post += R * w
                        R = (1-U)*R
                        lastupdate = t''')  # Brian 2.1.3 needs this line




S.connect(i=0, j=0)
S_alt.connect(i=0, j=1)

S.R = 1
S_alt.R = 1

### SYNAPSE 2 Facil. ###
#I = PoissonGroup(1, 23*Hz)

# Facilitating clock-driven
S2 = Synapses(H, G,
             model=''' dR/dt = (1-R)/taurec2 : 1 (clock-driven)
                       du/dt = (U1-u)/taufacil : 1 (clock-driven) ''',
             on_pre=''' ge_post += R * w
                        R = (1-u)*R
                        u = u + U1*(1-u)''')

# Facilitating event-driven
S2_alt = Synapses(H, G,
             model=''' R : 1
                       u : 1 
                       lastupdate : second''',
             on_pre=''' R = R + (1-R)*(1 - exp(-(t-lastupdate)/taurec2))
                        u = u + (U1-u)*(1 - exp(-(t-lastupdate)/taufacil))
                        ge_post += R * w
                        R = (1-u)*R
                        u = u + U1*(1-u)
                        lastupdate = t''')
#u = u * exp(-(t-lastupdate)/taufacil)


S2.connect(i=0, j=2)
S2_alt.connect(i=0, j=3)

S2.u = U1
S2.R = 1
S2_alt.u = U1
S2_alt.R = 1

########################
SM = StateMonitor(S, ('R'), record=True)
S_altM = StateMonitor(S_alt, ('R'), record=True)

S2M = StateMonitor(S2, ('R', 'u'), record=True)
S2_altM = StateMonitor(S2_alt, ('R', 'u'), record=True)

# SM2 = StateMonitor(S2, ('R'), record=True)
run(2000 * ms)


plt.subplots(2,2)
plt.suptitle('Clock- vs. event-driven synapses with STP')

vm_lim = [-70.1, -68.5]

### Membrane voltage plot
plt.subplot(2,2,1)
plt.title('$V_m$ (depressing)')
plt.plot(M.t/ms, M.vm[0]/mV, label='CD')
plt.plot(M.t/ms, M.vm[1]/mV, label='ED')
# plt.plot(M_spikes.t/ms, [0*mV] * len(M_spikes.t), '.')
xlabel('Time (ms)')
ylabel('V_m (V)')
plt.legend()
plt.ylim(vm_lim)

###
plt.subplot(2,2,2)
plt.title('R (depressing)')
plt.plot(SM.t/ms, SM.R[0], label='CD')
plt.plot(S_altM.t/ms, S_altM.R[0], label='ED')
xlabel('Time (ms)')
ylabel('Frac. resources (1)')
plt.ylim([0, 1])
plt.legend()

### Membrane voltage plot
plt.subplot(2,2,3)
plt.title('$V_m$ (facil)')
plt.plot(M.t/ms, M.vm[2]/mV, label='CD')
plt.plot(M.t/ms, M.vm[3]/mV, label='ED')
# plt.plot(M.t/ms, M.vm[1]/mV, label='Neuron 2')
# plt.plot(M_spikes.t/ms, [0*mV] * len(M_spikes.t), '.')
xlabel('Time (ms)')
ylabel('V_m (V)')
plt.legend()
plt.ylim(vm_lim)

###
plt.subplot(2,2,4)
plt.title('R, u (facil)')
plt.plot(S2M.t/ms, S2M.R[0], label='R, CD')
plt.plot(S2M.t/ms, S2M.u[0], label='u, CD')
plt.plot(S2_altM.t/ms, S2_altM.R[0], label='R, ED')
plt.plot(S2_altM.t/ms, S2_altM.u[0], label='u, ED')
# plt.plot(S_altM.t/ms, S_altM.R[0], label='Event-driven')
xlabel('Time (ms)')
ylabel('R/u (1)')
plt.ylim([0, 1])
plt.legend()



plt.show()