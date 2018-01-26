# Test suite for spike frequency adaptation
# According to Fuhrmann et al. J Neurophysiol 2002 vol 87.
#
# Author Henri Hokkanen <henri.hokkanen@helsinki.fi> 14 Nov 2017

from brian2 import *
import matplotlib.pyplot as plt
import efel

efel.api.setThreshold(-69.5)

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
G = NeuronGroup(1, eq_soma, threshold='vm > '+repr(Vcut), reset='vm = '+repr(V_res), refractory=refr_time, method='euler')
G.vm = EL
G.gi = 0
G.ge = 0

M = StateMonitor(G, ('vm', 'ge', 'gi'), record=True)
M_spikes = SpikeMonitor(G)


### STP parameters ###
taurec1 = 450*ms
U = 0.25

taurec2 = 130*ms
taufacil = 670*ms
U1 = 0.09

w = 0.80*nS

### SYNAPSE Depr. ###
# H = PoissonGroup(1, 50*Hz)
base_firing_rate = 10 #*Hz
N_synapses = 5
spike_times = np.linspace(1, 2, base_firing_rate)*second
spikegroup_indices = list(np.zeros(base_firing_rate))
spikegroup_times = list(spike_times)
for i in arange(1,N_synapses):
    spikegroup_indices.extend(np.ones(base_firing_rate)*i)
    spikegroup_times.extend(list(spike_times))

# H = SpikeGeneratorGroup(1, np.zeros(base_firing_rate),spike_times)
H = SpikeGeneratorGroup(N_synapses, spikegroup_indices, spikegroup_times)


# Clock-driven depressing
# R - resources ie. presynaptic vesicles
# Whether to have ge_post += U*R*w or R*w is a question of taste (what "w" represents; release of U*all_vesicles or all_vesicles)
# S = Synapses(H, G,
#              model='dR/dt = (1-R)/taurec1 : 1 (clock-driven)',
#              on_pre=''' ge_post += R * w
#                         R = (1-U)*R ''')

# Facilitating clock-driven
S = Synapses(H, G,
             model=''' dR/dt = (1-R)/taurec2 : 1 (clock-driven)
                       du/dt = (U1-u)/taufacil : 1 (clock-driven) ''',
             on_pre=''' ge_post += R * u * w
                        R = (1-u)*R
                        u = u + U1*(1-u)''')

S.connect()
S.R = 1

########################
SM = StateMonitor(S, ('R'), record=True)


# SM2 = StateMonitor(S2, ('R'), record=True)
run(3000 * ms)
vm_lim = [-75, -55]

# traces = []
# trace = {}
# trace['T'] = M.t/ms
# trace['V'] = M.vm[0]/mV
# trace['stim_start'] = [1000]
# trace['stim_end'] = [2000]
# traces.append(trace)
# stuff = efel.getMeanFeatureValues(traces, ['peak_indices'])
#
# print stuff


plt.subplots(1,3)
### Membrane voltage plot
plt.subplot(1,3,1)
plt.title('$V_m$ (depressing)')
plt.plot(M.t/ms, M.vm[0]/mV, label='CD')
baseline_times = spike_times
spike_times = spike_times + 7*ms
peak_volt_indices = (spike_times/ms)/0.1
peak_volts = [M.vm[0][int(i)] for i in peak_volt_indices]
baseline_volt_indices = (baseline_times/ms)/0.1
baseline_volts = [M.vm[0][int(i)] for i in baseline_volt_indices]

plt.plot(spike_times/ms, peak_volts/mV, '.')
xlabel('Time (ms)')
ylabel('V_m (V)')
plt.legend()
plt.ylim(vm_lim)

###
plt.subplot(1,3,2)
plt.title('R (depressing)')
plt.plot(SM.t/ms, SM.R[0], label='CD')
xlabel('Time (ms)')
ylabel('Frac. resources (1)')
plt.ylim([0, 1])
plt.legend()

###
plt.subplot(1,3,3)
plt.title('EPSP height')
#amplitudes = [pv+70*mV for pv in peak_volts]
amplitudes = [peak_volts[i]-baseline_volts[i] for i in range(len(spike_times))]
relative_peak_heights = [amp/amplitudes[0] for amp in amplitudes]
peak_i = np.arange(1, len(relative_peak_heights)+1)
plt.plot(peak_i, relative_peak_heights, '.')
xlabel('Spike #')
ylabel('Relative amplitude')
xlim(1,10)
ylim(0,15)
plt.show()