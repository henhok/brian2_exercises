from brian2 import *
import matplotlib.pyplot as plt

# set_device('cpp_standalone', directory='cpptest')

tonic_current = 0.0*72*pA


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

# Synaptic parameters; redundant in this tool as there are no synaptic conductances
tau_e = 3*ms  # Depends on neuron type
tau_i = 8*ms  # Depends on neuron type
Ee = 0*mV
Ei = -75*mV
tau_m = C/gL


# Noise parameters
gemean = 0*nS
gestd = 0*nS
gimean = 0*nS
gistd = 0*nS
noise_sigma = 8*mV


# Stochastic equation with fluctuating synaptic conductances (set ge/gi mean/std to zero if you don't want stochasticity)
eq_soma = '''
 dvm/dt = ((gL*(EL-vm) + ge*(Ee-vm) + gi*(Ei-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) + tonic_current) / C) + noise_sigma*xi_3*tau_m**-0.5: volt
 dge/dt = -(ge-gemean)/tau_e + sqrt((2*gestd**2)/tau_e)*xi_1: siemens
 #dgealpha/dt = (ge-gealpha)/tau_e : siemens
 dgi/dt = -(gi-gimean)/tau_i + sqrt((2*gistd**2)/tau_i)*xi_2 : siemens
 #dgialpha/dt = (gi-gialpha)/tau_i : siemens
 I: amp
 '''

# eq_PC_soma = '''
#  dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) + I + (1/R_apical)*(v_apical-vm) + (1/R_basal)*(v_basal-vm) ) / C : volt
#  dv_apical/dt = (gL_apical*(EL-v_apical) + (1/R_apical)*(vm-v_apical))/C_apical : volt
#  dv_basal/dt = (gL_basal*(EL-v_basal) + (1/R_apical)*(vm-v_basal))/C_basal : volt
#  I: amp
#  '''
# G.v_apical = EL
# G.v_basal = EL

# Main
G = NeuronGroup(2,eq_soma, threshold='vm > '+repr(Vcut), reset='vm = '+repr(V_res), refractory=refr_time, method='euler')
G.vm = EL
G.gi = gimean
G.ge = gemean

M = StateMonitor(G, ('vm', 'ge', 'gi'), record=True)
M_spikes = SpikeMonitor(G)


### Poisson-noise
# H = PoissonGroup(1, 6*Hz)
# S = Synapses(H,G,on_pre='ge_post += 0.125*nS')
# S.connect(i=0, j=0)
#
# I = PoissonGroup(1,13*Hz)
# S2 = Synapses(I,G,on_pre='gi_post = 0.625*nS')
# S2.connect(i=0,j=0)

### Alternative Poisson
# bg_rate = 4*Hz
# exc_weight = 0.3*nS
# inh_scaling = 4
# Pe = PoissonInput(target=G, target_var='ge', N=3600, rate=bg_rate, weight=exc_weight)
# Pi = PoissonInput(G, 'gi', 500, bg_rate, inh_scaling*exc_weight)

### Timed spikes
# times = array([100, 200])*ms
# indices = array([0]*len(times))
# H = SpikeGeneratorGroup(1, indices, times)
# S = Synapses(H,G,on_pre='ge_post += 6*nS')
# S.connect(i=0, j=0)
# run(1000*ms)

run(10000 * ms)


plt.subplots(1,3)
# plt.suptitle('Bg rate: '+repr(bg_rate))
print std(M.vm[0])

### Membrane voltage plot
plt.subplot(1,3,1)
plt.title('$V_m$ with spikes')
plt.plot(M.t/ms, M.vm[0])
plt.plot(M.t/ms, M.vm[1])
plt.plot(M_spikes.t/ms, [0*mV] * len(M_spikes.t), '.')
xlabel('Time (ms)')
ylabel('V_m (V)')
ylim([-0.075, 0.02])

### Conductance plot
plt.subplot(1,3,2)
plt.title('Conductance')
plt.plot(M.t/ms, M.ge[0], label='ge', c='g')
plt.plot(M.t/ms, M.gi[0], label='gi', c='r')
xlabel('Time (ms)')
ylabel('Conductance (S)')
#ylim([0, 50e-9])
plt.legend()

# ### ge/gi plot with AP threshold line
plt.subplot(1,3,3)
plt.title('Excitatory vs. inhibitory conductance')

def gi_line(x): return (-x*(Ee-VT) - gL*(EL-VT) - tonic_current)/(Ei-VT)

x_values = np.arange(0*nS,20*nS,1*nS)
plt.plot(x_values, [gi_line(x) for x in x_values], label='$dV_m/dt = 0$')

for spike_time in M_spikes.t/defaultclock_dt:
    plt.plot(M.ge[0][spike_time], M.gi[0][spike_time], 'g.')

plt.plot(M.ge[0], M.gi[0], 'y.', alpha=0.02)
plt.axis('equal')
plt.xlabel('ge (S)')
plt.ylabel('gi (S)')
plt.legend()

plt.show()