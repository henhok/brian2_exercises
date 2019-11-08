from brian2 import *

num_neurons = 500
duration = 1*second

refr_time = 4*ms


# Parameters
passive_params = {'C': 135 * pF, 'gL': 5.5 * nS, 'EL': -73 * mV,
                  'VT': -42 * mV, 'DeltaT': 4 * mV,
                  'Vcut': 20 * mV, 'refr_time': 4 * ms}

C = 135*pF
gL = 5.5*nS
tau_m = C/gL
EL = -73*mV
Vcut = 20*mV
DeltaT = 2.9*mV
VT = -61.8*mV

a = 1.1*nS
tau_w = 261*ms
b = 101*pA
V_res = -63*mV

tau_e = 3*ms  # Depends on neuron type
tau_i = 8*ms  # Depends on neuron type
Ee = 0*mV
Ei = -75*mV
gemean = 13*nS
gestd = 2*nS
gimean = 20*nS
gistd = 6*nS

eqs = '''
dvm/dt = (gL*(EL-vm) + ge*(Ee-vm) + gi*(Ei-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) -w + I) / C : volt (unless refractory)
dge/dt = -(ge-gemean)/tau_e + sqrt((2*gestd**2)/tau_e)*xi_1: siemens
dgi/dt = -(gi-gimean)/tau_i + sqrt((2*gistd**2)/tau_i)*xi_2 : siemens
dw/dt = (-a*(EL-vm)-w)/tau_w : amp
I : amp
'''
G = NeuronGroup(1, eqs, threshold='vm > ' + repr(Vcut),
                reset='vm = ' + repr(V_res) + '; w=w+' + repr(b),
                refractory=refr_time, method='euler')

G.w = 0*pA
G.vm = EL

M = StateMonitor(G, ('vm', 'w'), record=True)
M_spikes = SpikeMonitor(G)


run(2000*ms)


############
# PLOTTING #
############

plt.subplots(1,3)

plt.subplot(131)
plt.title('$V_m$ with spikes')
plt.plot(M.t/ms, M.vm[0], c='black')
# plt.plot(M_spikes.t/ms, [0*mV] * len(M_spikes.t), '.')
xlabel('Time (ms)')
ylabel('V_m (V)')
xlim([1000, 2000])
ylim([-0.075, 0.02])

plt.subplot(132)
plt.plot(M.t/ms, M.w[0]/pA)

plt.subplot(133)
plt.plot(M.vm[0]/mV, M.w[0]/pA)
xlabel('V_m (V)')
ylabel('Adap.var. w (pA)')

plt.show()
