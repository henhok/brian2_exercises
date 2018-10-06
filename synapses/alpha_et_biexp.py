from brian2 import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

test_current = 72*pA

refr_time = 4*ms
defaultclock.dt = 0.1*ms
DeltaT = 2*mV

# SS cell (alternative; params wanted within physiological range)
C = 110*pF
gL = 3.1*nS
VT = -45*mV
Vcut = 20*mV
V_res = -70*mV
EL = -70*mV

# Synaptic parameters; redundant in this tool as there are no synaptic conductances
tau_e_rise = 0.2*ms
tau_e_decay = 1.7*ms
tau_e = 1.7*ms
tau_e_alpha = 1.5*ms
Ee = 0*mV
Ei = -75*mV
tau_m = C/gL
K_alpha = exp(1)

syn_weight = 1*nS

t_peak = (tau_e_decay*tau_e_rise)/(tau_e_decay - tau_e_rise)*log(tau_e_decay/tau_e_rise)
# K = 1/exp(-t_peak/tau_e_decay)

print t_peak

I = 50*pA

### ALPHA BEGINS
# Synaptic parameters; redundant in this tool as there are no synaptic conductances
eqs_eif_alpha = '''
dvm/dt = (gL*(EL-vm) + gealpha*(Ee-vm) + gL*DeltaT*exp((vm-VT) / DeltaT) + I) / C : volt (unless refractory)
dge/dt = -ge/tau_e_alpha : siemens
dgealpha1/dt = (ge-gealpha1)/tau_e_alpha : siemens
gealpha = K_alpha * gealpha1 : siemens
'''
### ALPHA ENDS

### EXPDEC BEGINS
eqs_eif_expdec = '''
dvm/dt = (gL*(EL-vm) + ge*(Ee-vm) + gL*DeltaT*exp((vm-VT) / DeltaT) + I) / C : volt (unless refractory)
dge/dt = -ge/tau_e : siemens
'''

### EXPDEC ENDS
eqs_eif_biexp = '''
dvm/dt = (gL*(EL-vm) + gealpha*(Ee-vm) + gL*DeltaT*exp((vm-VT) / DeltaT) + I) / C : volt (unless refractory)
dge/dt = -ge/tau_e_decay : siemens
dgealpha1/dt = (ge-gealpha1)/tau_e_rise : siemens
gealpha = (tau_e_decay/tau_e_rise)**(tau_e_rise/(tau_e_decay-tau_e_rise)) * gealpha1 : siemens
'''

G = NeuronGroup(1, eqs_eif_biexp, threshold='vm > '+repr(Vcut), reset='vm = '+repr(V_res), refractory=refr_time, method='euler')
G_alpha = NeuronGroup(1, eqs_eif_alpha, threshold='vm > '+repr(Vcut), reset='vm = '+repr(V_res), refractory=refr_time, method='euler')
G_expdec = NeuronGroup(1, eqs_eif_expdec, threshold='vm > '+repr(Vcut), reset='vm = '+repr(V_res), refractory=refr_time, method='euler')

# H = PoissonGroup(1, 1*Hz)
H = SpikeGeneratorGroup(1, [0], [500*ms])
S = Synapses(H, G, on_pre='ge_post += '+repr(syn_weight))
S.connect(i=0, j=0)

S_alpha = Synapses(H, G_alpha, on_pre='ge_post += '+repr(syn_weight))
S_alpha.connect(i=0, j=0)

S_expdec = Synapses(H, G_expdec, on_pre='ge_post += '+repr(syn_weight))
S_expdec.connect(i=0, j=0)

statemon = StateMonitor(G, ('vm', 'gealpha'), record=True)
#spikemon = SpikeMonitor(H)

statemon_alpha = StateMonitor(G_alpha, ('vm', 'gealpha'), record=True)
#spikemon_alpha = SpikeMonitor(G_alpha)

statemon_expdec = StateMonitor(G_expdec, ('vm', 'ge'), record=True)

run(1000*ms)

plt.subplots(2,1)
plt.suptitle('Postsynaptic response models, dt 0.01ms')

plt.subplot(211)
ax1 = plt.gca()
plt.title('Membrane potential')
plt.plot(statemon.t/ms, statemon.vm[0]/mV, label='Bi-exponential')
plt.plot(statemon_alpha.t/ms, statemon_alpha.vm[0]/mV, label='PC GABA-A')
plt.plot(statemon_expdec.t/ms, statemon_expdec.vm[0]/mV, label='Non-PC GABA-A')
#plt.plot(spikemon.t/ms, -30*np.ones(len(spikemon.t)), '.')
plt.legend()

plt.subplot(212, sharex=ax1)
plt.title('Conductance')
plt.plot(statemon.t/ms, statemon.gealpha[0]/nS, label='Bi-exponential')
plt.plot(statemon_alpha.t/ms, statemon_alpha.gealpha[0]/nS, label='PC GABA-A')
plt.plot(statemon_expdec.t/ms, statemon_expdec.ge[0]/nS, label='Non-PC GABA-A')
plt.legend()
#plt.plot(spikemon.t/ms, 0*np.ones(len(spikemon.t)), '.')
#plt.plot(t_peak/ms + spikemon.t/ms, 5*np.ones(len(spikemon.t)), '.', color='red')

# plt.subplot(313, sharex=ax1)
# plt.title('ge')
# plt.plot(statemon.t/ms, statemon.ge[0]/nS)
# plt.plot(spikemon.t/ms, 10*np.ones(len(spikemon.t)), '.')

plt.show()
