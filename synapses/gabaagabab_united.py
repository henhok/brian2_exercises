from brian2 import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')



refr_time = 4*ms
defaultclock.dt = 0.1*ms  # Note that for small rise times conductance overshoots unless dt is small
DeltaT = 2*mV

# SS cell (alternative; params wanted within physiological range)
C = 110*pF
gL = 3.1*nS
VT = -45*mV
Vcut = 20*mV
V_res = -70*mV
EL = -70*mV
rheobase = 72*pA

# Synaptic parameters; redundant in this tool as there are no synaptic conductances
tau_e_rise = 0.1*ms
tau_e_decay = 8.3*ms
tau_nmda_rise = 3.0*ms
tau_nmda_decay = 260*ms
Ee = 0*mV
Ei = -75*mV
Ee = Ei
tau_m = C/gL

nmda_scaling = 0.2
K_alpha = 1
tau_e_alpha = 8.3*ms
tau_e = 8.3*ms

t_peak = (tau_e_decay*tau_e_rise)/(tau_e_decay - tau_e_rise)*log(tau_e_decay/tau_e_rise)
# K = 1/exp(-t_peak/tau_e_decay)

print t_peak

eqs_eif_united = '''
dvm/dt = (gL*(EL-vm) + gealpha*(Ee-vm) + gnmda_alpha*(Ee-vm) + gL*DeltaT*exp((vm-VT) / DeltaT) + I) / C : volt (unless refractory)
dge/dt = -ge/tau_e_decay : siemens
dgealpha1/dt = (ge-gealpha1)/tau_e_rise : siemens
gealpha = (tau_e_decay/tau_e_rise)**(tau_e_rise/(tau_e_decay-tau_e_rise)) * gealpha1 : siemens
dgnmda/dt = -gnmda/tau_nmda_decay : siemens
dgnmda_alpha1/dt = (gnmda-gnmda_alpha1)/tau_nmda_rise : siemens
gnmda_alpha = nmda_scaling * gnmda_alpha1 * (tau_nmda_decay/tau_nmda_rise)**(tau_nmda_rise/(tau_nmda_decay-tau_nmda_rise)) : siemens
I : amp
'''

### ALPHA BEGINS
# Synaptic parameters; redundant in this tool as there are no synaptic conductances
eqs_eif_alpha = '''
dvm/dt = (gL*(EL-vm) + gealpha*(Ee-vm) + gL*DeltaT*exp((vm-VT) / DeltaT) + I) / C : volt (unless refractory)
dge/dt = -ge/tau_e_alpha : siemens
dgealpha1/dt = (ge-gealpha1)/tau_e_alpha : siemens
gealpha = K_alpha * gealpha1 : siemens
I: amp
'''
### ALPHA ENDS

### EXPDEC BEGINS
eqs_eif_expdec = '''
dvm/dt = (gL*(EL-vm) + ge*(Ee-vm) + gL*DeltaT*exp((vm-VT) / DeltaT) + I) / C : volt (unless refractory)
dge/dt = -ge/tau_e : siemens
I : amp
'''
### EXPDEC ENDS


G = NeuronGroup(1, eqs_eif_united, threshold='vm > ' + repr(Vcut), reset='vm = ' + repr(V_res), refractory=refr_time, method='euler')
G_alpha = NeuronGroup(1, eqs_eif_alpha, threshold='vm > ' + repr(Vcut), reset='vm = ' + repr(V_res), refractory=refr_time, method='euler')
G_expdec = NeuronGroup(1, eqs_eif_expdec, threshold='vm > ' + repr(Vcut), reset='vm = ' + repr(V_res), refractory=refr_time, method='euler')


H = SpikeGeneratorGroup(1, [0], [500*ms])
S = Synapses(H, G, on_pre='ge_post += 1*nS\ngnmda_post += 1*nS')
S.connect(i=0, j=0)

S_alpha = Synapses(H, G_alpha, on_pre='ge_post += 1*nS')
S_alpha.connect(i=0, j=0)

S_expdec = Synapses(H, G_expdec, on_pre='ge_post += 1*nS')
S_expdec.connect(i=0, j=0)


statemon = StateMonitor(G, ('vm', 'gealpha', 'gnmda_alpha'), record=True)
statemon_alpha = StateMonitor(G_alpha, ('vm', 'gealpha'), record=True)
statemon_expdec = StateMonitor(G_expdec, ('vm', 'ge'), record=True)

G.I = 0.95 * rheobase
G_alpha.I = 0.95 * rheobase
G_expdec.I = 0.95 * rheobase
run(1000*ms)

plt.subplots(2,1)
plt.suptitle('Alpha approximation, non-normalized, tau 3ms, NMDA/AMPA 0.4')

plt.subplot(211)
ax1 = plt.gca()
plt.title('Vm trace')
plt.plot(statemon.t/ms, statemon.vm[0]/mV, label='AMPA+NMDA')
plt.plot(statemon_alpha.t/ms, statemon_alpha.vm[0]/mV, label='Alpha')
plt.plot(statemon_expdec.t/ms, statemon_expdec.vm[0]/mV, label='1-exp decay')
plt.legend()


plt.subplot(212, sharex=ax1)
plt.title('Conductance')
plt.plot(statemon.t/ms, statemon.gealpha[0]/nS, label='AMPA')
plt.plot(statemon.t/ms, statemon.gnmda_alpha[0]/nS, label='NMDA')
plt.plot(statemon.t/ms, (statemon.gealpha[0]+statemon.gnmda_alpha[0])/nS, label='AMPA+NMDA')
plt.plot(statemon_alpha.t/ms, statemon_alpha.gealpha[0]/nS, label='Alpha')
plt.plot(statemon_expdec.t/ms, statemon_expdec.ge[0]/nS, label='1-exp decay')
plt.legend()

plt.show()
