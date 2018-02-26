from brian2 import *
import matplotlib.pyplot as plt


test_current = 72*pA

refr_time = 4*ms
defaultclock_dt = 0.1*ms  # Just for visualization! Changing this doesn't change the clock.
DeltaT = 2*mV

# SS cell (alternative; params wanted within physiological range)
C = 110*pF
gL = 3.1*nS
VT = -45*mV
Vcut = 20*mV
V_res = -70*mV
EL = -70*mV

# Synaptic parameters; redundant in this tool as there are no synaptic conductances
tau_ampa_rise = 0.2*ms
tau_ampa_decay = 1.73*ms
tau_nmda_rise = 0.29*ms
tau_nmda_decay = 43*ms
Ee = 0*mV
Ei = -75*mV
tau_m = C/gL

eqs_eif_biexp = '''
dvm/dt = (gL*(EL-vm) + gampa_alpha*(Ee-vm) + gnmda_alpha*(Ee-vm) + I + gL*DeltaT*exp((vm-VT) / DeltaT)) / C : volt (unless refractory)
dgampa/dt = -gampa/tau_ampa_decay : siemens
dgampa_alpha/dt = (gampa-gampa_alpha)/tau_ampa_rise : siemens
dgnmda/dt = -gnmda/tau_nmda_decay : siemens
dgnmda_alpha1/dt = (gnmda-gnmda_alpha1)/tau_nmda_rise : siemens
gnmda_alpha = gnmda_alpha1 * B : siemens
B = 1/(1+exp(-62*(vm/volt))*(1/3.57)) : 1
'''

G = NeuronGroup(1, eqs_eif_biexp, threshold='vm > '+repr(Vcut), reset='vm = '+repr(V_res), refractory=refr_time, method='euler')

H = PoissonGroup(1, 10*Hz)
S = Synapses(H, G, on_pre='gampa += 10*nS; gnmda += 10*nS')
S.connect(i=0, j=0)

statemon = StateMonitor(G, ('vm', 'gampa', 'gampa_alpha', 'gnmda', 'gnmda_alpha', 'B'), record=True)
spikemon = SpikeMonitor(H)



# Bfunc = lambda vm: 1/(1+exp(-0.062*vm)*(1/3.57))
# x_range = np.linspace(-70, 20, 100)
# y_val = [Bfunc(x) for x in x_range]
# plt.plot(x_range, y_val)
# plt.show()

I = 0*pA
run(3000*ms)
I = 50*pA
run(3000*ms)

plt.subplots(3,1)
plt.subplot(311)
ax1 = plt.gca()
plt.title('Vm trace')
plt.plot(statemon.t/ms, statemon.vm[0]/mV)
plt.plot(spikemon.t/ms, -30*np.ones(len(spikemon.t)), '.')

plt.subplot(312, sharex=ax1)
plt.title('ge alpha')
plt.plot(statemon.t/ms, statemon.gampa_alpha[0]/nS, label='AMPA')
plt.plot(statemon.t/ms, statemon.gnmda_alpha[0]/nS, label='NMDA')
plt.plot(statemon.t/ms, 10*statemon.B[0], label='B')
plt.plot(spikemon.t/ms, 0*np.ones(len(spikemon.t)), '.')
plt.legend()

plt.subplot(313, sharex=ax1)
plt.title('ge')
plt.plot(statemon.t/ms, statemon.gampa[0]/nS, label='AMPA')
plt.plot(statemon.t/ms, statemon.gnmda[0]/nS, label='NMDA')
plt.plot(spikemon.t/ms, 10*np.ones(len(spikemon.t)), '.')
plt.legend()

plt.show()
