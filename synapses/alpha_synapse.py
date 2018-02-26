from brian2 import *
import matplotlib.pyplot as plt


test_current = 72*pA

refr_time = 4*ms
defaultclock_dt = 0.01*ms  # Just for visualization! Changing this doesn't change the clock.
defaultclock.dt = defaultclock_dt
DeltaT = 2*mV

# SS cell (alternative; params wanted within physiological range)
C = 110*pF
gL = 3.1*nS
VT = -45*mV
Vcut = 20*mV
V_res = -70*mV
EL = -70*mV

# Synaptic parameters; redundant in this tool as there are no synaptic conductances
tau_e_alpha = 3*ms
Ee = 0*mV
Ei = -75*mV
tau_m = C/gL

t_peak = tau_e_alpha
#K = (exp(-t_peak/tau_e_decay) - exp(-t_peak/tau_e_rise))**-1
# a = tau_e_decay - tau_e_rise
# b = tau_e_decay/tau_e_rise
# K = exp(-tau_e_rise/a * log(b)) - exp(-tau_e_decay/a * log(b))

K=exp(1)
print str(K)

eqs_eif_biexp = '''
dvm/dt = (gL*(EL-vm) + K*gealpha*(Ee-vm) + gL*DeltaT*exp((vm-VT) / DeltaT)) / C : volt (unless refractory)
dge/dt = -ge/tau_e_alpha : siemens
dgealpha/dt = (ge-gealpha)/tau_e_alpha : siemens
'''

G = NeuronGroup(1, eqs_eif_biexp, threshold='vm > '+repr(Vcut), reset='vm = '+repr(V_res), refractory=refr_time, method='euler')

H = PoissonGroup(1, 3*Hz)
S = Synapses(H, G, on_pre='ge_post += 10*nS')
S.connect(i=0, j=0)

statemon = StateMonitor(G, ('vm', 'ge', 'gealpha'), record=True)
spikemon = SpikeMonitor(H)

run(2000*ms)

plt.subplots(3,1)
plt.subplot(311)
ax1 = plt.gca()
plt.title('Vm trace')
plt.plot(statemon.t/ms, statemon.vm[0]/mV)
plt.plot(spikemon.t/ms, -30*np.ones(len(spikemon.t)), '.')

plt.subplot(312, sharex=ax1)
plt.title('ge alpha')
plt.plot(statemon.t/ms, K*statemon.gealpha[0]/nS)
plt.plot(spikemon.t/ms, 0*np.ones(len(spikemon.t)), '.')
plt.plot(t_peak/ms + spikemon.t/ms, 5*np.ones(len(spikemon.t)), '.', color='red')

plt.subplot(313, sharex=ax1)
ax = plt.gca()
plt.title('ge')
plt.plot(statemon.t/ms, statemon.ge[0]/nS)
plt.plot(spikemon.t/ms, 10*np.ones(len(spikemon.t)), '.')
for x0 in (t_peak/ms + spikemon.t/ms):
    ax.axvline(x0, linestyle='--')

plt.show()

plt.figure()
plt.scatter(statemon.ge[0]/nS, statemon.gealpha[0]/nS)
plt.xlabel('ge')
plt.ylabel('ge_alpha')
ax=plt.gca()
ax.set_aspect('equal')
plt.show()