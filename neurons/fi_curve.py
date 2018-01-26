from brian2 import *

num_neurons = 500
duration = 1*second

refr_time = 4*ms




# Parameters
C = 181*pF
gL = 4.6*nS
tau_m = C/gL
EL = -74*mV
Vcut = 20*mV
DeltaT = 4*mV
VT = -56*mV

a = 0.198*nS
tau_w = 69.9*ms
b = 315*pA
V_res = -51.269*mV

eqs = '''
dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) -w + I) / C : volt (unless refractory)
dw/dt = (-a*(EL-vm)-w)/tau_w : amp
I : amp
'''
group = NeuronGroup(num_neurons, eqs, threshold='vm > ' + repr(Vcut),
                reset='vm = ' + repr(V_res) + '; w=w+' + repr(b),
                refractory=refr_time, method='euler')


group.vm = EL
group.I = '0.5*nA * i / num_neurons'

monitor = SpikeMonitor(group)

run(duration)

rheo_idx = min(np.where(monitor.count > 0)[0])
print group.I[rheo_idx]

plot(group.I/nA, monitor.count / duration, '.')
xlabel('I (nA)')
ylabel('Firing rate (sp/s)')
show()