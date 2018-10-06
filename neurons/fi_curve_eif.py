from brian2 import *

num_neurons = 500
duration = 1*second

refr_time = 4*ms




# Parameters
C = 41.18*pF
g_leak = 1.56*nS
tau_m = C/g_leak
Vr = -70.36*mV
Vcut = 20*mV
DeltaT = 4*mV
VT = -55.09*mV

V_res = -76.64*mV

eqs = '''
dvm/dt = (g_leak*(Vr-vm) + g_leak * DeltaT * exp((vm-VT) / DeltaT) + I) / C : volt (unless refractory)
I : amp
'''
group = NeuronGroup(num_neurons, eqs,
                    threshold='vm > ' + repr(Vcut),
                    reset='vm = ' + repr(V_res),
                    refractory=refr_time,
                    method='euler')

group.vm = Vr
group.I = '0.5*nA * i / num_neurons'

monitor = SpikeMonitor(group)

run(duration)

rheo_idx = min(np.where(monitor.count > 0)[0])
print group.I[rheo_idx]

plot(group.I/nA, monitor.count / duration, '.')
xlabel('I (nA)')
ylabel('Firing rate (sp/s)')
show()