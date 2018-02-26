from __future__ import division
from brian2 import *
from os import getcwd, path
import numpy as np
import matplotlib.pyplot as plt

defaultclock.dt = 0.1*ms

# Hodgkin-Huxley & AdEx Parameters
area = 20000*umetre**2
Cm = (1*ufarad*cm**-2) * area
gl = (5e-5*siemens*cm**-2) * area

El = -60*mV
EK = -90*mV
ENa = 50*mV
g_na = (100*msiemens*cm**-2) * area
g_kd = (30*msiemens*cm**-2) * area
VT = -63*mV
# Time constants
taue = 5*ms
taui = 10*ms
# Reversal potentials
Ee = 0*mV
Ei = -80*mV
we = 6*nS  #excitatory synaptic weight
wi = 67*nS   # nS inhibitory synaptic weight

# AdEx parameters
a,tau_w,b,V_res = [2, 30, 20, -80]
DeltaT = 2*mV
Vcut = 20*mV
refr_time = 3*ms

# a,tau_w,b,V_res,VT,DeltaT=
# VT = VT * mV; DeltaT = DeltaT * mV

a = a*nS; tau_w = tau_w*ms; b = b*pA; V_res = V_res*mV

# The Hodgkin-Huxley model
eqs_hh = Equations('''
dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK) + I_hypol + I_depol)/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
         (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
        (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
         (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
I_hypol : amp
I_depol : amp
''')

eqs_adex = '''
dv/dt = (gl*(El-v) + ge*(Ee-v) + gi*(Ei-v) + gl*DeltaT*exp((v-VT) / DeltaT) -w) / Cm : volt (unless refractory)
dw/dt = (-a*(El-v)-w)/tau_w : amp
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
'''

eqs_eif = '''
dv/dt = (gl*(El-v) + ge*(Ee-v) + gi*(Ei-v) + gl*DeltaT*exp((v-VT) / DeltaT)) / Cm : volt (unless refractory)
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
'''

def step_injections_hh(current_steps=[0, 0.7], plot_traces=False, traces_dir=None):

    stim_total = 3000*ms
    stim_start = 700*ms
    stim_end = 2700*ms

    n_steps = len(current_steps)
    G = NeuronGroup(n_steps, model=eqs_hh, threshold='v>-20*mV', refractory=3 * ms,
                    method='exponential_euler')
    G.v = El
    G.I_hypol = 0
    G.I_depol = 0
    M = StateMonitor(G, ('v'), record=True)

    G.I_hypol = current_steps[0] * nA
    run(stim_start)

    for i_step in range(1, n_steps):
        G.I_depol[i_step] = current_steps[i_step] * nA
    run(stim_end - stim_start)

    G.I_depol = 0 * nA
    run(stim_total - stim_end)

    if traces_dir is not None:
        traces_path = path.join(getcwd(), traces_dir)
        for i in range(1,n_steps):
            trace_filename = 'soma_voltage_step'+str(i)+'.dat'
            trace_file_abs = traces_path + '/' + trace_filename
            data = []
            data.append(M.t/ms)
            data.append(M.v[i]/mV)
            data = np.array(data)
            np.savetxt(trace_file_abs, data.transpose())
        print 'Traces saved to ' + traces_dir


    if plot_traces is True:
        plt.subplots(1, n_steps-1)
        for i_subplot in range(1, n_steps):
            plt.subplot(1, n_steps-1, i_subplot)
            plt.plot(M.t/ms, M.v[i_subplot]/mV)
            plt.title(str(current_steps[i_subplot])+' nA')

        plt.show()


def plot_coba(trace, spikes, title=''):
    plt.subplots(1, 2)
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plot(spikes.t / ms, spikes.i, ',k')
    xlabel('Time (ms)')
    ylabel('Neuron index')

    plt.subplot(1, 2, 2)
    plot(trace.t / ms, trace[1].v / mV)
    plot(trace.t / ms, trace[10].v / mV)
    plot(trace.t / ms, trace[100].v / mV)
    xlabel('t (ms)')
    ylabel('v (mV)')

    show()

def coba_reporter(elapsed, completed, start, duration):
    if completed % 0.25 == 0:
        print '%f completed\t\t%.3f elapsed' % (completed, elapsed)
    else:
        pass


def run_coba(model='hh', coba_runtime=1.0 * second, coba_scale=1, k=1, report='text', profile=False, show_traces=False, ax_raster=None):

    assert model in ['hh', 'adex', 'eif'], "Check neuron model name!"
    if model == 'hh':
        # Threshold&refr-time just for spike counting
        P = NeuronGroup(coba_scale * 4000, model=eqs_hh, threshold='v>-20*mV', refractory=3 * ms,
                        method='exponential_euler')

    elif model == 'adex':
        P = NeuronGroup(coba_scale * 4000, model=eqs_adex, threshold='v>' + repr(Vcut),
                        reset='v = ' + repr(V_res) + '; w=w+' + repr(b),
                        refractory=refr_time, method='exponential_euler')

    else:
        P = NeuronGroup(coba_scale * 4000, model=eqs_eif, threshold='v>' + repr(Vcut),
                        reset='v = ' + repr(V_res),
                        refractory=refr_time, method='exponential_euler')

    Pe = P[:coba_scale*3200]
    Pi = P[coba_scale*3200:]

    coba_scale_factor = (1. / 2 ** (-(coba_scale - 1)))

    we_scaled = we * coba_scale_factor
    wi_scaled = wi * coba_scale_factor * k
    Ce = Synapses(Pe, P, on_pre='ge+=we_scaled')
    Ci = Synapses(Pi, P, on_pre='gi+=wi_scaled')
    Ce.connect(p=0.02)
    Ci.connect(p=0.02)

    # Initialization
    P.v = 'El + (randn() * 5 - 5)*mV'
    P.ge = '(randn() * 1.5 + 4) * 10. * %f *nS' % coba_scale_factor
    P.gi = '(randn() * 12 + 20) * 10. * %f *nS' % coba_scale_factor

    # Record a few traces
    trace = StateMonitor(P, 'v', record=[1, 10, 100])

    # Record spikes
    spikes = SpikeMonitor(P)

    run(coba_runtime, report=report, profile=profile)
    if profile is True:
        print profiling_summary()

    if ax_raster is not None:
        ax_raster.plot(spikes.t / ms, spikes.i, ',k')

    if show_traces is True:
        plot_coba(trace, spikes, str.upper(model))


def run_coba_stp(model='hh', coba_runtime=1.0 * second, coba_scale=1, k=1, report='text', profile=False, show_traces=False, ax_raster=None):

    assert model in ['hh', 'adex', 'eif'], "Check neuron model name!"
    if model == 'hh':
        # Threshold&refr-time just for spike counting
        P = NeuronGroup(coba_scale * 4000, model=eqs_hh, threshold='v>-20*mV', refractory=3 * ms,
                        method='exponential_euler')

    elif model == 'adex':
        P = NeuronGroup(coba_scale * 4000, model=eqs_adex, threshold='v>' + repr(Vcut),
                        reset='v = ' + repr(V_res) + '; w=w+' + repr(b),
                        refractory=refr_time, method='exponential_euler')

    else:
        P = NeuronGroup(coba_scale * 4000, model=eqs_eif, threshold='v>' + repr(Vcut),
                        reset='v = ' + repr(V_res),
                        refractory=refr_time, method='exponential_euler')

    Pe = P[:coba_scale*3200]
    Pi = P[coba_scale*3200:]

    coba_scale_factor = (1. / 2 ** (-(coba_scale - 1)))
    we_scaled = we * coba_scale_factor
    wi_scaled = wi * coba_scale_factor * k

    tau_d = 670*ms
    U_E = 0.05
    U_I = 0.05
    synapse_model_exc = ''' R = R + (1-R)*(1 - exp(-(t-lastupdate)/tau_d))
                            ge += R * U_E * we_scaled
                            R = R - U_E * R'''

    synapse_model_inh = ''' R = R + (1-R)*(1 - exp(-(t-lastupdate)/tau_d))
                            gi += R * U_I * wi_scaled
                            R = R - U_I * R'''


    Ce = Synapses(Pe, P, model='R:1', on_pre=synapse_model_exc)
    Ci = Synapses(Pi, P, model='R:1', on_pre=synapse_model_inh)
    Ce.connect(p=0.02)
    Ci.connect(p=0.02)
    Ce.R = 1
    Ci.R = 1


    # Initialization
    P.v = 'El + (randn() * 5 - 5)*mV'
    P.ge = '(randn() * 1.5 + 4) * 10. * %f *nS' % coba_scale_factor
    P.gi = '(randn() * 12 + 20) * 10. * %f *nS' % coba_scale_factor

    # Record a few traces
    trace = StateMonitor(P, 'v', record=[1, 10, 100])

    # Record spikes
    spikes = SpikeMonitor(P)

    run(coba_runtime, report=report, profile=profile)
    if profile is True:
        print profiling_summary()

    if ax_raster is not None:
        ax_raster.plot(spikes.t / ms, spikes.i, ',k')

    if show_traces is True:
        plot_coba(trace, spikes, str.upper(model))


if __name__ == '__main__':

    run_coba(model='eif', coba_runtime=1 * second, show_traces=True)

    # k_list = [0.5, 1.0, 1.5]
    # k_total = len(k_list)
    # fig, ax = plt.subplots(1, k_total)
    # i=0
    # for k in k_list:
    #     run_coba_hh(coba_runtime=1*second, k=k, ax_raster=ax[i])
    #     i += 1
    #
    # plt.show()

    # BENCHMARK
    # trials = 10
    # for i in range(trials):
    #     print '%d\t trial\t HH' % (i+1)
    #     run_coba_hh(coba_runtime=10*second, report='text')

    # trials = 10
    # for i in range(trials):
    #     print '%d\t trial\t HH' % (i+1)
    #     run_coba_stp(model='hh', coba_runtime=10*second, report='text')