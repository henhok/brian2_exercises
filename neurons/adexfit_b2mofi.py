from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from os import path, getcwd
import efel
import sys
from brian2tools import *

sys.path.append('/home/henhok/PycharmProjects/brian2modelfitting/')
from brian2modelfitting import *

prefs.codegen.target = 'numpy'


class MarkramStepInjectionTraces(object):
    """
    Extracts data from step current injection traces
    """

    def __init__(self, traces_directory='', traces_file_prefix='', current_steps=[], stim_total=3000, stim_start=700, stim_end=2700):
        # stim_total/start/end in milliseconds
        times = []
        voltages = []
        traces_file_suffix = '.dat'
        traces_file_pointer = path.join(getcwd(), traces_directory) + '/' + traces_file_prefix
        self.n_steps = len(current_steps)

        # Read the data from files and insert into eFel-compatible dictionary
        self.traces = []
        for step_number in range(1, self.n_steps):
            data = np.loadtxt(traces_file_pointer + str(step_number) + traces_file_suffix)
            times.append(data[:, 0])  # dt should be 0.1 ms
            voltages.append(data[:, 1])  # dt should be 0.1 ms

            trace = {}
            trace['T'] = times[step_number-1]
            trace['V'] = voltages[step_number-1]
            trace['stim_start'] = [stim_start]
            trace['stim_end'] = [stim_end]
            trace['step_current'] = [current_steps[step_number]]
            trace['hyperpolarization'] = [current_steps[0]]
            self.traces.append(trace)

        self.current_steps = current_steps
        self.stim_total = stim_total

        # Make brian2mofi-compatible arrays
        self.vm_traces = np.zeros((self.n_steps-1, stim_total * 10))
        self.input_traces = np.zeros((self.n_steps-1, stim_total * 10))

        for step_number in range(1, self.n_steps):
            self.vm_traces[step_number-1, :] = voltages[step_number-1][0:stim_total * 10] * mV
            self.input_traces[step_number-1, :] = current_steps[0] * np.ones(stim_total*10)
            self.input_traces[step_number-1, stim_start*10:stim_end*10] = self.input_traces[step_number-1, stim_start*10:stim_end*10] + current_steps[step_number]
            self.input_traces[step_number-1, :] = self.input_traces[step_number-1, :]*nA


    def plotInputOutput(self):
        plt.subplots(2,self.n_steps-1,sharey='row',sharex='all',figsize=(16,4))

        for i in range(self.n_steps-1):
            n_timepoints = len(self.vm_traces[i,:])
            # Plot output trace
            plt.subplot(2, self.n_steps-1, i+1)
            plt.plot(self.vm_traces[i,:]/mV)
            plt.ylabel('Vm (mV)')

            # Plot input trace
            plt.subplot(2, self.n_steps-1, i + self.n_steps)
            plt.plot(self.input_traces[i,:]/pA)
            plt.ylabel('I_input (pA)')

        plt.show(block=False)


    def plotTraces(self, optimizable_traces=None, ax=None):
        n_traces = self.n_steps - 1
        try:
            opt_traces_iter = iter(optimizable_traces)
        except TypeError:
            opt_traces_iter = None

        if ax is None:
            plt.subplots(1, n_traces)
            i=1
            for target_trace in self.traces:
                plt.subplot(1, n_traces, i)
                plt.plot(target_trace['T'], target_trace['V'], color='black', linewidth=1, linestyle='-')
                if opt_traces_iter is not None:
                    opt_trace = opt_traces_iter.next()
                    plt.plot(opt_trace['T'], opt_trace['V'], color='red', linewidth=0.5, linestyle='-')

                plt.xlim([0, self.stim_total])
                plt.title(str(target_trace['step_current'][0]) + ' nA')
                i += 1

            plt.show(block=False)

        else:
            i=0
            for target_trace in self.traces:
                # plt.subplot(1, n_traces, i)
                ax[i].plot(target_trace['T'], target_trace['V'], color='black', linewidth=1, linestyle='-')
                if opt_traces_iter is not None:
                    opt_trace = opt_traces_iter.next()
                    ax[i].plot(opt_trace['T'], opt_trace['V'], color='red', linewidth=0.5, linestyle='-')

                plt.xlim([0, self.stim_total])
                ax[i].set_title(str(target_trace['step_current'][0]) + ' nA')
                i += 1


    def getTargetValues(self, feature_list):
        feature_values = efel.getFeatureValues(self.traces, feature_list)
        return feature_values

    def optimize_adex(self, n_samples=5, n_rounds=5, opt_method='DE'):
        # Passive params for bbp_traces/L5_TTPC1_cADpyr232_4/hoc_recordings/
        DeltaT = 2 * mV
        C = 110 * pF
        gL = 3.1 * nS
        EL = -70 * mV
        VT = -42 * mV
        DeltaT = 4 * mV
        Vcut = 20 * mV
        refr_time = 4 * ms

        equation_soma = '''
        dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) -w + I_input) / C : volt (unless refractory)
        dw/dt = (a*(vm-EL)-w)/tau_w : amp
        a : siemens (constant)
        tau_w : second (constant)
        b : amp (constant)
        Vres : volt (constant)
        '''
        # Original equation:
        # G = NeuronGroup(num_neurons, equation_final, threshold='vm > ' + repr(neupar['Vcut']),
        #                 reset='vm = ' + repr(V_res) + '; w=w+' + repr(b),
        #                 refractory=neupar['refr_time'], method='euler')

        optimizer = NevergradOptimizer(method=opt_method, num_workers=4)
        traces_times = [[0 * ms, 3000 * ms]]
        feat_list = ['Spikecount_stimint', 'inv_time_to_first_spike', 'inv_first_ISI',
                     'inv_last_ISI', 'minimum_voltage']  # , 'min_voltage_between_spikes']
        feat_weights = {'Spikecount_stimint': 1,
                        'inv_time_to_first_spike': 1,
                        'inv_first_ISI': 1,
                        'inv_last_ISI': 1,
                        'minimum_voltage': 1}

        metric = FeatureMetric(traces_times, feat_list, weights=feat_weights, combine=None)
        # metric = MSEMetric()

        inp_traces = self.input_traces * amp
        out_traces = self.vm_traces * volt

        fitter = TraceFitter(model=equation_soma,
                             input=inp_traces,
                             output=out_traces,
                             input_var='I_input',
                             output_var='vm',
                             reset='vm = Vres; w=w+b',
                             refractory=refr_time,
                             threshold='vm > Vcut',
                             method='exponential_euler',
                             dt=0.1 * ms,
                             n_samples=n_samples,
                             param_init={'vm': EL})

        result, error = fitter.fit(optimizer=optimizer,
                                   metric=metric,
                                   n_rounds=n_rounds,
                                   a=[-10*nS, 10*nS],
                                   tau_w=[1*ms, 500*ms],
                                   b=[1 * pA, 300 * pA],
                                   Vres=[-80 * mV, -60 * mV])

        return result, error


if __name__ == '__main__':
    # Passive params for bbp_traces/L5_TTPC1_cADpyr232_4/hoc_recordings/
    DeltaT = 2 * mV
    C = 110 * pF
    gL = 3.1 * nS
    EL = -70 * mV
    VT = -42 * mV
    DeltaT = 4 * mV
    Vcut = 20 * mV
    refr_time = 4 * ms
    equation_soma = '''
    dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) -w + I_input) / C : volt (unless refractory)
    dw/dt = (a*(vm-EL)-w)/tau_w : amp
    I_input: amp
    '''

    current_steps = [-0.247559, 0.55425, 0.6004375, 0.646625]  # *nA
    test_target = MarkramStepInjectionTraces('bbp_traces/L5_TTPC1_cADpyr232_4/hoc_recordings/',
                                             'soma_voltage_step', current_steps)

    result, error = test_target.optimize_adex(n_rounds=300, n_samples=300, opt_method='DE')
    # Features-based fitting results...
    # CMA 10/30 -> 165; DE 10/30 -> 230
    # CMA 100/30 -> 82.5; DE 100/30 -> 82.5

    a = result['a'] * siemens
    tau_w = result['tau_w'] * second
    b = result['b'] * amp
    Vres = result['Vres'] * volt

    G = NeuronGroup(1, equation_soma, threshold='vm > Vcut',
                    reset='vm = Vres; w=w+b',
                    refractory=refr_time, method='exponential_euler')
    G.vm = EL

    statemon = StateMonitor(G, ['vm'], record=True)

    depol_start = 700*ms
    depol_end = 2700*ms
    stim_total = 3000*ms
    n_steps = 4

    # Run the current injections
    # Hyperpolarizing current
    G.I_input = current_steps[0] * nA
    run(depol_start)

    # Depolarizing steps
    # for step in range(1, n_steps):
    #     G.I_depol[step] = current_steps[step] * nA
    # run(depol_end - depol_start)
    G.I_input += current_steps[3] * nA
    run(depol_end - depol_start)

    # Stimulation ends
    G.I_input -= current_steps[3] * nA
    run(stim_total - depol_end)

    brian_plot(statemon)
    plt.show()

    # TODO
    # - Initialize vm etc
    # - What does traces_times actually mean?
    # - Is dt simultaneously the time scale for sims and trace data?
    # - What's n_samples and n_rounds? Size of population in each round?
    # - How is error computed? Scaling of errors could be useful cos otherwise it's hard for optimizer to know what's "negligible"
    # - Play around a bit more... after 10 rounds of DE, error still decreasing
    # - Why is "number of samples" limited to 30? => unexpected keyword argument "popsize"