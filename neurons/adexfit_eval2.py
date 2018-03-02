"""
Fitness evaluation for adaptive exponential integrate-and-fire neurons
using traces from simulated BBP neurons
by Henri Hokkanen <henri.hokkanen@helsinki.fi>, January 2018

Neuron models from
1) https://bbp.epfl.ch/nmc-portal/downloads

Implementation inspired by
2) https://github.com/BlueBrain/eFEL/blob/master/examples/nmc-portal/L5TTPC2.ipynb
3) http://efel.readthedocs.io/en/latest/deap_optimisation.html
4) Naud et al. 2008

For generating step injection traces
  - download neuron model from (1)
  - follow instructions in (2) on how to run the model
"""

from __future__ import division
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from os import path, getcwd
import efel
from scipy.integrate import trapz
from copy import deepcopy
prefs.codegen.target = 'numpy'

defaultclock.dt = 0.1*ms
efel.api.setThreshold(-35)  # Lowering threshold so that eFel recognizes EIF/AdEx spikes


class MarkramStepInjectionTraces(object):
    """
    Extracts data from step current injection traces
    """

    def __init__(self, traces_directory='', traces_file_prefix='', current_steps=[], stim_total=3000, stim_start=700, stim_end=2700):
        times = []
        voltages = []
        traces_file_suffix = '.dat'
        traces_file_pointer = path.join(getcwd(), traces_directory) + '/' + traces_file_prefix
        self.n_steps = len(current_steps)

        # Read the data from files and insert into eFel-compatible dictionary
        self.traces = []
        for step_number in range(1, self.n_steps):
            data = np.loadtxt(traces_file_pointer + str(step_number) + traces_file_suffix)
            times.append(data[:, 0])
            voltages.append(data[:, 1])

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

            plt.show()

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


class AdexOptimizable(object):
    """
    Runs experiments on an AdEx neuron & evaluates fitness
    """

    def __init__(self, passive_params, target_traces=MarkramStepInjectionTraces(), efel_feature_names=None, custom_feature_names=[], dendritic_extent=0, stim_total=3000, stim_start=700, stim_end=2700):
        self.adex_passive_params = passive_params

        self.target = target_traces

        self.stim_total = stim_total
        self.stim_start = stim_start
        self.stim_end = stim_end

        # Constants
        self.TIME_INFTY = (self.stim_end - self.stim_start) * 2
        self.VM_PRESTIM_FLAT = -60
        self.VM_PRESPIKE_FLAT = 20

        assert type(dendritic_extent) is int and dendritic_extent >= 0, \
            "Dendritic extent must be an integer >= 0"

        self.dendritic_extent = dendritic_extent

        if dendritic_extent == 0:  # NON-pyramidal cells
            self.equation_soma = '''
            dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) -w + I_hypol + I_depol) / C : volt (unless refractory)
            dw/dt = (a*(vm-EL)-w)/tau_w : amp
            I_depol : amp
            I_hypol : amp
            '''
        else:  # Pyramidal cells
            # Adaptation term in soma
            self.eq_template_soma = '''
            dvm/dt = ((gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) -w +I_dendr + I_hypol + I_depol) / C) : volt (unless refractory)
            dw/dt = (a*(vm-EL)-w)/tau_w : amp
            I_depol : amp
            I_hypol : amp
            '''
            # NO adaptation term in dendrites
            self.eq_template_dend = '''
            dvm/dt = (gL*(EL-vm) + I_dendr) / C : volt
            '''
            # self.preparePyramidalEqs()

        # Objectives: number of spikes, first spike latency, first ISI, last ISI (...) => squared error
        # For possible features: efel.api.getFeatureNames()
        self.all_custom_features = ['prestim_waveform_diff', 'prespike_waveform_diff']

        # assert...
        self.efel_feature_names = efel_feature_names
        self.custom_feature_names = custom_feature_names

        self.target_values = self.target.getTargetValues(self.efel_feature_names)
        self.target_traces = self.target.traces

    def preparePyramidalEqs(self, individual):

        # Repetition here (sorry, coding in a hurry)
        neupar = self.adex_passive_params
        if len(individual) == 4:
            a, tau_w, b, V_res = individual
        else:
            a, tau_w, b, V_res, VT, DeltaT = individual
            neupar['VT'] = VT*mV
            neupar['DeltaT'] = DeltaT*mV

        a = a*nS; tau_w = tau_w*ms; b = b*pA; V_res = V_res*mV

        # Dendritic parameters
        neuron_namespace = dict()
        Cm = 1 * uF * cm ** -2
        gl = 4.2e-5 * siemens * cm ** -2
        Area_tot_pyram = 25000 * 0.75 * um ** 2

        fract_areas = {1: array([0.2, 0.03, 0.15, 0.2]),
                       2: array([0.2, 0.03, 0.15, 0.15, 0.2]),
                       3: array([0.2, 0.03, 0.15, 0.09, 0.15, 0.2]),
                       4: array([0.2, 0.03, 0.15, 0.15, 0.09, 0.15, 0.2])}

        neuron_namespace['Ra'] = [100, 80, 150, 150, 200] * Mohm

        neuron_namespace['C'] = fract_areas[self.dendritic_extent] * \
                                Cm * Area_tot_pyram * 2        # *2 to correct for dendritic spine area
        neuron_namespace['gL'] = fract_areas[self.dendritic_extent] * \
                                 gl * Area_tot_pyram
        neuron_namespace['taum_soma'] = neuron_namespace['C'][1] / neuron_namespace['gL'][1]


        self.pyramidal_eqs = Equations(self.eq_template_dend, vm="vm_basal", ge="ge_basal",
                               gealpha="gealpha_basal",
                               C=neuron_namespace['C'][0],
                               gL=neuron_namespace['gL'][0],
                               gi="gi_basal", geX="geX_basal", gialpha="gialpha_basal",
                               gealphaX="gealphaX_basal", I_dendr="Idendr_basal")

        self.pyramidal_eqs += Equations(self.eq_template_soma, gL=neuron_namespace['gL'][1],
                                ge='ge_soma', geX='geX_soma', gi='gi_soma', gealpha='gealpha_soma',
                                gealphaX='gealphaX_soma',
                                gialpha='gialpha_soma', C=neuron_namespace['C'][1],
                                I_dendr='Idendr_soma',
                                taum_soma=neuron_namespace['taum_soma']) #, w='w_soma')

        for _ii in range(self.dendritic_extent + 1):  # extra dendritic compartment in the same level of soma
            self.pyramidal_eqs += Equations(self.eq_template_dend, vm="vm_a%d" % _ii,
                                    C=neuron_namespace['C'][_ii],
                                    gL=neuron_namespace['gL'][_ii],
                                    ge="ge_a%d" % _ii,
                                    gi="gi_a%d" % _ii, geX="geX_a%d" % _ii,
                                    gealpha="gealpha_a%d" % _ii, gialpha="gialpha_a%d" % _ii,
                                    gealphaX="gealphaX_a%d" % _ii, I_dendr="Idendr_a%d" % _ii)  #, w='w_a%d' % _ii)

        # Defining decay between soma and basal dendrite & apical dendrites
        self.pyramidal_eqs += Equations('I_dendr = gapre*(vmpre-vmself)  : amp',
                                gapre=1 / (neuron_namespace['Ra'][0]),
                                I_dendr="Idendr_basal", vmself="vm_basal", vmpre="vm")
        self.pyramidal_eqs += Equations('I_dendr = gapre*(vmpre-vmself)  + gapost*(vmpost-vmself) : amp',
                                gapre=1 / (neuron_namespace['Ra'][1]),
                                gapost=1 / (neuron_namespace['Ra'][0]),
                                I_dendr="Idendr_soma", vmself="vm",
                                vmpre="vm_a0", vmpost="vm_basal")
        self.pyramidal_eqs += Equations('I_dendr = gapre*(vmpre-vmself) + gapost*(vmpost-vmself) : amp',
                                gapre=1 / (neuron_namespace['Ra'][2]),
                                gapost=1 / (neuron_namespace['Ra'][1]),
                                I_dendr="Idendr_a0", vmself="vm_a0", vmpre="vm_a1", vmpost="vm")

        # Defining decay between apical dendrite compartments
        for _ii in arange(1, self.dendritic_extent):
            self.pyramidal_eqs += Equations('I_dendr = gapre*(vmpre-vmself) + gapost*(vmpost-vmself) : amp',
                                    gapre=1 / (neuron_namespace['Ra'][_ii]),
                                    gapost=1 / (neuron_namespace['Ra'][_ii - 1]),
                                    I_dendr="Idendr_a%d" % _ii, vmself="vm_a%d" % _ii,
                                    vmpre="vm_a%d" % (_ii + 1), vmpost="vm_a%d" % (_ii - 1))

        self.pyramidal_eqs += Equations('I_dendr = gapost*(vmpost-vmself) : amp',
                                I_dendr="Idendr_a%d" % self.dendritic_extent,
                                gapost=1 / (neuron_namespace['Ra'][-1]),
                                vmself="vm_a%d" % self.dendritic_extent,
                                vmpost="vm_a%d" % (self.dendritic_extent - 1))


    def _listMean(self, key, val):
        """
        Handling of possible lists from feature extraction

        :param val:
        :return:
        """
        if val is None:
            return 0
        else:
            try:
                return float(val)
            except:
                if len(val) > 0:
                    return mean(val)
                else:
                    return 0

    def _listFirst(self, key, val):
        """
        Handling of lists/values from feature extraction

        :param val:
        :return:
        """


        if val is None:  # eg. no spikes
            return 0
        elif key[:3] == 'inv':  # inv stands for 1/x (ie. 'inv_time_to_first_spike', 'inv_first_ISI', 'inv_last_ISI')
            if val == 0:
                return self.TIME_INFTY
            else:
                return (1/val)*1000  # we assume here val is in seconds (converted to ms)

        else:  # not None
            try:
                return float(val)
            except:
                if len(val) > 0:
                    return float(val[0])
                else:
                    return 0

    def prespikeWaveformArea(self, voltage_traces, spike_latencies):
        """
        Computes area of traces after stimulus start up to first spike.
        Uses scipy.integrate.trapz for integration.

        :param voltage_traces: list of dicts (see evaluateFitness)
        :param spike_latencies: latencies from stimulus start to first spike
        :return:
        """

        infinity=1e9  # shouldn't be needed; will pop out clearly
        n_steps = self.target.n_steps
        stim_start_loc = self.stim_start * 10   # sampling is at 0.1ms intervals & stim_start given in ms
        end_loc = (np.array(spike_latencies)*10 + stim_start_loc).astype(int)

        areas_btw_curves = []
        for step_number in range(0, n_steps-1):
            if end_loc[step_number]/10 < self.stim_end:
                try:
                    flat_trace = self.VM_PRESPIKE_FLAT*np.ones(end_loc[step_number] - stim_start_loc)  # horizontal/flat trace at +20mV
                    diff_trace = flat_trace - voltage_traces[step_number]['V'][stim_start_loc:end_loc[step_number][0]]
                    diff_trace = abs(diff_trace)
                    areas_btw_curves.append(trapz(diff_trace, dx=0.1))
                except:
                    areas_btw_curves.append(infinity)
            else:
                areas_btw_curves.append(infinity)

        return areas_btw_curves


    def prespikeWaveformDifference(self, optimizable_traces, spike_latencies):
        """
        Computes area between traces after stimulus start up to integration_duration * ms.
        Uses scipy.integrate.trapz for integration.

        :param optimizable_traces: list of dicts (see evaluateFitness)
        :param spike_latencies: latencies from stimulus start to first spike
        :return:
        """
        infinity=10e9
        n_steps = self.target.n_steps
        stim_start_loc = self.stim_start * 10   # sampling is at 0.1ms intervals & stim_start given in ms

        target_latencies = [self.target_values[i]['inv_time_to_first_spike'] for i in range(0, n_steps - 1)]
        spike_latencies_mean = [mean([spike_latencies[i], target_latencies[i]]) for i in range(0,n_steps-1)]

        end_loc = (np.array(spike_latencies_mean)*10 + stim_start_loc).astype(int)

        areas_btw_curves = []
        for step_number in range(0, n_steps-1):
            try:
                diff_trace = np.array(self.target_traces[step_number]['V'][stim_start_loc:end_loc[step_number]]) - \
                             np.array(optimizable_traces[step_number]['V'][stim_start_loc:end_loc[step_number]])
                diff_trace = abs(diff_trace)
                areas_btw_curves.append(trapz(diff_trace, dx=0.1))
            except:
                areas_btw_curves.append(infinity)

        return areas_btw_curves

    def prestimulusWaveformDifference(self, optimizable_traces, flat_target_trace=False):
        """
        Computes area between traces before stimulus start.
        Uses scipy.integrate.trapz for integration.

        :param optimizable_traces: list of dicts (see evaluateFitness)
        :return:
        """
        stim_start_loc = self.stim_start * 10   # sampling is at 0.1ms intervals & stim_start given in ms
        step_number = 0  # prestimulus behavior exactly the same in all experiments

        if flat_target_trace is True:
            target_voltages = self.VM_PRESTIM_FLAT * np.ones(stim_start_loc)
        else:
            target_voltages = np.array(self.target_traces[step_number]['V'][:stim_start_loc])

        diff_trace = target_voltages - np.array(optimizable_traces[step_number]['V'][:stim_start_loc])
        diff_trace = abs(diff_trace)
        area_btw_curves = trapz(diff_trace, dx=0.1)

        return area_btw_curves

    def plotFICurve(self, individual, only_rheobase=False, ax=None):

        # Prepare neuron group
        duration = 500*ms
        num_neurons = 500

        # // Repetition...
        neupar = self.adex_passive_params
        if len(individual) == 4:
            a, tau_w, b, V_res = individual
        else:
            a, tau_w, b, V_res, VT, DeltaT = individual
            neupar['VT'] = VT*mV
            neupar['DeltaT'] = DeltaT*mV

        # n_steps = self.target.n_steps
        # current_steps = self.target.current_steps

        # Assign units to raw numbers
        a = a*nS; tau_w = tau_w*ms; b = b*pA; V_res = V_res*mV

        # Make neuron group
        # NON-pyramidal cells
        if self.dendritic_extent == 0:
            equation_final = Equations(self.equation_soma, C=neupar['C'], gL=neupar['gL'], EL=neupar['EL'],
                                       DeltaT=neupar['DeltaT'], VT=neupar['VT'],
                                       a=a, tau_w=tau_w)
        # PYRAMIDAL cells
        else:
            self.preparePyramidalEqs(individual)
            equation_final = self.pyramidal_eqs
            EL = neupar['EL']; DeltaT = neupar['DeltaT']; VT = neupar['VT']

        G = NeuronGroup(num_neurons, equation_final, threshold='vm > ' + repr(neupar['Vcut']),
                        reset='vm = ' + repr(V_res) + '; w=w+' + repr(b),
                        refractory=neupar['refr_time'], method='euler')

        if self.dendritic_extent == 0:
            G.vm = neupar['EL']  # NB! eFel will fail without this line (for unknown reasons)
        else:
            G.vm = neupar['EL']
            G.vm_basal = neupar['EL']
            for _ii in range(self.dendritic_extent + 1):
                setattr(G, "vm_a%d" % _ii, neupar['EL'])
        # // Repetition ends...

        G.I_depol = '1*nA * i / num_neurons'

        monitor = SpikeMonitor(G)
        run(duration)

        try:
            rheo_idx = min(np.where(monitor.count > 0)[0])
        except:
            rheo_idx = -1

        if ax is None:
            if only_rheobase is False:
                print 'Rheobase: ' + str(G.I_depol[rheo_idx])
                plt.figure()
                plt.plot(G.I_depol / nA, monitor.count / duration, '.')
                plt.xlabel('I (nA)')
                plt.ylabel('Firing rate (sp/s)')
                plt.show()

            else:
                if rheo_idx >= 0:
                    return G.I_depol[rheo_idx]
                else:
                    return 2*nA

        else:
            ax.plot(G.I_depol / nA, monitor.count / duration)
            ax.set_title('F-I curve')
            ax.set_xlim([0, 0.5])

    def isFeasible(self, individual):
        N_params = len(individual)
        if N_params not in [4, 6]:
            return False

        # Initial setup
        V_res = individual[3]
        if N_params == 4:
            VT = self.adex_passive_params['VT']
        else:
            VT = individual[5]

        # Actual checking
        if V_res >= VT:
            return False
        else:
            return True


    def evaluateFitness(self, individual, plot_traces=False, verbose=False):
        """
        Runs model with given parameters and evaluates results with respect to target features

        :param individual: [a, tau_w, b, V_res] in units nS, ms, pA, mV, respectively
        :return: list of errors, length depending number of extracted features
        """
        if verbose is True:
            print 'Current AdEx params (a, tau_w, b, V_res): ' + str(individual)

        failed_individual_costs = np.ones(len(self.efel_feature_names) + len(self.custom_feature_names))*500

        # 1. CURRENT INJECTIONS
        # 1.1. Prepare for running current injections
        # Better variable names for sake of clarity
        neupar = self.adex_passive_params
        if len(individual) == 4:
            a, tau_w, b, V_res = individual
        else:
            a, tau_w, b, V_res, VT, DeltaT = individual
            neupar['VT'] = VT*mV
            neupar['DeltaT'] = DeltaT*mV

            if V_res >= VT:  # we don't resetting above threshold!
                return failed_individual_costs

        n_steps = self.target.n_steps
        current_steps = self.target.current_steps

        # Assign units to raw numbers
        a = a*nS; tau_w = tau_w*ms; b = b*pA; V_res = V_res*mV

        # Make neuron group
        # NON-pyramidal cells
        if self.dendritic_extent == 0:
            equation_final = Equations(self.equation_soma, C=neupar['C'], gL=neupar['gL'], EL=neupar['EL'],
                                       DeltaT=neupar['DeltaT'], VT=neupar['VT'],
                                       a=a, tau_w=tau_w)
        # PYRAMIDAL cells
        else:
            self.preparePyramidalEqs(individual)
            equation_final = self.pyramidal_eqs
            EL = neupar['EL']; DeltaT = neupar['DeltaT']; VT = neupar['VT']


        G = NeuronGroup(n_steps, equation_final, threshold='vm > ' + repr(neupar['Vcut']),
                        reset='vm = ' + repr(V_res) + '; w=w+' + repr(b),
                        refractory=neupar['refr_time'], method='euler')

        if self.dendritic_extent == 0:
            G.vm = neupar['EL']  # NB! eFel will fail without this line (for unknown reasons)
        else:
            G.vm = neupar['EL']
            G.vm_basal = neupar['EL']
            for _ii in range(self.dendritic_extent + 1):
                setattr(G, "vm_a%d" % _ii, neupar['EL'])

        M = StateMonitor(G, ('vm'), record=True)
        spikes = SpikeMonitor(G)

        # 1.2. Run the current injections
        # Hyperpolarizing current
        G.I_hypol = current_steps[0] * nA
        run(self.stim_start * ms)

        # Depolarizing steps
        for step in range(1, n_steps):
            G.I_depol[step] = current_steps[step] * nA
        run((self.stim_end-self.stim_start) * ms)

        # Stimulation ends
        G.I_depol = 0*nA
        run((self.stim_total-self.stim_end) * ms)

        # 1.3. Extract voltage traces
        optimizable_traces = []
        traces = []
        for step_number in range(1, n_steps):

            trace = {}
            trace['T'] = M.t/ms
            trace['V'] = M.vm[step_number]/mV
            trace['stim_start'] = [self.stim_start]
            trace['stim_end'] = [self.stim_end]
            trace['step_current'] = [current_steps[step_number]]
            trace['hyperpolarization'] = [current_steps[0]]
            optimizable_traces.append(trace)

        # 2. COMPUTE FITNESS
        # 2.1. eFel-features
        # raise_warning disabled to allow non-spiking configs
        individual_values = efel.getFeatureValues(optimizable_traces, self.efel_feature_names, raise_warnings=False)
        individual_values_raw = deepcopy(individual_values)  # extempore solution here

        # Preprocess value lists (invert inverses; take first value or mean)
        # handle_list = lambda (k, v): self._listFirst(k, v)
        for step in range(0, n_steps - 1):
            individual_values[step] = {k: self._listFirst(k, v) for k, v in individual_values[step].items()}
            self.target_values[step] = {k: self._listFirst(k, v) for k, v in self.target_values[step].items()}

        # eFel fails to count spikes with some AdEx parameters, so safer to use Brian2's spikemonitor
        if 'Spikecount_stimint' in self.efel_feature_names:
            for step in range(0, n_steps-1):
                individual_values[step]['Spikecount_stimint'] = spikes.count[step+1]

        if 'min_voltage_between_spikes' in self.efel_feature_names:
            for step in range(0, n_steps-1):
                step_spikes = spikes.t[spikes.i == step+1]
                try:
                    first_spike = int(step_spikes[0]/ms*10)  # bc vm monitor indexed with ints at 0.1ms interval
                    second_spike = int(step_spikes[1]/ms*10)
                    min_volt = min(M.vm[step+1][first_spike:second_spike])/mV
                except:
                    min_volt = -100

                individual_values[step]['min_voltage_between_spikes'] = min_volt

        # Calculate errors in extracted features by averaging over all current steps
        feature_errors=[]
        for feature in self.efel_feature_names:
            abs_errors = [abs(individual_values[i][feature] - self.target_values[i][feature]) for i in range(0, n_steps - 1)]

            # // Update 2018-01-26 begins
            acceptable_errors = [0.05*abs(self.target_values[i][feature]) for i in range(0, n_steps - 1)]
            scaled_abs_errors = [abs_errors[i]/acceptable_errors[i] for i in range(0, n_steps - 1)]
            avg_error = mean(scaled_abs_errors)
            #avg_error = mean(abs_errors)
            # // Update ends

            feature_errors.append(avg_error)
            if verbose is True:
                print '---'
                print feature + ' current: ' + str([individual_values[i][feature] for i in range(0, n_steps-1)])
                print feature + ' target:  ' + str([self.target_values[i][feature] for i in range(0, n_steps - 1)])
                print feature + ' diff:    ' + str(abs_errors)

        # 2.2. Custom features
        # Calculate errors in waveforms
        if 'prestim_waveform_diff' in self.custom_feature_names:
            prestim_waveform_opt = self.prestimulusWaveformDifference(optimizable_traces, flat_target_trace=True)
            prestim_waveform_target = self.prestimulusWaveformDifference(self.target_traces, flat_target_trace=True)
            prestim_acceptable = 0.05 * prestim_waveform_target
            prestim_waveform_diff = abs(prestim_waveform_opt - prestim_waveform_target)
            feature_errors.append(prestim_waveform_diff/prestim_acceptable)
            if verbose is True:
                print '---'
                print 'prestim_waveform current: ' + str(prestim_waveform_opt)
                print 'prestim_waveform target: ' + str(prestim_waveform_target)
                print 'prestim_waveform diff: ' + str(prestim_waveform_diff)

            # Obsolete as of 2018-01-26
            # prestim_waveform_diff = self.prestimulusWaveformDifference(optimizable_traces)
            # feature_errors.append(prestim_waveform_diff)
            # if verbose is True:
            #     print '---'
            #     print 'prestim_waveform_diff current: ' + str(prestim_waveform_diff)


        if 'prespike_waveform_diff' in self.custom_feature_names:
            assert 'inv_time_to_first_spike' in self.efel_feature_names, \
                "Cannot compute prespike waveform difference unless spike latencies are extracted"
            individual_latencies = [individual_values[i]['inv_time_to_first_spike'] for i in range(0, n_steps - 1)]
            target_latencies = [self.target_values[i]['inv_time_to_first_spike'] for i in range(0, n_steps - 1)]

            # If no spikes with some current, integrate up to target's 1st spike
            for i in range(0, n_steps - 1):
                if individual_latencies[i] == self.TIME_INFTY:
                    individual_latencies[i] = target_latencies[i]

            individual_areas = self.prespikeWaveformArea(optimizable_traces, individual_latencies)
            target_areas = self.prespikeWaveformArea(self.target_traces, target_latencies)

            abs_errors = [abs(individual_areas[i] - target_areas[i]) for i in
                          range(0, n_steps - 1)]

            acceptable_errors = [0.05 * abs(target_areas[i]) for i in range(0, n_steps - 1)]
            scaled_abs_errors = [abs_errors[i] / acceptable_errors[i] for i in range(0, n_steps - 1)]
            avg_error = mean(scaled_abs_errors)

            feature_errors.append(avg_error)
            if verbose is True:
                print '---'
                print 'Prespike waveform current: ' + str(individual_areas)
                print 'Prespike waveform target:  ' + str(target_areas)
                print 'Prespike waveform diff:    ' + str(abs_errors)

        if 'rheobase' in self.custom_feature_names:
            assert 'rheobase_target' in self.adex_passive_params, \
            "Cannot estimate rheobase error unless rheobase target value is given"

            target_rheo = self.adex_passive_params['rheobase_target']
            rheo_acceptable_error = 0.05*target_rheo
            individual_rheo = self.plotFICurve(individual, only_rheobase=True)
            scaled_rheo_error = abs(target_rheo - individual_rheo) / rheo_acceptable_error

            feature_errors.append(scaled_rheo_error)


            # Obsolete as of 2018-01-26
            # individual_latencies = [individual_values[i]['inv_time_to_first_spike'] for i in range(0, n_steps-1)]
            # areas_btw_traces = self.prespikeWaveformDifference(optimizable_traces, individual_latencies)
            # avg_error = mean(areas_btw_traces)
            # feature_errors.append(avg_error)
            # if verbose is True:
            #     print '---'
            #     print 'prespike_waveform_diff current: ' + str(areas_btw_traces)

        if verbose is True:
            print 'Mean errors:'
            print self.efel_feature_names
            print self.custom_feature_names
            print feature_errors

        # 3. PLOT TRACES & RETURN ERRORS
        if plot_traces is True:
            fig, ax = plt.subplots(1, 4)
            self.target.plotTraces(optimizable_traces, ax=ax[:3])
            self.plotFICurve(individual, ax=ax[3])
            plt.show()

        return feature_errors


if __name__ == '__main__':
    current_steps = [-0.052612, 0.1413048, 0.1530802, 0.1648556]

    test_target = MarkramStepInjectionTraces('bbp_traces/L4_SS_cADpyr230_1/hoc_recordings/',
                                                      'soma_voltage_step', current_steps)
    passive_params = {'C': 110 * pF, 'gL': 3.1 * nS, 'EL': -70 * mV,
                      'VT': -42 * mV, 'DeltaT': 4 * mV,
                      'Vcut': 20 * mV, 'refr_time': 4 * ms}

    adex_neuron = AdexOptimizable(passive_params, test_target,
                                  efel_feature_names=['Spikecount_stimint', 'inv_time_to_first_spike', 'inv_first_ISI', 'inv_last_ISI', 'min_voltage_between_spikes'],
                                  custom_feature_names=['prestim_waveform_diff', 'prespike_waveform_diff'],
                                  dendritic_extent=0)

    # adex_neuron = AdexOptimizable(passive_params, test_target,
    #                               ['Spikecount_stimint'])

    init_guess = [0.9748883755,	281.47013649,	35.8342068769,	-68.1128513402,	-50.0230825328,	2.1430450182]









    adex_neuron.evaluateFitness(init_guess, plot_traces=True, verbose=True)
    # adex_neuron.plotFICurve(init_guess)

    # Example use of classes
    # current_steps = [-0.037109, 0.1291404, 0.1399021, 0.1506638]
    # test_target = MarkramStepInjectionTraces('L5_MC_bAC217_1/hoc_recordings/', 'soma_voltage_step', current_steps)
    #
    # # MC_params_Heikkinen = {'C': 92.1*pF, 'gL': 4.2*nS, 'VT': -42.29*mV, 'DeltaT': 4*mV,
    # #                        'Vcut': 20*mV, 'EL': -60.38*mV, 'refr_time': 4*ms}
    #
    # MC_params_Markram = {'C': 66.9*pF, 'gL': 3.04*nS, 'VT': -59*mV, 'DeltaT': 4*mV,
    #                      'Vcut': 20*mV, 'EL': -72.3*mV, 'refr_time': 4*ms}
    #
    # test_neuron = AdexOptimizable(MC_params_Markram, test_target,
    #                               ['Spikecount_stimint', 'inv_time_to_first_spike', 'inv_first_ISI',
    #                                'inv_last_ISI', 'AHP_depth_abs', 'AP_duration'])

    # Visualization of optimized parameters (after 100 generations); Heikkinen params
    # init_guess = [0.7199995715982088, 153.18939361105447, 62.88915388914671, -45.405472248324564]

    # # Visualization of optimized parameters (after 100 generations); Markram params
    # #init_guess = [1.2968038567964619, 181.3738415208683, 64.854393555649267, -62.763913778771048, -55.02246427009463, 3.6570056782860014]
    # # init_guess = [1.2734582399870464, 181.3738415208683, 64.854393555649267, -62.372383644267806, -55.02246427009463, 3.6925921422418688]
    # init_guess = [0.99660736178414533, 291.45151162166781, 51.773263344568818, -69.250301243212661, -58.934488767114964, 3.8898250852650285]
    #
    # test_neuron.evaluateFitness(init_guess, plot_traces=True, verbose=True)

