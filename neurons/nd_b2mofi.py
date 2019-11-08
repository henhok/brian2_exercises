from brian2.units import *
import brian2 as b2
import matplotlib.pyplot as plt
from traces import MarkramStepInjectionTraces
import sys
sys.path.append('/home/henhok/PycharmProjects/brian2modelfitting/')
import brian2modelfitting as mofi
sys.path.append('/home/henhok/PycharmProjects/neurodynlib/')
import neurodynlib as nd
import efel

if __name__ == '__main__':
    # Passive params for bbp_traces/L5_TTPC1_cADpyr232_4/hoc_recordings/
    # C = 110 * pF
    # #gL = 3.1 * nS
    # EL = -70 * mV
    # # VT = -42 * mV
    # # DeltaT = 4 * mV
    # Vcut = 20 * mV
    # refr_time = 4 * ms
    # equation_soma = '''
    # dvm/dt = (gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) -w + I_input) / C : volt (unless refractory)
    # dw/dt = (a*(vm-EL)-w)/tau_w : amp
    # I_input: amp
    # '''

    ### Get target traces
    current_steps = [-0.247559, 0.55425, 0.6004375, 0.646625]  # *nA
    test_target = MarkramStepInjectionTraces('bbp_traces/L5_TTPC1_cADpyr232_4/hoc_recordings/',
                                             'soma_voltage_step', current_steps)

    # test_target.plotTraces()
    # plt.show()

    ### Optimize
    # result, error = test_target.optimize_adex(n_rounds=10, n_samples=100, opt_method='DE')
    # Features-based fitting results...
    # CMA 10/30 -> 165; DE 10/30 -> 230
    # CMA 100/30 -> 82.5; DE 100/30 -> 82.5

    optimizer = mofi.NevergradOptimizer(method='DE', num_workers=4)
    traces_times = [[0 * ms, 3000 * ms]]
    feat_list = ['Spikecount_stimint',
                 'inv_time_to_first_spike', 'inv_first_ISI', 'inv_last_ISI',
                 'minimum_voltage']  # , 'min_voltage_between_spikes']
    feat_weights = {'Spikecount_stimint': (1 / 2),
                    'inv_time_to_first_spike': (1 / 2),
                    'inv_first_ISI': (1 / 5),
                    'inv_last_ISI': (1 / 5),
                    'minimum_voltage': (1 / 6)}
    efel.api.setThreshold(-35)  # Lowering threshold so that eFel recognizes EIF/AdEx spikes

    metric = mofi.FeatureMetric(traces_times, feat_list, weights=feat_weights, combine=None)
    # metric = MSEMetric()

    inp_traces = test_target.input_traces * amp
    out_traces = test_target.vm_traces * volt

    # Neurodynlib begins
    neuron = nd.LifAscNeuron()
    neuron.add_external_current('I_input')
    eqs = neuron.get_neuron_equations()
    params = neuron.get_neuron_parameters()
    # Neurodynlib ends

    n_samples = 100
    n_rounds = 4

    fitter = mofi.TraceFitter(model=eqs,
                              input=inp_traces,
                              output=out_traces,
                              input_var='I_input',
                              output_var='vm',
                              reset=neuron.reset_statements,
                              refractory=params['refractory_period'],
                              threshold=neuron.threshold_condition,
                              method='exponential_euler',
                              dt=0.1 * ms,
                              n_samples=n_samples,
                              param_init=neuron.get_initial_values())

    fitter.setup_neuron_group(fitter.n_neurons, params)

    result, error = fitter.fit(optimizer=optimizer,
                               metric=metric,
                               n_rounds=n_rounds,
                               A_asc1=[10 * pA, 100 * pA]
                               )

    # return result, error


    ### Plot the results
    # a = result['a'] * siemens
    # tau_w = result['tau_w'] * second
    # b = result['b'] * amp
    # Vres = result['Vres'] * volt
    # VT = result['VT'] * volt
    # DeltaT = result['DeltaT'] * volt
    # # C = result['C'] * farad
    # gL = result['gL'] * siemens
    # # EL = result['EL'] * volt
    #
    #
    # G = NeuronGroup(1, equation_soma, threshold='vm > Vcut',
    #                 reset='vm = Vres; w=w+b',
    #                 refractory=refr_time, method='exponential_euler')
    # G.vm = EL
    #
    # statemon = StateMonitor(G, ['vm'], record=True)
    #
    # depol_start = 700*ms
    # depol_end = 2700*ms
    # stim_total = 3000*ms
    # n_steps = 4
    #
    # # Run the current injections
    # # Hyperpolarizing current
    # G.I_input = current_steps[0] * nA
    # run(depol_start)
    #
    # # Depolarizing steps
    # # for step in range(1, n_steps):
    # #     G.I_depol[step] = current_steps[step] * nA
    # # run(depol_end - depol_start)
    # G.I_input += current_steps[3] * nA
    # run(depol_end - depol_start)
    #
    # # Stimulation ends
    # G.I_input -= current_steps[3] * nA
    # run(stim_total - depol_end)
    #
    # brian_plot(statemon)
    # plt.show()
