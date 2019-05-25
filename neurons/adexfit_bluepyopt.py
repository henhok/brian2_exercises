import multiprocessing
from brian2 import *
prefs.codegen.target = 'cython'

import bluepyopt as bpop
from bluepyopt.parameters import Parameter

# def run_sim(parameters, input_current, dt=0.1*ms, runtime=3000*ms):
#

# class AdexOptimizable

# pool = multiprocessing.Pool()
# evaluator = AdexOptimizable(...)
# opt = bpop.deapext.optimisations.DEAPOptimisation(evaluator, map_function=pool.map)
# final_pop, hall_of_fame, logs, hist = opt.run(max_ngen=20)
# best_params = hall_of_fame[0]
# print('Top params: ', best_params)

# From adexfit_parallel
# efel_features = ['Spikecount_stimint', 'inv_time_to_first_spike', 'inv_first_ISI', 'inv_last_ISI', 'min_voltage_between_spikes']
# custom_features = ['prestim_waveform_diff', 'prespike_waveform_diff']
# test_neuron = adexeval.AdexOptimizable(passive_params, test_target, efel_feature_names=efel_features, custom_feature_names=custom_features, dendritic_extent=dendritic_extent)
# toolbox.register("evaluate", test_neuron.evaluateFitness)
# toolbox.register("map", pool.map)
# etc etc
print('hello world')