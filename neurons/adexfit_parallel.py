"""
Parameter fitting for adaptive exponential integrate-and-fire neurons
using traces from simulated BBP neurons
by Henri Hokkanen <henri.hokkanen@helsinki.fi>, January 2018

See adexfit_eval.py for cost function

This code adapted from
http://efel.readthedocs.io/en/latest/deap_optimisation.html

Parallel optimization made using
- Distributed Evolutionary Algorithms in Python (DEAP) library (http://deap.readthedocs.io)
- pathos parallelization library (https://github.com/uqfoundation/pathos) (due to pickling issues)
"""

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy as np
from brian2 import pF, nS, mV, ms
import pathos.multiprocessing as mp

import adexfit_eval2 as adexeval


# SET NEURON & STIMULATION PARAMETERS HERE
# Initialize the neuron to be fitted
# current_steps = [-0.037109, 0.1291404, 0.1399021, 0.1506638]
# test_target = adexfit_eval.MarkramStepInjectionTraces('L5_MC_bAC217_1/hoc_recordings/', 'soma_voltage_step', current_steps)
#
# MC_params_Markram = {'C': 66.9 * pF, 'gL': 3.04 * nS, 'VT': -59 * mV, 'DeltaT': 4 * mV,
#                      'Vcut': 20 * mV, 'EL': -72.3 * mV, 'refr_time': 4 * ms}

# feature_names = ['Spikecount_stimint', 'inv_time_to_first_spike', 'inv_first_ISI', 'inv_last_ISI', 'min_voltage_between_spikes']
# test_neuron = adexfit_eval.AdexOptimizable(MC_params_Markram, test_target, feature_names)

# Initialize the neuron to be fitted
current_steps = [-0.084106, 0.2073756, 0.2246569, 0.2419382]
test_target = adexeval.MarkramStepInjectionTraces('bbp_traces/L4_PC_cADpyr230_3/hoc_recordings/', 'soma_voltage_step', current_steps)
passive_params = {'C': 1 * pF, 'gL': 1 * nS, 'EL': -72 * mV,
                  'VT': -56 * mV, 'DeltaT': 4 * mV,
                  'Vcut': 20 * mV, 'refr_time': 4 * ms}

efel_features = ['Spikecount_stimint', 'inv_time_to_first_spike', 'inv_first_ISI', 'inv_last_ISI', 'min_voltage_between_spikes']
custom_features = ['prestim_waveform_diff', 'prespike_waveform_diff']

#N_features = len(feature_names)
# feature_weights = [-0.5, -0.1, -0.1, -0.1,
#                    -1, -0.01, -0.1]
feature_weights = [-0.5, -1, -5, -0.1, -1,
                   -5, -1]

test_neuron = adexeval.AdexOptimizable(passive_params, test_target, efel_feature_names=efel_features, custom_feature_names=custom_features, dendritic_extent=2)


# Set bounds for values (a, tau_w, b, V_res)
IND_SIZE = 4  # individual's size = number of AdEx parameters
bounds = np.array([[-10, 10], [0, 300], [0, 400], [-70, -50]])
# IND_SIZE = 6
# bounds = np.array([[-5, 5], [0, 300], [0, 400], [-70, -40], [-60, -40], [0.2, 4]])

# Set optimization parameters here
NGEN = 5
POP_SIZE = 25
OFFSPRING_SIZE = POP_SIZE
CXPB = 0.7   # crossover fraction
MUTPB = 0.3  # mutation frequency
N_HALLOFFAME = 10


# OPTIMIZATION ALGORITHM (probably no need to touch this)
ALPHA = POP_SIZE
MU = OFFSPRING_SIZE
LAMBDA = OFFSPRING_SIZE
ETA = 10.0

SELECTOR = "NSGA2"

LOWER = list(bounds[:, 0])
UPPER = list(bounds[:, 1])


creator.create("Fitness", base.Fitness, weights=feature_weights)
creator.create("Individual", list, fitness=creator.Fitness)


def uniform(lower_list, upper_list, dimensions):
    """Fill array """

    if hasattr(lower_list, '__iter__'):
        return [random.uniform(lower, upper) for lower, upper in
                zip(lower_list, upper_list)]
    else:
        return [random.uniform(lower_list, upper_list)
                for _ in range(dimensions)]


toolbox = base.Toolbox()
toolbox.register("uniformparams", uniform, LOWER, UPPER, IND_SIZE)
toolbox.register(
    "Individual",
    tools.initIterate,
    creator.Individual,
    toolbox.uniformparams)
toolbox.register("population", tools.initRepeat, list, toolbox.Individual)


toolbox.register("evaluate", test_neuron.evaluateFitness)

toolbox.register(
    "mate",
    tools.cxSimulatedBinaryBounded,
    eta=ETA,
    low=LOWER,
    up=UPPER)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=ETA,
                 low=LOWER, up=UPPER, indpb=0.1)

toolbox.register("variate", algorithms.varAnd)

toolbox.register(
    "select",
    tools.selNSGA2)

opthistory = tools.History()
toolbox.decorate("mate", opthistory.decorator)
toolbox.decorate("mutate", opthistory.decorator)




# FOLLOWING RUN ONLY BY ROOT PROCESS
if __name__ == '__main__':

    random.seed()
    N_CPU = int(mp.cpu_count()*0.80)
    pool = mp.Pool(processes=N_CPU)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=MU)
    opthistory.update(pop)
    hof = tools.HallOfFame(N_HALLOFFAME)

    # feature_names = ['Spikecount_stimint', 'inv_time_to_first_spike', 'inv_first_ISI', 'inv_last_ISI', 'AHP_depth_abs']
    stats01 = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats02 = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats03 = tools.Statistics(key=lambda ind: ind.fitness.values[2])
    stats04 = tools.Statistics(key=lambda ind: ind.fitness.values[3])
    stats05 = tools.Statistics(key=lambda ind: ind.fitness.values[4])
#    stats06 = tools.Statistics(key=lambda ind: ind.fitness.values[5])
#    stats07 = tools.Statistics(key=lambda ind: ind.fitness.values[6])
    stats = tools.MultiStatistics(spikecount=stats01,
                                  first_spike_latency=stats02,
                                  first_isi=stats03,
                                  last_isi=stats04, min_volt=stats05)
#                                  prestim=stats06, prespike=stats07)

    stats.register("min", np.min, axis=0)
    # stats.register("avg", np.mean, axis=0)

    print "Running optimization with %d cores... please wait.\n" % N_CPU
    pop, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        MU,
        LAMBDA,
        CXPB,
        MUTPB,
        NGEN,
        stats,
        halloffame=hof, verbose=True)

    pool.close()

    print ''
    print '================='
    print 'TOP %d PARAMETERS' % N_HALLOFFAME
    print '=================\n'
    i = 1
    for params in hof:
        print "%d. a = %.3f nS\t tau_w = %.3f ms\t b = %.3f pA\t\t V_res = %.3f mV" % (i, params[0], params[1], params[2], params[3])
        i += 1

    i = 1
    for params in hof:
        print "%d. %s" % (i, str(params))
        i += 1

    #print 'Saving genealogy'
    # take genealogy_history (dict) to pandas
    # calculate fitness for each