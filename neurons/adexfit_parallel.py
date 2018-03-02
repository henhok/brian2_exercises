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
from brian2 import pF, nS, mV, ms, pA
import pathos.multiprocessing as mp
import pandas as pd
import logging
import sys

import adexfit_eval2 as adexeval

# TODO - Count spikes from spike monitor, not with eFel; get minimum vm between spikes from Brian2, not with eFel

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

output_csv_file = 'New_L4SS'
current_steps = [-0.052612,	0.1413048,	0.1530802,	0.1648556]

test_target = adexeval.MarkramStepInjectionTraces('bbp_traces/L4_SS_cADpyr230_1/hoc_recordings/',
                                                  'soma_voltage_step', current_steps)
passive_params = {'C': 110 * pF, 'gL': 3.1 * nS, 'EL': -70 * mV,
                  'VT': -42 * mV, 'DeltaT': 4 * mV,
                  'Vcut': 20 * mV, 'refr_time': 4 * ms}
dendritic_extent = 0

efel_features = ['Spikecount_stimint', 'inv_time_to_first_spike', 'inv_first_ISI', 'inv_last_ISI', 'min_voltage_between_spikes']
custom_features = ['prestim_waveform_diff', 'prespike_waveform_diff']

#N_features = len(feature_names)
# feature_weights = [-0.5, -0.1, -0.1, -0.1,
#                    -1, -0.01, -0.1]
# feature_weights = [-0.5, -1, -5, -0.1, -1,
#                    -5, -1]
feature_weights = [-1, -1, -1, -1, -1,
                   -1, -1]

test_neuron = adexeval.AdexOptimizable(passive_params, test_target, efel_feature_names=efel_features, custom_feature_names=custom_features, dendritic_extent=dendritic_extent)

IND_SIZE = 6

# Set bounds for values (a, tau_w, b, V_res)
# IND_SIZE = 4  # individual's size = number of AdEx parameters
# bounds = np.array([[-10, 10], [0, 300], [0, 400], [-70, -50]])
# IND_SIZE = 6
# bounds = np.array([[-5, 5], [0, 300], [0, 400], [-70, -40], [-60, -40], [0.2, 4]])

assert IND_SIZE in [4, 6], "IND_SIZE must be either 4 or 6!"

# Default IND_SIZE = 4 (a, tau_w, b, V_res)
adex_param_names = ['a', 'tau_w', 'b', 'V_res']
bounds = [[-10, 10],
          [0, 300],
          [0, 400],
          [-70, -60]]

if IND_SIZE == 6:  # if also VT, DeltaT are included
    adex_param_names += ['VT', 'DeltaT']
    bounds.append([-70, -50])
    bounds.append([1, 4])

bounds = np.array(bounds)

# Set optimization parameters here
NGEN = 20
POP_SIZE = 1000
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
# toolbox.decorate("evaluate", tools.DeltaPenalty(test_neuron.isFeasible, 10000))

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

allindiv = dict()
allfitness = dict()
gen_counter = 0
generation_stats = dict()


def saveGen(gen_individuals):
    global gen_counter
    fitness_sums = []

    for indiv in gen_individuals:
        allindiv[indiv.history_index] = [gen_counter] + list(indiv)
        allfitness[indiv.history_index] = indiv.fitness.values
        fitness_sums.append(np.sum(indiv.fitness.values))

    generation_stats[gen_counter] = [np.mean(fitness_sums), np.std(fitness_sums), np.min(fitness_sums)]
    gen_counter += 1

    return 'ok!'


# FOLLOWING RUN ONLY BY ROOT PROCESS
if __name__ == '__main__':

    random.seed()

    N_CPU = int(mp.cpu_count()*0.80)
    pool = mp.Pool(processes=N_CPU)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=MU)
    opthistory.update(pop)

    hof = tools.HallOfFame(N_HALLOFFAME)

    # Working stats generator
    # stats01 = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    # stats02 = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    # stats03 = tools.Statistics(key=lambda ind: ind.fitness.values[2])
    # stats04 = tools.Statistics(key=lambda ind: ind.fitness.values[3])
    # stats05 = tools.Statistics(key=lambda ind: ind.fitness.values[4])
    # stats = tools.MultiStatistics(spikecount=stats01,
    #                               first_spike_latency=stats02,
    #                               first_isi=stats03,
    #                               last_isi=stats04, min_volt=stats05)
    # stats.register("min", np.min, axis=0)

    # Save aggregate data on each generation for immediate viewing
    stats_gen = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats_gen.register("avg", np.mean, axis=0)
    stats_gen.register("std", np.std, axis=0)
    stats_gen.register("min", np.min, axis=0)
    stats_gen.register("max", np.max, axis=0)

    # Save every individual with its history_index & fitness values
    stats_save = tools.Statistics(key=lambda ind: ind)
    stats_save.register("saved", saveGen)

    stats = tools.MultiStatistics(stats_gen=stats_gen, stats_save=stats_save)

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


    # print ''
    # print '================='
    # print 'TOP %d PARAMETERS' % N_HALLOFFAME
    # print '=================\n'
    # i = 1
    # for params in hof:
    #     print "%d. a = %.3f nS\t tau_w = %.3f ms\t b = %.3f pA\t\t V_res = %.3f mV" % (i, params[0], params[1], params[2], params[3])
    #     i += 1
    #
    # i = 1
    # for params in hof:
    #     print "%d. %s" % (i, str(params))
    #     i += 1

    print 'Saving genealogy'

    # Save all neurons (search space)
    df_fitness = pd.DataFrame.from_dict(allfitness, orient='index')
    df_fitness.columns = efel_features + custom_features
    df_fitness_sum = pd.DataFrame(df_fitness.sum(axis=1))
    df_fitness_sum.columns = ['fitness_sum']

    df_indiv = pd.DataFrame.from_dict(allindiv, orient='index')
    df_indiv.columns = ['generation'] + adex_param_names

    df_final = pd.concat([df_indiv, df_fitness, df_fitness_sum], axis='columns')

    df_final.to_csv(output_csv_file+'_allneurons.csv')

    # Save stats about generations
    featurestats = logbook.chapters['stats_gen']
    fts = pd.DataFrame(featurestats)
    df_statslist = []
    all_features = efel_features + custom_features
    for i in range(len(all_features)):
        tmp_df = pd.DataFrame([fts['avg'].str[i], fts['std'].str[i], fts['min'].str[i]]).T
        tmp_df.columns = [all_features[i]+'_mean', all_features[i]+'_std', all_features[i]+'_min']
        df_statslist.append(tmp_df)

    df_overallstats = pd.DataFrame.from_dict(generation_stats, orient='index')
    df_overallstats.columns = ['fitness_mean', 'fitness_std', 'fitness_min']
    df_statslist.append(df_overallstats)

    df_genfinal = pd.concat(df_statslist, axis='columns')

    df_genfinal.to_csv(output_csv_file + '_genstats.csv')

    print 'Finished!'