"""
Parameter fitting for adaptive exponential integrate-and-fire neurons
using traces from simulated BBP neurons
by Henri Hokkanen <henri.hokkanen@helsinki.fi>, January 2018

See adexfit_eval.py for cost function

This is the NON-PARALLEL version adapted from
http://efel.readthedocs.io/en/latest/deap_optimisation.html

Please use the parallel version for faster results
"""


from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy
import adexfit_eval
import brian2 as br2


# Initialize the neuron to be fitted
current_steps = [-0.037109, 0.1291404, 0.1399021, 0.1506638]
test_target = adexfit_eval.MarkramStepInjectionTraces('L5_MC_bAC217_1/hoc_recordings/', 'soma_voltage_step', current_steps)

MC_passive_params = {'C': 92.1 * br2.pF, 'gL': 4.2 * br2.nS, 'VT': -42.29 * br2.mV, 'DeltaT': 4 * br2.mV,
                     'Vcut': 20 * br2.mV, 'EL': -60.38 * br2.mV, 'refr_time': 4 * br2.ms}

test_neuron = adexfit_eval.AdexOptimizable(MC_passive_params, test_target)

# Set bounds for values (a, tau_w, b, V_res)
LOWER = [0, 0, 0, -70]
UPPER = [10, 300, 400, -40]


random.seed(1)

POP_SIZE = 100
OFFSPRING_SIZE = 100

NGEN = 300
ALPHA = POP_SIZE
MU = OFFSPRING_SIZE
LAMBDA = OFFSPRING_SIZE
CXPB = 0.7
MUTPB = 0.3
ETA = 10.0

SELECTOR = "NSGA2"

IND_SIZE = 4


creator.create("Fitness", base.Fitness, weights=[-1.0] * 4)
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

pop = toolbox.population(n=MU)

first_stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
second_stats = tools.Statistics(key=lambda ind: ind.fitness.values[1])
stats = tools.MultiStatistics(obj1=first_stats, obj2=second_stats)
stats.register("min", numpy.min, axis=0)


if __name__ == '__main__':
    pop, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        MU,
        LAMBDA,
        CXPB,
        MUTPB,
        NGEN,
        stats,
        halloffame=None, verbose=True)