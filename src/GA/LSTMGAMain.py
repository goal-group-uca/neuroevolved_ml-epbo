
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

from LSTMGAOperators import IntegerSimpleRandomMutation, IntegerUniformCrossover
from LSTMGAProblem import LSTMProblem

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.util.termination_criterion import StoppingByConvergence
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.observer import ProgressBarObserver, WriteFrontToFileObserver

import pandas as pd
import logging



config_n_processes = 2 # Number of processes used to split the population
config_epochs = 20 # Minimum number of epochs
config_population_size = 96 # Total population size of the GA

def main_ga(config_population_size, config_n_processes, config_epochs):
    LOGGER = logging.getLogger("jmetal")
    LOGGER.disabled = True

    config_offspring_population_size = config_population_size 
    config_probability_mutation = 1. / 5. # Mutation probability for the GA
    config_probability_crossover = 0.9 # Crossover probability for the GA
    config_crossover_operator = IntegerUniformCrossover(config_probability_crossover)
    config_mutation_operator = IntegerSimpleRandomMutation(config_probability_mutation)
    config_selection_operator = BinaryTournamentSelection()

    # Problem set
    problem = LSTMProblem()

    algorithm = GeneticAlgorithm(problem = problem,
        population_size = config_population_size,
        offspring_population_size=config_offspring_population_size,
        population_evaluator = MultiprocessEvaluator(processes = config_n_processes),
        selection=config_selection_operator,
        mutation = config_mutation_operator,
        crossover= config_crossover_operator,
        termination_criterion = StoppingByConvergence(max_dif = 0, last_n = 5, min_epochs = config_epochs))

    # Setting algorithm observers
    algorithm.observable.register(WriteFrontToFileObserver("./generation_front_files"))
    #algorithm.observable.register(NonDomWriteFrontToFileObserver("./generation_nondom_front_files"))
    #algorithm.observable.register(NonDomPlotFrontToFileObserver("./generation_nondom_front_plots"))
    algorithm.observable.register(ProgressBarObserver(max = config_epochs * config_population_size))


    # Run genetic algorithm
    algorithm.run()

    with open("results.data","w") as file:
        #Outputs
        file.write('\nSettings:')
        file.write(f'\n\tAlgorithm: {algorithm.get_name()}')
        file.write(f'\n\tProblem: {problem.get_name()}')
        file.write(f'\n\tComputing time: {algorithm.total_computing_time} seconds')
        file.write(f'\n\tMin evaluations: {config_epochs * config_population_size}')
        file.write(f'\n\tPopulation size: {config_population_size}')
        file.write(f'\n\tOffspring population size: {config_offspring_population_size}')
        file.write(f'\n\tProbability mutation: {config_probability_mutation}')
        file.write(f'\n\tProbability crossover: {config_probability_crossover}')
        file.write('\nResults:')
        solution=algorithm.get_result()
        file.write(f'\n\tBest solution: {solution.variables}')
        file.write(f'\n\tFitness: [{solution.objectives[0]}]')

    print('\nSettings:')
    print(f'\tAlgorithm: {algorithm.get_name()}')
    print(f'\tProblem: {problem.get_name()}')
    print(f'\tComputing time: {algorithm.total_computing_time} seconds')
    print(f'\tMin evaluations: {config_epochs * config_population_size}')
    print(f'\tPopulation size: {config_population_size}')
    print(f'\tOffspring population size: {config_offspring_population_size}')
    print(f'\tProbability mutation: {config_probability_mutation}')
    print(f'\tProbability crossover: {config_probability_crossover}')
    print('\nResults:')
    solution=algorithm.get_result()
    print(f'\tBest solution: {solution.variables}')
    print(f'\tFitness: [{solution.objectives[0]}]')




if __name__ == '__main__':
    main_ga(config_population_size, config_n_processes, config_epochs)

