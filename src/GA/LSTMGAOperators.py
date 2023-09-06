import random
import copy
from typing import List


from jmetal.core.operator import Mutation
from jmetal.core.solution import BinarySolution, Solution, FloatSolution, IntegerSolution, PermutationSolution, \
    CompositeSolution
from jmetal.core.operator import Crossover
from jmetal.util.ckecking import Check
from jmetal.util.comparator import DominanceComparator
from jmetal.util.ckecking import Check


class IntegerSimpleRandomMutation(Mutation[IntegerSolution]):

    def __init__(self, probability: float):
        super(IntegerSimpleRandomMutation, self).__init__(probability=probability)

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        Check.that(type(solution) is IntegerSolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                solution.variables[i] = random.randrange(solution.lower_bound[i], solution.upper_bound[i]+1)
        return solution

    def get_name(self):
        return 'Integer Simple random_search mutation'

class IntegerUniformCrossover(Crossover[IntegerSolution, IntegerSolution]):

    def __init__(self, probability: float):
        super(IntegerUniformCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        Check.that(type(parents[0]) is IntegerSolution, "Solution type invalid")
        Check.that(type(parents[1]) is IntegerSolution, "Solution type invalid")
        Check.that(len(parents) == 2, 'The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()

        if rand <= self.probability:

            for i in range(offspring[0].number_of_variables):
                if random.random() < 0.50:
                    offspring[0].variables[i] = parents[0].variables[i]
                    offspring[1].variables[i] = parents[1].variables[i]

                else:
                    offspring[0].variables[i] = parents[1].variables[i]
                    offspring[1].variables[i] = parents[0].variables[i]

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return 'Integer Uniform crossover'
