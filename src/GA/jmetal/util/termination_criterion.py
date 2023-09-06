import threading
from abc import ABC, abstractmethod

from jmetal.core.observer import Observer
from jmetal.core.quality_indicator import QualityIndicator

"""
.. module:: termination_criterion
   :platform: Unix, Windows
   :synopsis: Implementation of stopping conditions.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class TerminationCriterion(Observer, ABC):

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def is_met(self):
        pass


class StoppingByEvaluations(TerminationCriterion):

    def __init__(self, max_evaluations: int):
        super(StoppingByEvaluations, self).__init__()
        self.max_evaluations = max_evaluations
        self.evaluations = 0

    def update(self, *args, **kwargs):
        self.evaluations = kwargs['EVALUATIONS']

    @property
    def is_met(self):
        return self.evaluations >= self.max_evaluations


class StoppingByConvergence(TerminationCriterion):

    def __init__(self, max_dif: float, last_n : int, min_epochs : int):
        super(StoppingByConvergence, self).__init__()

        print("Convergence Criterion")

        self.max_dif = max_dif
        self.min_epochs = min_epochs
        self.last_n = last_n

        if self.min_epochs < self.last_n:
            self.min_epochs = -1#self.last_n

        self.total_epochs = 0

        self.l_solutions = []

    def update(self, *args, **kwargs):
        self.evaluations = kwargs['EVALUATIONS']
        self.l_solutions.append(kwargs['SOLUTIONS'].objectives[0])
        print(self.l_solutions)
        self.total_epochs += 1
        while len(self.l_solutions) > self.last_n:
            self.l_solutions.pop(0)

    @property
    def is_met(self):
        print(self.total_epochs, self.min_epochs)
        if self.total_epochs <= self.min_epochs:
            print("Epoch: {}".format(self.total_epochs))
            return False
        else:
            print("##################")
            print(self.l_solutions[0])
            print(self.l_solutions[-1])
            print("####################")
            print("Epoch: {}, dif = {}".format(self.total_epochs,abs(self.l_solutions[0]-self.l_solutions[-1])))
            return abs(self.l_solutions[0]-self.l_solutions[-1]) <= self.max_dif


class StoppingByTime(TerminationCriterion):

    def __init__(self, max_seconds: int):
        super(StoppingByTime, self).__init__()
        self.max_seconds = max_seconds
        self.seconds = 0.0

    def update(self, *args, **kwargs):
        self.seconds = kwargs['COMPUTING_TIME']

    @property
    def is_met(self):
        return self.seconds >= self.max_seconds


def key_has_been_pressed(stopping_by_keyboard):
    input('PRESS ANY KEY + ENTER: ')
    stopping_by_keyboard.key_pressed = True


class StoppingByKeyboard(TerminationCriterion):

    def __init__(self):
        super(StoppingByKeyboard, self).__init__()
        self.key_pressed = False
        thread = threading.Thread(target=key_has_been_pressed, args=(self,))
        thread.start()

    def update(self, *args, **kwargs):
        pass

    @property
    def is_met(self):
        return self.key_pressed


class StoppingByQualityIndicator(TerminationCriterion):

    def __init__(self, quality_indicator: QualityIndicator, expected_value: float, degree: float):
        super(StoppingByQualityIndicator, self).__init__()
        self.quality_indicator = quality_indicator
        self.expected_value = expected_value
        self.degree = degree
        self.value = 0.0

    def update(self, *args, **kwargs):
        solutions = kwargs['SOLUTIONS']

        if solutions:
            self.value = self.quality_indicator.compute(solutions)

    @property
    def is_met(self):
        if self.quality_indicator.is_minimization:
            met = self.value * self.degree < self.expected_value
        else:
            met = self.value * self.degree > self.expected_value

        return met
