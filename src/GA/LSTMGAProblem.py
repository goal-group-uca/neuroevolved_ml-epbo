from jmetal.core.problem import Problem
from jmetal.core.solution import IntegerSolution
from sklearn.model_selection import KFold
from create_dataset import get_all_context, normalize_dataset

import random
import pandas as pd
import numpy as np
import time
import tensorflow as tf

class LSTMProblem(Problem):
    def __init__(self):
        super(LSTMProblem, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = 4
        self.number_of_constraints = 0

        self.initial_solution = True

        self.lower_bounds = [1, 1, 150, 1]
        self.upper_bounds = [20, 20, 250, 20]

        self.kf = KFold(n_splits=10)

        self.original_data = pd.read_csv("../dataset/trn_iSUN_segments_dataset.csv", index_col=0)


        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["Accuracy"]

    def create_solution(self) -> IntegerSolution:


        new_solution = IntegerSolution(number_of_objectives=self.number_of_objectives,
                                        lower_bound= self.lower_bounds, upper_bound= self.upper_bounds)

        new_solution.variables = \
        [random.randint(new_solution.lower_bound[0], new_solution.upper_bound[0]),
        random.randint(new_solution.lower_bound[1], new_solution.upper_bound[1]),
        random.randint(new_solution.lower_bound[2], new_solution.upper_bound[2]),
        random.randint(new_solution.lower_bound[3], new_solution.upper_bound[3])
        ]

        return new_solution

    def get_name(self) -> str:
      return 'LSTM Neuroevolution Problem'

    def redefine_problem(self, data, sequence_size):
        new_data = []
        for seq in data:
            cont = 0
            new_data.append([])
            for tramo in seq:
                new_data[-1].append(np.array(tramo))
                cont += 1
                if cont == sequence_size:
                    cont = 0
                    new_data.append([])
    
        return np.array(new_data, dtype=object)
    
    def prepare_data(self, after_context, before_context):
        dataset = self.original_data.copy()

        dataset = get_all_context(dataset, before_context, after_context)
        dataset = normalize_dataset(dataset)
        dataset = dataset.drop('id', axis=1)

        sequence_ids = dataset["sequence_id"]
        dataset = dataset.drop('sequence_id', axis=1).to_numpy()

        x = []
        previous_id = sequence_ids[0]
        cont = 0 
        x.append([])
        for index, value in enumerate(dataset):
            if sequence_ids[index] != previous_id:
                previous_id = sequence_ids[index]
                cont += 1
                x.append([])
            x[cont].append(np.array(value))
        
        x = np.array(x, dtype=object)

        x = tf.keras.preprocessing.sequence.pad_sequences(
            x, padding="post", dtype='float32'
        )

        return x, np.unique(np.array(sequence_ids))

    def split_data(self, data, sequence_ids, train_index, test_index):
        data_trn = data[train_index]
        data_tst = data[test_index]

        xtrn = np.array([np.array(x)[:, 0:-1] for x in data_trn])
        ytrn = np.array([np.array(x)[:, -1] for x in data_trn])
        xtst = np.array([np.array(x)[:, 0:-1] for x in data_tst])
        ytst = np.array([np.array(x)[:, -1] for x in data_tst])


        return xtrn, ytrn, xtst, ytst
    
    def compile_and_fit_model(self, xtrn, ytrn, xtst, ytst, timesteps,
                                 features, epochs, learning_rate):
        # define a sequential model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Masking(mask_value=0.,
                                    input_shape=(timesteps, features)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(timesteps, input_shape=(timesteps,features), return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation='softmax')),
        ]) 

        #Compiling the network
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=['accuracy'] )

        #Fitting the data to the model
        history = model.fit(xtrn,
                ytrn,
                epochs=epochs,
                verbose= False,
                validation_data=(xtst, ytst))

        return model, history.history['val_accuracy'][-1]

    def evaluate(self, solution):
        before_context, after_context, epochs, learning_rate =\
            solution.variables[0], solution.variables[1], solution.variables[2],  solution.variables[3] * 0.001
        
        x, sequence_ids = self.prepare_data(after_context, before_context)

        _, timesteps, features = len(x), len(x[0]) , len(x[0][0]) - 1

        kfold_accuracy = []
        for i, (train_index, test_index) in enumerate(self.kf.split(sequence_ids)):
            xtrn, ytrn, xtst, ytst = self.split_data(x, sequence_ids, train_index, test_index)
            _, accuracy = self.compile_and_fit_model(xtrn, ytrn, xtst, ytst, timesteps,
                                                        features, epochs, learning_rate)
            kfold_accuracy.append(accuracy)
        
        solution.objectives[0] = - np.mean(np.array(kfold_accuracy))

        return solution

        
