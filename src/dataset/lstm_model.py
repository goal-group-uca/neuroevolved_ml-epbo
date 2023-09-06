
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from create_dataset import get_all_context, normalize_dataset



def compile_and_fit_model(xtrn, ytrn, xtst, ytst):
    # define a sequential model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Masking(mask_value=0.,
                                  input_shape=(timesteps, features)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(timesteps, input_shape=(timesteps,features), return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation='softmax')),
    ]) 

    #Compiling the network
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'] )

    #Fitting the data to the model
    history = model.fit(xtrn,
            ytrn,
            epochs=100,
            verbose= False,
            validation_data=(xtst, ytst))

    return model, history.history['accuracy'][-1]

def split_data(data, sequence_ids, train_index, test_index):
    data_trn = data[train_index]
    data_tst = data[test_index]

    xtrn = np.array([np.array(x)[:, 0:-1] for x in data_trn])
    ytrn = np.array([np.array(x)[:, -1] for x in data_trn])
    xtst = np.array([np.array(x)[:, 0:-1] for x in data_tst])
    ytst = np.array([np.array(x)[:, -1] for x in data_tst])


    return xtrn, ytrn, xtst, ytst

def redefine_problem(data, sequence_size):
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
    
    return np.array(new_data)



if __name__ == "__main__":
    dataset = pd.read_csv("trn_iSUN_segments_dataset.csv", index_col=0)
    dataset = get_all_context(dataset, 10, 10)
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
    
    x = np.array(x)
    print(len(x))
    x = redefine_problem(x, 3)
    print(len(x))

    x = tf.keras.preprocessing.sequence.pad_sequences(
        x, padding="post", dtype='float32'
    )
    batches, timesteps, features = len(x), len(x[0]) , len(x[0][0]) - 1
    print(x[0])

    sequence_ids = np.unique(np.array(sequence_ids))
    kf = KFold(n_splits=10)

    kfold_accuracy = []
    for i, (train_index, test_index) in enumerate(kf.split(sequence_ids)):
        xtrn, ytrn, xtst, ytst = split_data(x, sequence_ids, train_index, test_index)
        _, accuracy = compile_and_fit_model(xtrn, ytrn, xtst, ytst)
        kfold_accuracy.append(accuracy)
    
    print("Accuracy: {}".format(np.mean(np.array(kfold_accuracy))))


    #result = model.predict(np.array([xtst[0]]))
    #print(np.argmax(result[0],axis=1))
    
    #for index, x in enumerate(np.argmax(result[0],axis=1)):
    #    print("{} - {}".format(x, ytst[0][index]))

