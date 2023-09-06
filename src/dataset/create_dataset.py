import pandas as pd
import numpy as np
import json
import math
import tqdm
import os
import csv
import random

from getSolutionDetails import get_solution_details

def calculate_diferences(unique_solutions):
    features = [            "longitud",
                            "tipo del segmento",
                            "bateria restante",
                            "inclinacion",
                            "angulo de inclinacion",
                            "longitud restante",
                            "longitud restante ze",
                            "angulo de inclinacion medio absoluto restante"
    ]
    count_diff = []
    value_diff = []
    for index,seq in tqdm.tqdm(enumerate(unique_solutions)):
        count_diff.append([])
        value_diff.append([])
        count_diff[-1].append(0)
        index_2 = index + 1
        while index_2 < len(unique_solutions):
            count = 0
            value_diff[-1].append({})
            value_diff[-1][-1]["sequence_id"] = index_2
            value_diff[-1][-1]["values"] = []
            for i, tramo in enumerate(seq):
                value_diff[-1][-1]["values"].append({})

                for j, feature in enumerate(tramo):
                    if features[j] not in ["inclinacion", "angulo de inclinacion", "longitud restante", "angulo de inclinacion medio absoluto restante"]:
                        if i + 1 > len(unique_solutions[index_2]):
                            count += 1
                        else:
                            propiedad1 = feature
                            propiedad2 = unique_solutions[index_2][i][j]
                            if features[j] == "bateria restante":
                                propiedad1 = round(propiedad1, 3)
                                propiedad2 = round(propiedad2, 3)
                            if "longitud" in features[j]:
                                propiedad1 = round(propiedad1)
                                propiedad2 = round(propiedad2)
                            if propiedad1 != propiedad2:
                                count += 1
                                value_diff[-1][-1]["values"][-1][features[j]] = abs(propiedad1 - propiedad2)
                            else:
                                value_diff[-1][-1]["values"][-1][features[j]] = 0
            count_diff[-1].append(count)
            index_2 += 1
    with open("prueba.json", "w") as f:      
        json.dump(value_diff, f, indent = 4)
    with open("output.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(count_diff)               

def compare_solutions(solution, unique_solutions):
    result = 0
    features = [            "longitud",
                            "tipo del segmento",
                            "bateria restante",
                            "inclinacion",
                            "angulo de inclinacion",
                            "longitud restante",
                            "longitud restante ze",
                            "angulo de inclinacion medio absoluto restante"
    ]
    for index,seq in enumerate(unique_solutions):
        result = 1
        for i, tramo in enumerate(seq):
            for j, feature in enumerate(tramo):
                propiedad1 = feature
                propiedad2 = solution[i][j]
                if features[j] == "bateria restante":
                    propiedad1 = round(propiedad1, 3)
                    propiedad2 = round(propiedad2, 3)
                if "longitud" in features[j]:
                    propiedad1 = round(propiedad1)
                    propiedad2 = round(propiedad2)
                if propiedad1 != propiedad2:
                    '''print(f'Secuencia: {index}')
                    print(f'Tramo numero: {i}')
                    print(f'Feature: {features[j]}')
                    print(f"{feature},{solution[i][j]}")
                    print("------------")'''
                    result = 0
                    break
            if result == 0:
                break
    return result


def create_dataset(config_input_route):
    routes = ["Bus18", "BusM6", "Bus18_speed05", "BusM6_speed05", "Bus18_speed075", "BusM6_speed075"]
    scenarios = ["0%", "5%", "10%", "15%"]
    total_sequences = []

    segment_ids = []
    segment_lengths = []
    segment_types = []
    segment_batteries = []
    segment_remaining_lengths = []
    segment_remaining_ze_lengths = []
    segment_slopes = []
    segment_speed = []
    segment_slopes_angle = []
    segment_remaining_slope_angle = []
    sequence_ids = [] 
    sequence_cont = 0

    segment_labels = []
    patrones_intermedios = 0
    for route in tqdm.tqdm(routes):
        for scenario in scenarios:
            speed = ""
            splitted = route.split("_")
            if len(splitted) == 2:
                route = splitted[0]
                value = splitted[1].replace("speed", "").replace("0", "")
                speed = f"_speed_0.{value}"

            data = pd.read_csv("../route_info/processed_bus_route_{}_random_{}%ze{}.csv".format(route.replace("Bus",""), float(scenario.replace("%", "")), speed), index_col=0)
            
            if os.path.exists("{}/{}_Experiment/Hybrid_Bus{}_individuals.pf".format(config_input_route, route, scenario)):
                with open("{}/{}_Experiment/Hybrid_Bus{}_individuals.pf".format(config_input_route, route, scenario), "r") as filename:
                    for line in filename.readlines():
                        sequence = []
                        solution = line[line.find('"'):-1].replace('"', '').replace('[', '').replace(']', '').split(',')
                        solution = [int(sol) for sol in solution]
                        invalid , soc_values, remaining_charges, charge_per_zone, emissions_per_zone, green_kms_per_zone, solution = get_solution_details(solution,
                        "../route_info/processed_bus_route_{}_random_{}%ze.csv".format(route.replace("Bus",""), float(scenario.replace("%", ""))))
                        if not invalid:
                            solution_index = 0
                            for index, row in data.iterrows():
                                segment_ids.append("{}_{}{}_{}".format(index, route, speed.replace('_', '|'), scenario))
                                segment_lengths.append(data["Distance"][index])
                                segment_types.append(1 if data["Zone Type"][index] == 1 else -1)
                                segment_batteries.append(remaining_charges[index])
                                segment_speed.append(data["Avg Speed"][index])
                                segment_remaining_lengths.append(data["Distance"][index:-1].sum())
                                segment_slopes.append(data["Slope"][index])
                                segment_slopes_angle.append(data["Slope Angle"][index])

                                cont = index + 1
                                remaining_ze = 0
                                avg_slope_angle = 0
                                while cont < len(data.index):
                                    if data["Zone Type"][cont] == 1:
                                        remaining_ze += data["Distance"][cont]
                                    avg_slope_angle += abs(data["Slope Angle"][cont])
                                    cont += 1
                                if (cont - index - 1) == 0:
                                    segment_remaining_slope_angle.append(0)
                                else:
                                    segment_remaining_slope_angle.append(avg_slope_angle / (cont - index - 1))
                                segment_remaining_ze_lengths.append(remaining_ze)

                                if data["Zone Type"][index] == 1:
                                    segment_labels.append(1.0)
                                else:
                                    # Si tiene una inclinación menor a -2% se hace siempre en eléctrico
                                    if data["Slope"][index] > -0.02:
                                        if data["Bus Stop"][index] == 1 and solution[solution_index] <= 75:
                                            segment_labels.append(0.0)
                                        else:
                                            segment_labels.append(1.0 if solution[solution_index] >= 50 else 0.0)
                                        solution_index += 1
                                    else:
                                        segment_labels.append(1.0)
                                sequence_ids.append(sequence_cont)

                                sequence.append([segment_lengths[-1],
                                segment_types[-1],
                                segment_batteries[-1],
                                segment_speed[-1],
                                segment_slopes_angle[-1],
                                segment_remaining_lengths[-1],
                                segment_remaining_ze_lengths[-1],
                                segment_remaining_slope_angle[-1]])

                            sequence_cont += 1


    #print(segment_labels.count(1.0)/len(segment_labels))
    #print(segment_labels.count(-1.0)/len(segment_labels))
    print(len(total_sequences))
    print(len(np.unique(np.array(sequence_ids))))

    #calculate_diferences(total_sequences)
    dataset = []
    for index, _ in enumerate(segment_labels):
        dataset.append(pd.Series([
                sequence_ids[index],
                segment_ids[index],
                segment_lengths[index],
                segment_types[index],
                segment_batteries[index],
                segment_slopes_angle[index],
                segment_remaining_lengths[index],
                segment_remaining_ze_lengths[index],
                segment_remaining_slope_angle[index],
                segment_speed[index],
                segment_labels[index]
                ]
            )
        )
    
    dataset = pd.DataFrame(dataset)
    dataset.columns = ["sequence_id", "id", "length", "type", "remaining_charge", "slope_angle", "remaining_length", "remaining_ze_length",  "remaining_slope_angle", "speed", "result"]
    dataset.to_csv("iSUN_segments_dataset.csv")

def normalize_dataset(dataset):

    for feature in ["length", "remaining_charge" , "speed", "slope_angle", "remaining_length", "remaining_ze_length", "remaining_slope_angle",
                    "before_avg_slope", "before_avg_slope_angle", "before_up_distance", "before_down_distance",
                    "after_avg_slope", "after_avg_slope_angle", "after_up_distance", "after_down_distance"]:
        data_to_normalize = np.array(dataset[feature])

        mean =  np.mean(data_to_normalize)
        std =   np.std(data_to_normalize)

        data_to_normalize -= mean
        data_to_normalize /= std
            
        dataset[feature] = data_to_normalize
    
    return dataset


def get_context(id, n_before, n_after):
    value = id.split("_")[1]
    if "|" in value:
        route = value.split("|")[0]
        value = value.split("|")[1] + "_" + value.split("|")[2]
        speed = f"_{value}"
    else:
        route = value
        speed = ""
    data = pd.read_csv("../route_info/processed_bus_route_{}_random_{}%ze{}.csv".format(route.replace("Bus",""), float(id.split("_")[2].replace("%", "")), speed), index_col=0)

    id = int(id.split("_")[0])

    index_before = id -1 
    
    before_avg_slope = 0
    before_avg_slope_angle = 0
    before_up_distance = 0
    before_down_distance = 0

    while index_before >= 0 and index_before >= id - n_before:
        before_avg_slope += abs(data["Slope"][index_before])
        before_avg_slope_angle += abs(data["Slope Angle"][index_before])
        if data["Slope Angle"][index_before] > 0.0:
            before_up_distance += data["Distance"][index_before]
        elif data["Slope Angle"][index_before] <= -0.0:
            before_down_distance += data["Distance"][index_before]
        index_before -= 1
    
    before_avg_slope /= n_before
    before_avg_slope_angle /= n_before
    
    index_after = id + 1 
    
    after_avg_slope = 0
    after_avg_slope_angle = 0
    after_up_distance = 0
    after_down_distance = 0

    while index_after < len(data.index) and index_after <= id + n_after:
        after_avg_slope += abs(data["Slope"][index_after])
        after_avg_slope_angle += abs(data["Slope Angle"][index_after])
        if data["Slope Angle"][index_after] >= 0.0:
            after_up_distance += data["Distance"][index_after]
        elif data["Slope Angle"][index_after] <= -0.0:
            after_down_distance += data["Distance"][index_after]
        index_after += 1
    
    after_avg_slope /= n_after
    after_avg_slope_angle /= n_after

    '''
    print(before_avg_slope,
        before_avg_slope_angle,
        before_up_distance,
        before_down_distance
    )
    print("-----")
    print(after_avg_slope,
        after_avg_slope_angle,
        after_up_distance,
        after_down_distance
    )
    '''

    return [before_avg_slope, before_avg_slope_angle, before_up_distance, before_down_distance],\
        [after_avg_slope, after_avg_slope_angle, after_up_distance, after_down_distance]

def get_all_context(dataset, n_before, n_after):
    before_avg_slope = []
    before_avg_slope_angle = []
    before_up_distance = []
    before_down_distance = []

    after_avg_slope = []
    after_avg_slope_angle = [] 
    after_up_distance = []
    after_down_distance = []

    for index, row in dataset.iterrows():
        before, after = get_context(row["id"], n_before, n_after)

        before_avg_slope.append(before[0])
        before_avg_slope_angle.append(before[1])
        before_up_distance.append(before[2])
        before_down_distance.append(before[3])

        after_avg_slope.append(after[0])
        after_avg_slope_angle.append(after[1])
        after_up_distance.append(after[2])
        after_down_distance.append(after[3])
    
    result = dataset["result"]
    dataset = dataset.drop("result", axis=1)

    dataset["before_avg_slope"] = before_avg_slope
    dataset["before_avg_slope_angle"] = before_avg_slope_angle
    dataset["before_up_distance"] = before_up_distance
    dataset["before_down_distance"] = before_down_distance

    dataset["after_avg_slope"] = after_avg_slope
    dataset["after_avg_slope_angle"] = after_avg_slope_angle
    dataset["after_up_distance"] = after_up_distance
    dataset["after_down_distance"] = after_down_distance

    dataset["result"] = result

    return dataset

def split_dataset():
    dataset = pd.read_csv("iSUN_segments_dataset.csv", index_col=0)
    sequence_ids = dataset["sequence_id"]

    x = []
    dataset = dataset.to_numpy()
    previous_id = sequence_ids[0]
    x.append([])
    for index, value in enumerate(dataset):
        if sequence_ids[index] != previous_id:
            previous_id = sequence_ids[index]
            x.append([])
        x[sequence_ids[index]].append(np.array(value))
        

    x = np.array(x)
    sequence_ids = np.unique(np.array(sequence_ids))
    trn_size = 0.8
    split_index = round(len(sequence_ids) * trn_size)
    
    np.random.shuffle(sequence_ids)

    sqn_trn = sequence_ids[0:split_index]
    sqn_tst = sequence_ids[split_index + 1: -1]


    data_trn = x[sqn_trn]
    data_tst = x[sqn_tst]

    data_trn = [pd.Series(tramo) for seq in data_trn for tramo in seq]
    data_tst = [pd.Series(tramo) for seq in data_tst for tramo in seq]

    df_trn = pd.DataFrame(data_trn)
    df_trn.columns = ["sequence_id", "id", "length", "type", "remaining_charge", "slope_angle", "remaining_length", "remaining_ze_length",  "remaining_slope_angle", "speed", "result"]
    df_tst = pd.DataFrame(data_tst)
    df_tst.columns = ["sequence_id", "id", "length", "type", "remaining_charge", "slope_angle", "remaining_length", "remaining_ze_length",  "remaining_slope_angle", "speed", "result"]

    print(df_tst.head())
    df_trn.to_csv("trn_iSUN_segments_dataset.csv")
    df_tst.to_csv("tst_iSUN_segments_dataset.csv")
      
if __name__ == '__main__':
    #create_dataset(config_input_route='../results')
    split_dataset()
    #dataset = pd.read_csv("iSUN_segments_dataset.csv", index_col=0)
    #dataset = get_all_context(dataset, 10, 10)
    #dataset = normalize_dataset(dataset)