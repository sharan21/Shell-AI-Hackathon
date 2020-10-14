""" functions for train/test data generation, storage, processing and retrieval """

import pandas as pd 
from pprint import pprint
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import os
from Farm_Evaluator_Vec import getTurbLoc, checkConstraints, preProcessing, getAEP, loadPowerCurve, binWindResourceData
from tqdm import tqdm
from helpers import plot_farm_as_scatter

path = './Shell_Hackathon Dataset/turbine_loc_test.csv'
power_curve    =  loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')
wind_inst_freq =  binWindResourceData(r'./Shell_Hackathon Dataset/wind_data/wind_data_2007.csv')

def save_data(n_instances, inp, out, is_valid = True):

    file_offset = len(os.listdir('./datasets'))+1
    data = {"inp": inp, "out": out}

    if(is_valid):
        file_name = './datasets/valid_data_{}_{}'.format(n_instances, file_offset)
    else:
        file_name = './datasets/invalid_data_{}_{}'.format(n_instances, file_offset)

    save_path = open(file_name, 'wb') 
    pickle.dump(data, save_path)

    return file_name

def load_data(data_name):
    
    load_path = open(data_name, 'rb') 
    data = pickle.load(load_path)

    return data

def convert_cartesian_to_bit_map(coords, step_size = 1):
    # converts xy coords (50, 2) to bit map of shape (4000, 4000)

    coords_bm = np.zeros((4000, 4000), dtype=int) # resolution = 1 meter

    for coord in coords:
        coords_bm[coord[0]][coord[1]] = 1

    return coords_bm


def generate_invalid_data(no_of_instances, n_turbs = 50):

    inp = []
    out = []

    print("Creating {} invalid datapoints...".format(no_of_instances))

    for i in tqdm(range(no_of_instances)):

        xy_coords = np.random.uniform(low = 50, high = 3950, size = (n_turbs, 2))
        xy_coords = np.array(xy_coords, dtype=int)
                        
        if(checkConstraints(xy_coords, 100) == 0): #invalid coords
            
            bm_coords = convert_cartesian_to_bit_map(xy_coords) #(4000,4000)
            inp.append(bm_coords)
            out.append(0)   


    return np.array(inp), np.array(out) #converting to np.array takes time


def generate_valid_data(no_of_instances, n_turbs = 50):

    inp = []
    out = []

    print("Creating {} valid datapoints...".format(no_of_instances))

    for i in tqdm(range(no_of_instances)):

        count = 0
        xy_coords = []
        test_xy_coords = []

        while(count < n_turbs): #add sample points one by one making sure constraints not violated
            
            xy_coord = np.random.uniform(low = 50, high = 3950, size = (2, ))
            xy_coord = np.array(xy_coord, dtype=int)
            
            test_xy_coords.append(xy_coord)
                    
            if(checkConstraints(test_xy_coords, 100)):
                count += 1
                xy_coords.append(xy_coord)
            else:
                test_xy_coords.pop() 

    
        xy_coords = np.array(xy_coords) #(50,2)
        bm_coords = convert_cartesian_to_bit_map(xy_coords) #(4000,4000)
        
        inp.append(bm_coords)

        n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)
        AEP = getAEP(50, xy_coords, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t) 
        out.append(AEP)


    return np.array(inp),  np.array(out)  #(n_instances, 4000, 4000) & (n_instances,)


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def get_final_data(train_ratio = 0.75):

    paths = os.listdir('./datasets')

    inp_coll = []
    out_coll = []

    for path in paths:
        if(path == '.DS_Store' or path == 'processed_datasets'):
            continue

        data_here = load_data('./datasets/'+ path)
        inp_coll.extend(data_here['inp'])
        out_coll.extend(data_here['out'])

    shuffle_in_unison(inp_coll, out_coll)

    n_instances = len(inp_coll)

    x_train = np.array(inp_coll[0:int(train_ratio*n_instances)])
    x_test = np.array(inp_coll[int(train_ratio*n_instances) :])

    y_train = np.array(out_coll[0:int(train_ratio*n_instances)])
    y_test = np.array(out_coll[int(train_ratio*n_instances) :])

    return x_train, y_train, x_test, y_test
    

    

if __name__ == "__main__":

    """ Create Valid and Invalid data, Save to ./datasets """
    valid_inp, valid_out = generate_valid_data(no_of_instances = 5)
    invalid_inp, invalid_out = generate_invalid_data(no_of_instances = 5)
    
    
    save_data(len(valid_inp), valid_inp, valid_out, is_valid=True)
    save_data(len(invalid_inp), invalid_inp, invalid_out, is_valid=False)

    """ Create a dataset using all the data in ./datasets """

    x_train, y_train, x_test, y_test = get_final_data()

    print(x_train.shape) #(n_instances * train_ratio, n_turbines, 2)
    print(x_test.shape) #(n_instances * train_ratio, n_turbines, 2)
    

    

    

    

