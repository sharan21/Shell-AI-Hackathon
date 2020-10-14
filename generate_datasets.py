import pandas as pd 
from pprint import pprint
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import os
from Farm_Evaluator_Vec import getTurbLoc, checkConstraints, preProcessing, getAEP, loadPowerCurve, binWindResourceData
from tqdm import tqdm

path = './Shell_Hackathon Dataset/turbine_loc_test.csv'
power_curve    =  loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')
wind_inst_freq =  binWindResourceData(r'./Shell_Hackathon Dataset/wind_data/wind_data_2007.csv')

def save_data(n_instances, inp, out):

    file_offset = len(os.listdir('./datasets'))+1
    data = {"inp": inp, "out": out}
    file_name = './datasets/data_{}_{}'.format(n_instances, file_offset)
    save_path = open(file_name, 'wb') 
    pickle.dump(data, save_path)

    return file_name

def load_data(data_name):
    
    load_path = open(data_name, 'rb') 
    data = pickle.load(load_path)

    return data

def generate_valid_inp(no_of_instances, n_turbs = 50):

    inp = []
    out = []

    print("Creating {} datapoints...".format(no_of_instances))

    for i in tqdm(range(no_of_instances)):

        count = 0
        coords = []
        test_coords = []

        while(count < n_turbs): #add sample points one by one making sure constraints not violated
            # print("count is {}".format(count))
            
            coord = np.random.uniform(low = 50, high = 3950, size = (2, ))
            coord = np.array(coord, dtype=int)
            
            test_coords.append(coord)
                    
            if(checkConstraints(test_coords, 100)):
                count += 1
                coords.append(coord)
            else:
                test_coords.pop() 

    
        coords = np.array(coords)
        inp.append(coords)
        n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)
        AEP = getAEP(50, coords, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t) 
        out.append(AEP)

    inp = np.array(inp)
    out = np.array(out)
    
    return inp, out


if __name__ == "__main__":

    inp, out = generate_valid_inp(no_of_instances = 10)
    
    pprint(inp.shape)
    pprint(out.shape)
    
    filename = save_data(len(inp), inp, out)
    data = load_data(filename)

    pprint(data)

