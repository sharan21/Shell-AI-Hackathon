import pandas as pd 
from pprint import pprint
import matplotlib.pyplot as plt 
import numpy as np
from Farm_Evaluator_Vec import getTurbLoc

path = './Shell_Hackathon Dataset/turbine_loc_test.csv'
# test = pd.read_csv(path)


def plot_farm_as_scatter(turbine_coords):

    x = [ele[0] for ele in turbine_coords]
    y = [ele[1] for ele in turbine_coords]

    plt.scatter(x, y) 

    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    
    plt.title('farm layout') 
    plt.show() 

def convert_coords_to_conv_inp(coords, step_size = 1):
    # takes the 50 turbine locations and plots them on a 2D plane (2d array)
    # assumes that the coords and the resultant map do not break any constraints

    farm_map = np.zeros((4000, 4000), dtype=int) #assume step size = 1 meter
    pprint(farm_map[0][0].shape)
    # exit(0)

    for i in range(len(coords)):
        print("setting index x: {}, y: {}".format(coords[i][0], coords[i][1]))

        farm_map[int(coords[i][0])][int(coords[i][1])] = 1

    # pprint(farm_map)
    

    return farm_map
    





if __name__ == "__main__":
    coords = getTurbLoc(path)
    # pprint(coords)
    convert_coords_to_conv_inp(coords)


    

