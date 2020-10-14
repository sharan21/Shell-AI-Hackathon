""" miscellaneous functions """

import pandas as pd 
from pprint import pprint
import matplotlib.pyplot as plt 
import numpy as np
from Farm_Evaluator_Vec import getTurbLoc

path = './Shell_Hackathon Dataset/turbine_loc_test.csv'

def plot_farm_as_scatter(coords):

    x = [coord[0] for coord in coords]
    y = [coord[1] for coord in coords]

    plt.scatter(x, y) 

    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    
    plt.title('farm layout') 
    plt.show() 

def plot_farm_bit_map(farm_map):
    # farm map is np.array of shape (4000, 4000)
    pass




if __name__ == "__main__":

    coords = getTurbLoc(path)
    plot_farm_as_scatter(coords)
    


    

