import pandas as pd 
from pprint import pprint
import matplotlib.pyplot as plt 


# path = './Shell_Hackathon Dataset/turbine_loc_test_2.csv'
# test = pd.read_csv(path)

# x_coord = list(test.x)
# y_coord = list(test.y)

# # plotting the points  
# plt.scatter(x_coord, y_coord) 
  
# # naming the x axis 
# plt.xlabel('x - axis') 
# # naming the y axis 
# plt.ylabel('y - axis') 
  
# # giving a title to my graph 
# plt.title('farm layout') 

# # function to show the plot 
# plt.show() 

# print(len(y_coord))
# print(len(x_coord))

def plot_farm_as_scatter(turbine_coords):

    x = [ele[0] for ele in turbine_coords]
    y = [ele[1] for ele in turbine_coords]

    plt.scatter(x, y) 

    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    
    plt.title('farm layout') 
    plt.show() 

