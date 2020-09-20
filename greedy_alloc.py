from Farm_Evaluator_Vec import checkConstraints, getTurbLoc, loadPowerCurve, binWindResourceData, preProcessing, getAEP
from pprint import pprint
from tqdm import tqdm
import numpy  as np
import pandas as pd                     
from   math   import radians as DegToRad     
from shapely.geometry import Point      
from shapely.geometry.polygon import Polygon
from plot_farm import plot_farm_as_scatter
import pickle


import warnings
warnings.filterwarnings("ignore")

bound_clrnc      = 50
farm_peri = [(0, 0), (0, 4000), (4000, 4000), (4000, 0)]
farm_poly = Polygon(farm_peri)

def checkConstraintsNew(turb_coords, turb_diam):

    prox_constr_viol = False
    peri_constr_viol = False
   
    for turb in turb_coords:
        turb = Point(turb)
        inside_farm   = farm_poly.contains(turb)
        correct_clrnc = farm_poly.boundary.distance(turb) >= bound_clrnc
        if (inside_farm == False or correct_clrnc == False):
            peri_constr_viol = True
            return(False)
            break
    
    for i,turb1 in enumerate(turb_coords):
        for turb2 in np.delete(turb_coords, i, axis=0):
            if  np.linalg.norm(turb1 - turb2) < 4*turb_diam:
                prox_constr_viol = True
                return(False)
                break

    return(True)

def save_state(n_turbines, coords):
    saved_state = {"n_turbines": n_turbines, "coords": coords}
    
    file_pi = open('./saved_states/{}'.format(n_turbines), 'w') 
    pickle.dump(object_pi, file_pi)



if __name__ == "__main__":
    
    turb_specs    =  {   
                        'Name': 'Anon Name',
                        'Vendor': 'Anon Vendor',
                        'Type': 'Anon Type',
                        'Dia (m)': 100,
                        'Rotor Area (m2)': 7853,
                        'Hub Height (m)': 100,
                        'Cut-in Wind Speed (m/s)': 3.5,
                        'Cut-out Wind Speed (m/s)': 25,
                        'Rated Wind Speed (m/s)': 15,
                        'Rated Power (MW)': 3
                    }

    coords = []
    turb_diam = 50
    turb_diam      =  turb_specs['Dia (m)']
    turb_rad       =  turb_diam/2 
    step = 50

    turb_coords    =  getTurbLoc(r'./Shell_Hackathon Dataset/turbine_loc_test_2.csv')
    power_curve    =  loadPowerCurve('./Shell_Hackathon Dataset/power_curve.csv')
    wind_inst_freq =  binWindResourceData(r'./Shell_Hackathon Dataset/wind_data/wind_data_2007.csv')
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve, turb_coords.shape[0])
    # checkConstraints(turb_coords, turb_diam)

    # add turbines in 4 corners
    coords.append([50,50])
    coords.append([3950,50])
    coords.append([50,3950])
    coords.append([3950,3950])

    print("Added turbines at 4 corner...")

    for i in range(46):

        print("Adding turbine no.: {}".format(i+5))
        max_pow = 0

        for x in tqdm(range(50, 3950, step)):

            for y in range(50, 3950, step):

                new_coord = [x, y]
                coords.append(new_coord)

                # print("Testing coord: {}".format(new_coord))

                # check if coords are valid
                if(checkConstraintsNew(turb_coords=coords, turb_diam=turb_diam)):
                    pass
                    # print("{} is valid location for new turbine, storing power.".format(new_coord))
                else:
                    # print("{} is invalid location for new turbine, skipping.".format(new_coord))
                    coords.remove(new_coord)
                    continue
                
                #find power of current state
                pow_here = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t) 
                
                # update max pow
                if(pow_here > max_pow):
                    max_pow = pow_here
                    max_coord = new_coord

                coords.remove(new_coord)

        print("Turbine {}: max pow for {} is {}".format(i+5, max_pow, max_coord))
        coords.append(max_coord)
        save
    plot_farm_as_scatter(coords)

        
    
    





            






