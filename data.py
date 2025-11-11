#==================== main.py   ==============#
import pandas as pd
import numpy as np
import random

#==             Data        == 

# Source : https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

#---Inputs
#Cement			            kg/m^3	
#Blast Furnace Slag			kg/m^3	
#Fly Ash		            kg/m^3	
#Water			            kg/m^3	
#Superplasticizer			kg/m^3	
#Coarse Aggregate		    kg/m^3	
#Fine Aggregate		        kg/m^3	
#Age	Feature	Integer		day	

#Cement	Blast Furnace Slag Fly Ash Superplasticizer	Coarse Aggregate Fine Aggregate Age
#Fly Ash		            
#Water			           	
#Superplasticizer			
#Coarse Aggregate		   
#Fine Aggregate		       	
#Age	

#---Outputs 
#Concrete compressive strength	Target	Continuous		MPa	

class Data :
    """
    Class to support the data as a global variable.
    """
    # Advice : 70% for training, 30% for test 
    data = np.array(pd.read_excel("data/Concrete_Data.xls"))
    #data_list = list(data)
    sets_index = int(data.shape[0]*0.7)
    train_data = data[:sets_index,:] # samples
    test_data = data[sets_index:,:]


    X_train = train_data[:,:-1]
    X_test =  test_data[:,:-1]

    Y_train = train_data[:,-1]
    Y_test = test_data[:,-1]
    
    



# TODO : shuffle the data randomly with a given random seed

#==================== main.py  | END ==============#
