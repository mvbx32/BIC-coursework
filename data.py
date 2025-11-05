import pandas as pd
import numpy as np
import random

#==             Data        == 

random.seed(42)

# Source : https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

#---Inputs
#Cement	Feature	Continuous		kg/m^3	no
#Blast Furnace Slag	Feature	Integer		kg/m^3	no
#Fly Ash	Feature	Continuous		kg/m^3	no
#Water	Feature	Continuous		kg/m^3	no
#Superplasticizer	Feature	Continuous		kg/m^3	no
#Coarse Aggregate	Feature	Continuous		kg/m^3	no
#Fine Aggregate	Feature	Continuous		kg/m^3	no
#Age	Feature	Integer		day	no

#---Outputs 
#Concrete compressive strength	Target	Continuous		MPa	no

# Advice : 70% for training, 30% for test 
data = np.array(pd.read_excel("data/Concrete_Data.xls"))

# TODO : shuffle the data randomly with a given random seed

sets_index = int(data.shape[0]*0.7)
train_data = data[:sets_index,:] # samples
test_data = data[sets_index:,:]


X_train = train_data[:,:-1]
X_test =  test_data[:,:-1]

Y_train = train_data[:,-1]
Y_test = test_data[:,-1] 
