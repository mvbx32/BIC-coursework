#==================== data.py   ==============#
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
#==             Data        == 



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

class Data :
    """
    Class to support the data as a global variable.
    """
    # Advice : 70% for training, 30% for test 

    # == GPT 5 == #

    # Prompt : "For an AI training, what are the robust ways to shuffle and split the data set ?"
    data = np.array(pd.read_excel("data/Concrete_Data.xls"))
    
    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)
                

#==================== main.py  | END ==============#




