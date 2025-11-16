#==================== tools.py   ==============#
import numpy as np 
import random
from data import Data


#TODO : define a doctring, collect it with function.__doc__  and  save the formulas in a text file

# == Parameters control == 


def LTV(t,tf, yi, yf):
    """
    (t,yi) -> Linear / -> (tf,yf)
    Formula :  yf + (yf-yi)*(t-tf)/tf
    """
    return yf + (yf-yi)*(t-tf)/tf


# == Fitness == 

X_train, Y_train, X_test, Y_test = None, None, None, None
from particle import Particle

def MAE(X,Y,ANN_model):
    """
    MAE = SOMME {for i in [1, Ndata]} ( |Ydata_i - ANN(X_i) |) / Ndata

    """
    if type(ANN_model) == Particle : 
        ANN_model = ANN_model.ANN_model
   
    MAE = 0
    for k in range(len(Y)) :    
        err = Y[k] - ANN_model.forward(X[k]/np.sum(X[k]))
        MAE += np.abs(err)
    MAE = MAE  / Y.shape[0]
    return float(MAE)

def inv_ANN_MAE(ANN_model):
    
    """  1/MAE = 1/( SOMME {for i in [1, Ndata]} ( |Ydata_i - ANN(X_i) |) / Ndata ) """
    if type(ANN_model) == Particle : 
        ANN_model = ANN_model.ANN_model
    MAE = 0
    for k in range(Data.Y_train.shape[0]) :    
        err = Data.Y_train[k] - ANN_model.forward(Data.X_train[k]/np.sum(Data.X_train[k]))
        MAE += np.abs(err)
    MAE = float(MAE  / Data.Y_train.shape[0])
    return 1/MAE

def minusMAE(ANN_model,X = X_train,Y= Y_train):
    if type(ANN_model) == Particle : 
        ANN_model = ANN_model.ANN_model
    MAE = 0
    for k in range(Data.Y_train.shape[0]) :    
        err = Data.Y_train[k] - ANN_model.forward(Data.X_train[k]/np.sum(Data.X_train[k]))
        MAE += np.abs(err)
    MAE = float(MAE  / Data.Y_train.shape[0])
    return -MAE
    
#== Informants == 

def randomParticleSet(x,P,informants_number):

    """
    Gets randomly n particles from P.
    """
    # x Particle 
    # P set of Particle
    # 2nd idea suggested in Lecture 7 : random subset of P
    #Return new_informants (Particle list), fitnesses (list of float)

    R = random.sample(P,informants_number)

    new_informants = [p.vector.copy() for p in R]
    fitnesses =[float(p.fitness) for p in R]

    x.informants = (new_informants,fitnesses)
    return new_informants, fitnesses

def k_nearest_neighboors(x,P, informants_number):
    """
    K nearest particles from x with respect to ||.||2 norm.

    """
    # The nearest_neigboors
    nearestlist = P.copy()

    nearestlist.sort(key = lambda x2 : np.linalg.norm(x2.vector - x.vector,2))
    informants = nearestlist[:informants_number]
    return informants, [float(x.fitness) for x in informants] 

def KNN_andFitness(x,P, informants_number):
    """
    K nearest particles from x with respect to ||.||2 norm 
    including the fitness as position component.
    """
    assert(False) # Function to correct (formula)
    # The nearest_neigboors
    nearestlist = P.copy()
    nearestlist.sort(key = lambda x2 : np.linalg.norm(x2.vector - x.vector,2))
    informants = nearestlist[:informants_number]
    return informants, [float(x.fitness) for x in informants] 


def KNN_andFitness2(x,P, informants_number):
    """
    K nearest particles from x with respect to ||.||2 norm 
    including the fitness as position component.
    """
    assert(False) # Function to correct (formula)
    # The nearest_neigboors
    nearestlist = P.copy()
    nearestlist.sort(key = lambda x2 : np.linalg.norm(x2.vector - x.vector,2))
    informants = nearestlist[:informants_number]
    return informants, [float(x.fitness) for x in informants] 



#==                 LOGS                ==
def logexport():
    pass
#==================== tools.py  | END ==============#