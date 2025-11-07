#==================== tools.py   ==============#
import numpy as np 
import random
from data import Data


# == Fitness == 

X_train, Y_train, X_test, Y_test = None, None, None, None
from particle import Particle

def MSE(X,Y,ANN_model):
    if type(ANN_model) == Particle : 
        ANN_model = ANN_model.ANN_model
   
    mse = 0
    for k in range(len(Y)) :    
        err = Y[k] - ANN_model.forward(X[k])
        mse += np.abs(err)
    mse = mse  / Y.shape[0]
    return float(mse)

def inv_ANN_MSE(ANN_model):
    
    if type(ANN_model) == Particle : 
        ANN_model = ANN_model.ANN_model
    mse = 0
    for k in range(Data.Y_train.shape[0]) :    
        err = Data.Y_train[k] - ANN_model.forward(Data.X_train[k])
        mse += np.abs(err)
    mse = float(mse  / Data.Y_train.shape[0])
    return 1/mse

#== Informants == 

def randomParticleSet(x,P,informants_number):
    # x Particle 
    # P set of Particle
    # 2nd idea suggested in Lecture 7 : random subset of P
    #Return new_informants (Particle list), fitnesses (list of float)
    R = random.sample(P,informants_number)

    new_informants = [p.vector.copy() for p in R]
    fitnesses =[p.fitness for p in R]

    x.informants = (new_informants,fitnesses)
    return new_informants, fitnesses


#==                 LOGS                ==
def logexport():
    pass
#==================== tools.py  | END ==============#