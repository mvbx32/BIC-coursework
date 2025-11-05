from data import X_train, Y_train, X_test, Y_test
import numpy as np 
import random

random.seed(42)
# == Fitness == 

from particle import Particle

def inv_ANN_MSE(ANN_model):
    if type(ANN_model) == Particle : 
        ANN_model = ANN_model.ANN_model

    # arbitrary choice !!
    mse = 0
    for k in range(Y_train.shape[0]) :    
        err = Y_train[k] - ANN_model.forward(X_train[k])
        mse += np.abs(err)
    mse = mse  / Y_train.shape[0]
    return 1/mse

#== Informants == 

def randomParticleSet(x,P):
    # x Particle 
    # P set of Particle
    # 2nd idea suggested in Lecture 7 : random subset of P
    R = random.sample(P, Particle.informants_number)

    new_informants = [p.vector.copy() for p in R]
    fitnesses =[p.fitness for p in R]

    return new_informants, fitnesses


#==                 LOGS                ==
def logexport():
    pass
