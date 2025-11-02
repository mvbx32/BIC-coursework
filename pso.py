
import time
import numpy as np
from particle import Particle

# To do : mark the corresponding lines
# Relecture 
# Put it in a class / function
# 
# ==    Particule Class     ==
# We instantiated the Particule Class takes as input parameter the structure of the ANN 
# The constructor Particle() instantiates a 

# == Definition of the particle structure (ANN structure) ==
Particle.ANN_struture = None 

def AssessFitness(x): # funct input
    return None

swarmsize = 10 # [l1]

max_iteration_loop = 100 
max_iteration_time = 10
criteria = 1

# == PSO parameters == 

alpha = 1 # [l2]
beta = 1   # [l3]
gamma = 1  # [l4]
delta = 1    # [l5]
epsilon = 1   # [l6]

P = set()  # [l7]
for loop in range(swarmsize):  # [l8]
    P.add(Particle()) # [l9] # new random particle  

Best = None # [l10]

t0 = time.time()
it = 0 
while True : # [l11]

   

    for x in P : # [l12]
        AssessFitness(x) # [l13]
       
        
        if Best == None or x.fitness > Best.fitness : #[l14]
            Best = x # [l15]
    for x in P : # [l16]
        vel = x.velocity
        vector = x.vector
        new_vel = np.zeros_like(x.vector)
        # == Update of the fittest per catergory == x*,xplus, x!#
        xstar = x.best_x
        xplus = x.best_informant
        xmark = Particle.fittest_solution

        for i in range(p.vector.shape[0]) : # p.vector[i] = 1 works ? z
            b = np.random(0,1) * beta
            c = np.random(0,1) * gamma
            d = np.random(0,1) * delta
            new_vel[i,:] = alpha*vel[i,:] + b* (xstar - vector[i] ) + c* (xplus - vector[i]) + d * (xmark - vector[i])
        
    for x in P : 
        x.vector = x.vector + epsilon*x.velocity

    if Particle.best_fitness > criteria or time.time()-t0 <= max_iteration_time or it <=max_iteration_loop: break #[l27]

    