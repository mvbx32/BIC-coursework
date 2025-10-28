
import time
import numpy as np
from particle import Particle

# To do : mark the corresponding lines
# Relecture 
# Assumption the ANN structure (nb layers, neuros per layer) is constant during an entire testing session ??Vocabulary ??
# ==    Particule Class     ==
# Whe instantiated the Particule Class takes as input parameter the structure of the ANN 
# The constructor Particle() instantiates a 

# == Definition of the particle structure (ANN structure) ==
Particle.ANN_struture = None 

swarmsize = 10 #

max_iteration_loop = 100
max_iteration_time = 10
criteria = 1

# == PSO parameters == 

alpha = 1
beta = 1 
gamma = 1
delta = 1
epsilon = 1 

P = set()
for loop in range(swarmsize):
    P.add(Particle()) # new random particle 

Best = None

t0 = time.time()
it = 0 
while time.time()-t0 <= max_iteration_time or it <=max_iteration_loop : 

    if Particle.best_fitness > criteria : break 

    for x in P : 
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




    