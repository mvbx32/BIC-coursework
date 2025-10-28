
import numpy as np
from particle import Particle

# Assumption the ANN structure (nb layers, neuros per layer) is constant during an entire testing session ??Vocabulary ??
# ==    Particule Class     ==
# Whe instantiated the Particule Class takes as input parameter the structure of the ANN 
# The constructor Particle() instantiates a 

# == Definition of the particle structure (ANN structure) ==
Particle.ANN_struture = None 

# 
swarmsize = 10 #
alpha = 1
beta = 1 
gamma = 1
delta = 1
epsilon = 1 

P = set()
for loop in range(swarmsize):
    P.add(Particle()) # new random particle 

Best = None
for p in P : 
    vel = p.velocity
    vector = p.vector
    new_vel = np.zeros_like(p.vector)
    

    # == Update of the fittest per catergory == x*,xplus, x!#
    
    for i in range(p.vector.shape[0]) : # p.vector[i] = 1 works ? 
        b = np.random(0,1) * beta
        c = np.random(0,1) * gamma
        d = np.random(0,1) * delta
        new_vel[i,:] = alpha*vel[i,:] + b* () + c* () + d * ()



    