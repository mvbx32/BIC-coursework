#==================== pso.py   ==============#
import time
import random 
import numpy as np
import matplotlib.pyplot as plt
from particle import Particle
from ANN_alone import * 
import tqdm
import xlrd
from tools import * 

# TODO : 

# Set a default activation function as linear when instantiating an ANN
# Looks for biblio ressources to better understand how could be modeled the ANN : linear ? 
# How to test the PSO ???
#       - Compare a linear regression model with a linear ANN based PSO algorithm 
#       - same with forward-backward
# Additional features : sub class of particle to encode the activation function type
# Report on the code structure
# Set research Parameters as default
# -     Setup logs + Saving of intermediar / final models 
# ==           PSO          == 

class PSO : 

    def __init__(self, swarmsize, 
        alpha, 
        beta, 
        gamma,
        delta,
        epsi, 
        ANNStructure, 
        AssessFitness, 
        informants_number, 
        setInformants, 
        max_iteration_number, verbose):
        
        #Remove Best in Particle class
        self.ANN_structure = ANNStructure  # given by an ANN instantiation
       
        Particle.particleNumber = 0 

        self.setInformants = setInformants
        self.informants_number = informants_number
        self.fitnessFunc = AssessFitness

        # == PSO parameters == 
        self.swarmsize = swarmsize #           #10 -100                              [l1]
        
        self.alpha = alpha #         /! rule of thumb (Somme might be 4 )       [l2]
        self.beta = beta   #                                                     [l3]
        self.gamma = gamma  #                                                    [l4]
        self.delta = delta    #                                                  [l5]
        self.epsilon = epsi   #                                                  [l6]
        self.criteria = 1

        self.max_iteration_number = max_iteration_number

        if informants_number == 0 : 
            # <==>
            self.gamma == 0

        # == results == 
        self.P = []                                                          
        self.Best       = None #                                                       [l10]
        self.BestANN    = None
        self.bestFitness = -np.inf
        
        self.score_train = None
        self.score_test = None 
        self.run_time = None

    def train_step(self):
        pass

    def train(self,resume = False):

        self.P = []                                                             #[l7]
        for loop in range(self.swarmsize):                                      #[l8]
            p = Particle(self.ANN_structure)
            self.P.append(p)                                                    #[l9] # new random particle  

        t0 = time.time()
        it = 0 
        for t in range(self.max_iteration_number): #                          [l11]
            print("Iteration {}".format(t))
            # == Determination of the Best == 
            for x in self.P : #                                      [l12]
                x.assessFitness(self.fitnessFunc) #                              [l13]
                if  x.fitness > self.bestFitness: # [l14]
                    self.bestFitness = x.fitness
                    self.Best = x.vector.copy()#                                  [l15]

                    if type(self.BestANN) != ANN:
                         
                        layer_sizes = [  layerdim for i,layerdim in enumerate(self.ANN_structure) if i%2 == 0 ]
                        activations = [  layerdim for i,layerdim in enumerate(self.ANN_structure) if i%2 == 1 ]
                        self.BestANN =   ANN(layer_sizes=layer_sizes, activations=activations)

                    self.BestANN.set_params(self.Best)

            # == Determination of each velocities == 
            for x in self.P : # [l16]
                vel = x.velocity.copy()
                vector = x.vector.copy()
                new_vel =  x.velocity.copy()

                # == Update of the fittest per catergory (x*,xplus, x!) vector type ====
                xstar = x.best_x         #               [l17]
                
                # definition of the informants
                x.x_informants = self.setInformants(x,self.P,self.informants_number) 
                

                xplus = np.zeros_like(xstar)
                if self.informants_number != 0 :
                    xplus = x.best_informant   #               [l18]
                xmark = self.Best #                         [l19]
                
                
                self.alpha = 1 + (1-0.1)*(self.max_iteration_number - t)/self.max_iteration_number # adaptative  w
                
                

                self.beta = 2.05 #*(t-self.max_iteration_number)/self.max_iteration_number # C2
                self.beta = 2.05 #*(self.max_iteration_number - t)/self.max_iteration_number # C1
                for i in range(x.vector.shape[0]) : #           [l20] 
                    b = random.random() * self.beta   #               [l21]
                    c = random.random() * self.gamma  #               [l22]
                    d = random.random() * self.delta  #               [l23]

                    #              self inertia     global term                      social term (informants)     best version of x (~local fittest)
                    new_vel[i] = self.alpha*vel[i] +b* (xstar[i] - vector[i] )  +c* (xplus[i] - vector[i]) + d * (xmark[i] - vector[i])    # [l24]
                x.velocity = new_vel                                                                                        
            
            # == Mutation ==   
            for x in self.P : #                                      [l25]
                vector = x.vector.copy()
                x.vector +=  (self.epsilon*x.velocity) #      [l26]
            
            #if Particle.best_fitness > criteria : break # [l27]
            
            
            it +=1
        self.run_time = time.time() - t0

        self.score_train = MSE( Data.X_train, Data.Y_train,self.BestANN)
        self.score_test = MSE( Data.X_test, Data.Y_test,self.BestANN)
        return self.Best, self.bestFitness, self.score_train, self.score_test , self.run_time 


if __name__ == "__main__" : 

    from data import Data 
    
    np.random.seed(42)
    random.seed(42)
    # %% Example 1 
    # Issue : the result doesnot evolve with the change of max_iter when swarmsize = 1
    Informants = randomParticleSet
    AssessFitness = inv_ANN_MSE
 
    ANNStructure = [8,'input',16,'relu',1,'sigmoid']
  
    swarmsize = 10 
   
    alpha = 1 
    beta  = 1 
    gamma = 1
    delta = 1 
  
    epsi  = 0.3  
    informants_number = 3
    max_iteration_number = 100

    #== PSO == 

    pso = PSO(swarmsize, 
        alpha, 
        beta, 
        gamma,
        delta,
        epsi, 
        ANNStructure, 
        AssessFitness, 
        informants_number, 
        Informants, 
        max_iteration_number = max_iteration_number, 
        verbose = -1) 


    print(pso.train())
    
    pso.max_iteration_number = 10
    print(pso.train())
    pso.max_iteration_number = 20
    print(pso.train())
    

  

Particle.reset()
#==================== pso.py  | END ==================#