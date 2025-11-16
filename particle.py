#==================== particule.py   ==============#
import numpy as np
import random 
from ANN_alone import ANN
import matplotlib as plt
# WARNING and sources of mistakes / bugs
# The best solution are stored as class Particle 
# -> a multi task training (several experiments importing Particle at the same time) will fail
# -> Particle class must be reset 

from  data import *
class Particle :

    """

    Given an ANN structure, the Particule class provides a 
    vector representation to tune the associated ANN 
    with a PSO algorithm.

    """
    # == Common variables == 

    # ==   ANN     == 

    particleNumber = 0

    def reset():
        # WARNING : reset the Particle class between two consecutives PSO executions

        # ==   ANN     == 
      
        Particle.ANN_activation = None 
        Particle.particleNumber = 0

        # -- X! -- 
        # np.array type
      

    
    def __init__(self, ANN_structure):  
    
        Particle.particleNumber += 1 
        self.id = Particle.particleNumber

        # init a random vector from a Random ANN

        self.ANN_structure = ANN_structure
        layer_sizes = [  layerdim for i,layerdim in enumerate(ANN_structure) if i%2 == 0 ]
        activations = [  layerdim for i,layerdim in enumerate(ANN_structure) if i%2 == 1 ]

        a = ANN(layer_sizes=layer_sizes, activations=activations)
        self._vector = a.get_params().copy() #
        self._fitness = np.inf *-1
        # Instantiation of the ANN to compute the fitness
        self.ANN_model = ANN(layer_sizes=layer_sizes, activations=activations)
        self.ANN_model.set_params(self.vector)
        
        # Init of the class variables
        
        self.velocity = np.zeros_like(self.vector)

        # -- X* --
        self.best_x =  self             
        self._best_fitness = -1*np.inf # Fitness of the best version of self

        # -- X+ -- 
        self.best_informant =  self.vector  
        self._informants = [self.vector] 
        self.informants_fitness = [self.fitness] # -1*np.inf
        self.best_informant_fitness = self.fitness # -1*np.inf
        # Remark : the fitnesses will be updated during the first Fitness Assessment (at the beginning of the PSO)
       
        # == Stats == 

        self.pbests = []
        self.pbests_fitness = []
        self.improv_x = 0
        self.improv_x_list = [0]

   
    
    def __eq__(self, other): 
        if np.array_equal(self.vector, other.vector):
            return True 
        return False
    
    def __str__(self):
        return str(self.vector)

    def copy(self):

        clone = Particle(self.ANN_structure)
        clone.vector =  self.vector.copy()
        clone.ANN_model = self.ANN_model.copy()

        return clone
     
    # == Vector == 
    @property
    def vector(self) :
        return self._vector
    
    @vector.setter
    def vector(self,v):
        self._vector = v
        # preparation of the ANN model for the fitness calculus
        self.ANN_model.set_params(v) #Particle.vector2ANN(self.vector)
 

    # == Fitness == 
    @property
    def fitness(self): 
        return self._fitness
    
    @fitness.setter
    def fitness(self, new_fitness):  
        
        self._fitness = new_fitness
        # before update of the best to see improvements as spikes
        self.improv_x = (self.fitness  - self._best_fitness)/( self.fitness +self._best_fitness) 
        self.improv_x_list.append(self.improv_x)
     

        # x*
        if self._best_fitness < new_fitness : 
            self._best_fitness =  new_fitness 
            self.best_x = self.vector.copy()
        # x+
        # Remark : Since x belongs to the informants, 
        # so x is better than the best informants implies x is the best informants
        if new_fitness > self.best_informant_fitness : 
            self.best_informant = self.vector.copy()
            self.best_informant_fitness = new_fitness

        self.pbests.append(self.best_x) # 
        self.pbests_fitness.append(self._best_fitness)
      

    def assessFitness(self,fitnessFunc):
        self.fitness= fitnessFunc(self)

    @property
    def informants(self): 
        return self._informants
    
    @informants.setter
    def informants(self,informants_data):      
        
        assert(len(informants_data) == 2)
       
        new_informants, new_informants_fitness = informants_data[0], informants_data[1]
        informant_number = len(new_informants)

        if informant_number != 0 : 
             
            assert(type(new_informants[0]) == Particle and type(new_informants[1]) == float )
            for i in range(len(new_informants)) : 
                infor = new_informants[i]
                infor_fit = new_informants_fitness[i]
                if infor_fit> self.best_informant_fitness : 
                    self.best_informant = infor.copy()
                    self.best_informant_fitness = infor_fit
                if (infor == self.vector).all()  : self_is_in_informant = True
            
            self_is_in_informant = False
            if not self_is_in_informant : #  Injunction [l18] PSO : x must belong to the informants
                
                # replace the last informants generated by self so we still have the right number of informants
                new_informants[-1] = self.vector.copy()
                new_informants_fitness.append(self.fitness)
        # else (no informants) we keep self as the default best informant 
        self._informants  = new_informants 
        self.informants_fitness = new_informants_fitness

  

  
    def export(self): #TODO : complete
        """
        Export the vector as a .txt file. 
        """
        pass

    

if __name__ == "__main__":
    
    from data import Data
    from tools import * 

    Informants = randomParticleSet
    AssessFitness = inv_ANN_MAE
       
    ANN_structure = [8,'input',5,'linear',1,'sigmoid']
    ANN_activation = 'linear'
    Particle(ANN_structure=ANN_structure)

#==================== particule.py  | END ==============#