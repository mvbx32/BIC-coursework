#==================== particule.py   ==============#
import numpy as np
import random 
from ANN_alone import ANN


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
    ANN_structure = None
    ANN_activation = None 

    #np.array types
    AssessFitness = None
    # callable
    Informants        = None
    informants_number = None
    particleNumber = 0

    # -- X! -- 
    # np.array type
    fittest_solution = None 
    best_fitness = -1*np.inf   
    bestANN = None  


    def reset():
        # WARNING : reset the Particle class between two consecutives PSO executions

        # ==   ANN     == 
        Particle.ANN_structure = None
        Particle.ANN_activation = None 

        #np.array types
        Particle.AssessFitness = None
        # callable
        Particle.Informants        = None
        Particle.informants_number = None
        Particle.particleNumber = 0

        # -- X! -- 
        # np.array type
        Particle.fittest_solution = None 
        Particle.best_fitness = -1*np.inf   
        Particle.bestANN = None  

    
    def __init__(self):  
    
        # * Error * #
        assert(Particle.ANN_structure is not None) # -- Error Msg : ANN_structure undefined --
        assert(Particle.AssessFitness is not None ) # -- Error Msg : Fitness undefined --
        assert(Particle.Informants is not None)    # -- Error Msg : Informants selection undefined --
        assert(Particle.informants_number is not None) 
        # *-------* #

        Particle.particleNumber += 1 
        self.id = Particle.particleNumber

        # init a random vector from a Random ANN
        a = ANN(layer_sizes=Particle.ANN_structure, activations= [Particle.ANN_activation]*(len(Particle.ANN_structure)-1) )
        self._vector = a.get_params().copy()
        # Init of the class variables
        self.__fitness = np.inf *-1
      
        self.velocity = np.zeros_like(self.vector)

        # Instantiation of the ANN to compute the fitness
        self.ANN_model = ANN(layer_sizes=Particle.ANN_structure, activations=  [Particle.ANN_activation]*(len(Particle.ANN_structure)-1) )
        self.ANN_model.set_params(self.vector)
        
        # -- X* --
        self._best_fitness = -1*np.inf
        self.best_x =  self             # Particle type

        # -- X+ -- 

        self.best_informant =  self.vector   # Particle type  
        self.best_informant_fitness = self.fitness
        self._informants = [self.vector] # list of Particle Type 
        self.informants_fitness = [self.fitness]
       
        if Particle.particleNumber == 0  : 
            Particle.fittest_solution = self.vector.copy() # Particle type
        

    def __eq__(self, other): 
        if np.array_equal(self.vector, other.vector):
            return True 
        return False
    
    def __str__(self):
        return str(self.vector)

    def copy(self):
        '''
        clone = Particle()

        clone.vector = self.vector.copy()
        clone.ANN_model = self.ANN_model.copy()

        clone.fittest_solution = self.fittest_solution()
        pass
        '''
        return
     
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
        return self.__fitness
    
    @fitness.setter
    def fitness(self, new_fitness):  
        
        self.__fitness = new_fitness

        # x*
        if self._best_fitness < new_fitness : 
            self._best_fitness =  new_fitness 
            self.best_x = self.vector.copy()


        # x+
        if new_fitness > self.best_informant_fitness : 
            self.best_informant = self.vector.copy()
            self.best_informant_fitness = new_fitness
        # x!
        if new_fitness > Particle.best_fitness : 

            Particle.best_fitness = new_fitness
            Particle.bestANN  = self.ANN_model.copy()
            Particle.fittest_solution = self.ANN_model.get_params().copy()
            print(new_fitness)
            
        
            if new_fitness == np.array([0.00169545]) : 
                print(Particle.fittest_solution, mse)
                print()
         

    def assessFitness(self):
        self.fitness= Particle.AssessFitness(self)

    @property
    def informants(self): 
        return self._informants
    
    @informants.setter
    def informants(self,informants_data):      

        new_informants, new_informants_fitness = informants_data
        self_is_in_informant = False

        
        for i in range(len(new_informants)) : 
            infor = new_informants[i]
            infor_fit = new_informants_fitness[i]
            if infor_fit> self.best_informant_fitness : 
                self.best_informant = infor.copy()
                self.best_informant_fitness = infor_fit
            if (infor == self.vector).all()  : self_is_in_informant = True

        if not self_is_in_informant : #  Injunction [l18] PSO : x must belong to the informants
            new_informants.append(self.vector.copy()) 
            new_informants_fitness.append(self.fitness)

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
    AssessFitness = inv_ANN_MSE
       
    Particle.ANN_structure = [8,5,1]
    Particle.ANN_activation = 'linear'
    Particle.AssessFitness = AssessFitness
    Particle.Informants = Informants
    Particle.informants_number = 0
    Particle()

#==================== particule.py  | END ==============#