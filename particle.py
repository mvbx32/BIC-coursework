import numpy as np

# Clean the code recursively

class Particle():

    """

    Given an ANN structure, the Particule class provides a 
    vector representation to tune the associated ANN 
    with a PSO algorithm.

    """
    # == Common variables == 

    ANN_struture = None
    
    fittest_solution = None
    best_fitness = -1*np.inf 

    ANN2Vector = callable
    
    
    def __init__(self, params, vel):  # To test # To do : give a default random params value
    
        # * Error * 
        assert(Particle.ANN_struture == None) # -- Error Msg : ANN_structure undefined --
        
        # Instantiation of the ANN used to compute the fitness
        self.ANN_model = None
        
        # Find the corresponding vector representation
        self.vector = self.get_parameters_vector(self.ANN_model)
        self.shape = None 

        self.velocity = 0

        self.fitness = 0
        self._best_fitness = None

        # == PSO variables == 
        self._informants = None # list of Particle Type 

        # TODO : If we are at t, and looking into xi "previous best ..." could it be xi_1(t) ????
        self.best_x =  self.__vector  
        self.best_informant = None # set as self.x_informant.xbest 
        if Particle.fittest_solution == None  : Particle.fittest_solution = self.__vector


    
    def ANN_to_vector(ANN) : # TODO : complete # To test # Remove 
        return None

    
    # == Fitness == 
    @property
    def fitness(self): # To test
        return self.__fitness
    
    @fitness.setter
    def fitness(self, new_fitness):  # To test
        if self._best_fitness < new_fitness : 
            self._best_fitness =  new_fitness 
            self.best_x = self.vector

            if new_fitness > Particle.best_fitness : 
                Particle.best_fitness = new_fitness
                Particle.fittest_solution = self.vector
            
        self.__fitness = new_fitness

    @property
    def informants(self): # To test
        return self._informants
    
    @informants.setter
    def informants(self,new_informants):  # To test    
        self._informants  = new_informants
        for infor in new_informants : 
            if infor.fitness > self.best_informant.fitness : 
                self.best_informant.fitness  = infor
    

    def AssessFitness(self):  # To test
        #
        self.fitness = None 
        pass

    def export(self):
        """
        Export the vector as a .txt file. 
        """
        pass

    

if __name__ == "__main__":
    Particle()

