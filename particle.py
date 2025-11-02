import numpy as np

# To do : choose how to consider the fitness (LOW = GOOD ?), change the order relation in consequence
# Let us consider a High fitness as GOOD
# Unittests 
#
class Particle():

    """

    Given an ANN structure, the Particule class provides a 
    vector representation to tune the associated ANN 
    with a PSO algorithm.

    """
    # == Common variables == 

    ANN_struture = None
    best_fitness = -1*np.inf 
    fittest_solution = None
    
    def __init__(self, params, vel):  # To test # To do : give a default random params value
        # params
        assert(Particle.ANN_struture == None) # -- Error Msg : ANN_structure undefined --
        
        # instantiation of the ANN
        self.ANN_model = None

        # -- instantiation with the parameters -- # ANN implementation , To do : is it easier to gen ANN from the vector ? Depends on Arthur implementation
        self.vector = None # np.array type
        # find the corresponding vector representation
        self.vector = self.get_parameters_vector(self.ANN_model)
        self.velocity = 0

        self.fitness = 0
        self._best_fitness = None
        

        self.shape = None 

        # == PSO variables == 

        self.x_informant = None # Particle Type 

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

