import numpy as np
import random 
from ANN import ANN
# Clean the code recursively

class Particle():

    """

    Given an ANN structure, the Particule class provides a 
    vector representation to tune the associated ANN 
    with a PSO algorithm.

    """
    # == Common variables == 

    ANN_struture = None
    
    # x!
    fittest_solution = None
    best_fitness = -1*np.inf     
    
    def __init__(self, params, vel):  # To test # To do : give a default random params value
    
        # * Error * 
        assert(Particle.ANN_struture == None) # -- Error Msg : ANN_structure undefined --
        
    
        # Find the corresponding vector representation
        # Boucle FOR : parcours structure layer par layer, neuron par neuro => poids initialisé ALEATOIRE SI params = NONE 
        
        self.vector = []

        for i in range(1,len(Particle.ANN_structure)):
            # for each layer

            for neur_i in range(Particle.ANN_struture[i]):  
                # for each neuron

                for wi in range(Particle.ANN_struture[i-1]) :
                    # for each weigth to the neuron  
                    self.vector.append(random.random())

        self.vector = np.array(self.vector).transpose()



             

        self.vector = None  #self.get_parameters_vector(self.ANN_structure)
        self.velocity = 0

        # Instantiation of the ANN used to compute the fitness
        self.ANN_model = ANN(layer_sizes = Particle.ANN_struture) 
        



        # x*
        self.fitness = np.inf
        self._best_fitness = np.inf

        # == PSO variables == 
        self._informants = None # list of Particle Type 

        # TODO : If we are at t, and looking into xi "previous best ..." could it be xi_1(t) ????
        self.best_x =  self.__vector  
        self.best_informant = None # set as self.x_informant.xbest 
        if Particle.fittest_solution == None  : Particle.fittest_solution = self.__vector


    
    def ANN_to_vector(ANN) : # TODO : complete # To test # Remove 
        return None

    def __eq__(self, other): 
        if self.vector == other.vector :
            return True 
        return False

     
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
        if not self in new_informants :  # Injunction [l18] PSO : x must belong to the informants
            #/!\ DEBUG IN operator 
            new_informants.append(self)

        self._informants  = new_informants
    
        for infor in new_informants : 
            if infor.fitness > self.best_informant.fitness : 
                self.best_informant.fitness  = infor


    def export(self):
        """
        Export the vector as a .txt file. 
        """
        pass

    

if __name__ == "__main__":
    Particle()

