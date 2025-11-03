import numpy as np
import random 
from ANN_alone import ANN
# Clean the code recursively

class Particle():

    """

    Given an ANN structure, the Particule class provides a 
    vector representation to tune the associated ANN 
    with a PSO algorithm.

    """
    # == Common variables == 

    ANN_structure = None
    AssessFitness = callable
    Informants    = callable
    
    # -- X! -- 
    fittest_solution = None
    best_fitness = -1*np.inf     
    
    def __init__(self):  # To test # To do : give a default random params value
    
        # * Error * 
        assert(Particle.ANN_structure != None) # -- Error Msg : ANN_structure undefined --
        assert(Particle.AssessFitness != callable) # -- Error Msg : Fitness undefined --
        assert(Particle.Informants != callable)    # -- Error Msg : Informants selection undefined --

        # Construction of the vector representation
    
        weight_list = []

        for i in range(1,len(Particle.ANN_structure)):
            # for each layer

            for neur_i in range(Particle.ANN_structure[i]):  
                # for each neuron

                for wi in range(Particle.ANN_structure[i-1]) :
                    # for each weigth to the neuron  
                    weight_list.append(random.random())

        self._vector = np.array(weight_list).transpose()
        self.velocity = np.zeros_like(self.vector)

        # Instantiation of the ANN used to compute the fitness
        self.ANN_model = ANN(layer_sizes = Particle.ANN_structure, activations=["relu","relu"]) 
    
        
        
        # -- X* --
        self._best_fitness = -1*np.inf # integrate the FItness function to the class sinon, initialisant à np.inf on peut passer à côté de la solution
        self.best_x =  self.vector  

        self._informants = [self] # list of Particle Type 

        # ??? Previous
        
        # -- X+ -- 
        self.best_informant =  self  # set as self.x_informant.xbest 
        if type(Particle.fittest_solution) != np.array  : Particle.fittest_solution = self.vector

        self.fitness = Particle.AssessFitness(self)

    def __eq__(self, other): 
        if not((self.vector - other.vector).all()) :
            return True 
        return False
    
    # == Vector == 
    @property
    def vector(self) :
        return self._vector
    
    @vector.setter
    def vector(self,v):
        self.vector = v
        # preparation of the ANN model for the fitness calculus
        self.ANN_model = self.vector2ANN()
    

    # ==   ANN     == 
    def vector2ANN(self):
        i = 0
        for k in range(1,len(self.ANN.layers)) :                      # for each layer
            for l in range(len(self.ANN.layers[i])):                    # for each neuron of the layer
                for j in range(len(self.ANN.layers[i-1])) :               # for each input of the neuron
                    self.ANN.layers[i].W[j,l]= self.vector[i]
                    i+=1

        
    # == Fitness == 
    @property
    def fitness(self): # To test
        return self.__fitness
    
    @fitness.setter
    def fitness(self, new_fitness):  # To test
        
        self.__fitness = new_fitness

        # x*
        if self._best_fitness < new_fitness : 
            self._best_fitness =  new_fitness 
            self.best_x = self.vector

            # x+
            if new_fitness > self.best_informant.fitness : 
                self.best_informant = self

            # x!
            if new_fitness > Particle.best_fitness : 
                Particle.best_fitness = new_fitness
                Particle.fittest_solution = self.vector

    def assessFitness(self):
        self.fitness= Particle.AssessFitness(self)

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
                self.best_informant = infor


    def export(self): #TODO : complete
        """
        Export the vector as a .txt file. 
        """
        pass

    

if __name__ == "__main__":
    Particle.ANN_structure = [8,5,1]
    Particle()

