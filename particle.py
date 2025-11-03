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
    AssessFitness = None
    Informants    = None
    
    # -- X! -- 
    fittest_solution = None # Particle type
    best_fitness = -1*np.inf     
    
    def __init__(self):  # To test # To do : give a default random params value
    
        # * Error * 
        assert(Particle.ANN_structure != None) # -- Error Msg : ANN_structure undefined --
        assert(Particle.AssessFitness is not None ) # -- Error Msg : Fitness undefined --
        assert(Particle.Informants is not None)    # -- Error Msg : Informants selection undefined --

        # Construction of the vector representation

        # vector = (Layer1 , Layer2 , ... , Layer n)
        #   where Layeri = Suite of each Neurons Weights = Neuron1Weights,Neuron2Weights, ..., NeuronKWeights 
        #       where NeuronjWeights = inputs weights, biais = W1,W2,..,Wl, biais
        
        weight_list = []

        for i in range(1,len(Particle.ANN_structure)):
            # for each layer

            for neur_i in range(Particle.ANN_structure[i]):  
                # for each neuron

                for wi in range(Particle.ANN_structure[i-1]) :
                    # for each weigth to the neuron  
                    weight_list.append(random.random())
                
                # bias
                weight_list.append(random.random())

        self._vector = np.array(weight_list).transpose()
        self.velocity = np.zeros_like(self.vector)

        # Instantiation of the ANN used to compute the fitness
        self.ANN_model = ANN(layer_sizes=Particle.ANN_structure, activations=["relu" for loop in range(len(Particle.ANN_structure)-1)])
        
        self.ANN_model = self.vector2ANN()
    
        print(self.ANN_model.layer_sizes)
        
        # -- X* --
        self._best_fitness = -1*np.inf # integrate the FItness function to the class sinon, initialisant à np.inf on peut passer à côté de la solution
        self.best_x =  self             # Particle type

        self._informants = [self] # list of Particle Type 

        # ??? Previous
        
        # -- X+ -- 
        self.best_informant =  self     # Particle type  
        if type(Particle.fittest_solution) != Particle  : Particle.fittest_solution = self # Particle type

        self.assessFitness() # initialise the fitness

    def __eq__(self, other): 
        if np.array_equal(self.vector, other.vector):
            return True 
        return False
    
    def __str__(self):
        return str(self.vector)
    

    # == Vector == 
    @property
    def vector(self) :
        return self._vector
    
    @vector.setter
    def vector(self,v):
        self._vector = v
        # preparation of the ANN model for the fitness calculus
        self.ANN_model = self.vector2ANN()
    

    # ==   ANN     == 
    def vector2ANN(self):
            # Weight representation in the Layer class : for a given kth layer
        #
        #      \             Neuron1   Neuron2  ...        Neuron n_input
        # Weight 1              .          .                    .
        # Weight 2              .           .   ...             .           
        #   ...
        # Weight n_output       .           .                   .
        #ANN_model = ANN(layer_sizes=Particle.ANN_structure, activations=["relu" for loop in range(len(Particle.ANN_structure)-1)])
        
        #assert(ANN_model.layer_sizes == self.ANN_model.layer_sizes) #  Error : ANN structure had been changed
        i = 0
        # for each layer
        for k in range(1,len(self.ANN_model.layers)) :                      
            # for each neuron of the layer
            for l in range(self.ANN_model.layers[k].n_neuron):    

                # for each input of the neuron               
                for j in range(self.ANN_model.layers[k].n_input) :               
                    self.ANN_model.layers[k].W[j,l]= self.vector[i]
                    i+=1

                 # bias
                self.ANN_model.layers[k].b[l] = self.vector[i]
                i+=1

        return self.ANN_model
        
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
            self.best_x = self


            # x+
            if new_fitness > self.best_informant.fitness : 
                self.best_informant = self

            # x!
            if new_fitness > Particle.best_fitness : 
                Particle.best_fitness = new_fitness
                Particle.fittest_solution = self

    def assessFitness(self):
        print("AssessFitness")
        self.fitness= Particle.AssessFitness(self)

        print(self.fitness)


    @property
    def informants(self): # To test
        return self._informants
    
    @informants.setter
    def informants(self,new_informants):  # To test    
        if not self in new_informants :  # Injunction [l18] PSO : x must belong to the informants
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
    
    def Informants(x,P, informants_number):
        # x Particle 
        # P set of Particle
        # 2nd idea suggested in Lecture 7 : random subset of P
        return random.sample(P, informants_number)
    
    def AssessFitness(x):
        # arbitrary choice !!
        mse = 0
        for k in range(Y_train.shape[0]) :    
            err = Y_train[k] - x.ANN_model.forward(X_train[k])
            mse += np.abs(err)
        mse = mse  / Y_train.shape[0]

        return 1/mse # maximizing the fitness we minimize the error (>0)
       
    Particle.ANN_structure = [8,5,1]
    Particle.AssessFitness = AssessFitness
    Particle.Informants = Informants
    Particle()

