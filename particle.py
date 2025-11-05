import numpy as np
import random 
from ANN_alone import ANN
# Clean the code recursively
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
  
   

    #np.array types
    AssessFitness = None
    # callable
    Informants    = None
    informants_number = None
    particleNumber = 0

    # -- X! -- 
    # np.array type
    fittest_solution = None 
    best_fitness = -1*np.inf   
    bestANN = None  
    
    def __init__(self):  # To test # To do : give a default random params value
    
        # * Error * 
        assert(Particle.ANN_structure is not None) # -- Error Msg : ANN_structure undefined --
        assert(Particle.AssessFitness is not None ) # -- Error Msg : Fitness undefined --
        assert(Particle.Informants is not None)    # -- Error Msg : Informants selection undefined --
        assert(Particle.informants_number is not None) 

        Particle.particleNumber += 1 
        self.id = Particle.particleNumber

        # Construction of the vector representation

        # vector = (Layer1 , Layer2 , ... , Layer n)
        #   where Layeri = Suite of each Neurons Weights = Neuron1Weights,Neuron2Weights, ..., NeuronKWeights 
        #       where NeuronjWeights = inputs weights, biais = W1,W2,..,Wl, biais

        # init a random vector
        weight_list = []

        # TODO : replace that by the conversion of a random ANN
        '''
        for i in range(1,len(Particle.ANN_structure)):
            # for each hidden layer + output

            for neur_i in range(Particle.ANN_structure[i]):  
                # for each neuron

                for wi in range(Particle.ANN_structure[i-1]) :
                    # for each weight to the neuron  
                    weight_list.append(random.random())
                
                # bias
                weight_list.append(random.random())
        '''
        a = ANN(layer_sizes=Particle.ANN_structure, activations=["linear" for loop in range(len(Particle.ANN_structure)-1)])
        self._vector = a.get_params().copy()
        # Init of the class variables
        self.__fitness = np.inf *-1
      
        self.velocity = np.zeros_like(self.vector)

        # Instantiation of the ANN to compute the fitness
        self.ANN_model = ANN(layer_sizes=Particle.ANN_structure, activations=["linear" for loop in range(len(Particle.ANN_structure)-1)])
        self.ANN_model.set_params(self.vector)
        
      
        
        # -- X* --
        self._best_fitness = -1*np.inf
        self.best_x =  self             # Particle type

        
        # ??? Previous
         # -- X+ -- 

        self.best_informant =  self.vector   # Particle type  
        self.best_informant_fitness = self.fitness
        self._informants = [self.vector] # list of Particle Type 
        self.informants_fitness = [self.fitness]
       
        if Particle.particleNumber == 0  : 
            Particle.fittest_solution = self.vector.copy() # Particle type
        #self.assessFitness() # initialise the fitness # DANGEROUS
        

    def __eq__(self, other): 
        if np.array_equal(self.vector, other.vector):
            return True 
        return False
    
    def __str__(self):
        return str(self.vector)
    
    """ def copy(self):
        # ==        AI          ==#
        # Model : GPT
        # Version : 5
        # Prompt : "How do define a copy() method for the class Particle ? (given the class scrip as a prior)
      
        
        clone = Particle.__new__(Particle)  # bypass __init__

        # Copie du vecteur et de la vitesse
        clone._vector = np.copy(self._vector)
        clone.velocity = np.copy(self.velocity)

        # Créer un ANN totalement indépendant et le remplir à partir du vecteur copié
        clone.ANN_model = self.vector2ANN(clone._vector)

        # Copie des scalaires / meta
        clone.id = self.id
        clone._best_fitness = self._best_fitness
        clone.__fitness = getattr(self, "_Particle__fitness", None)

        # best_x et best_informant comme SNAPSHOTS (copies indépendantes)
        clone.best_x = self.best_x.copy() if (hasattr(self, "best_x") and self.best_x is not None) else None
        clone.best_informant = self.best_informant.copy() if (hasattr(self, "best_informant") and self.best_informant is not None) else None

        # informants : on peut garder des références ou des copies ; ici on fait des copies
        clone._informants = [p.copy() for p in getattr(self, "_informants", [self])]
        return clone """


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
    def fitness(self): # To test
        return self.__fitness
    
    @fitness.setter
    def fitness(self, new_fitness):  # To test
        
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
                
            #print("BEST", Particle.fittest_solution, Particle.best_fitness)
            #print(Particle.fittest_solution)

    def assessFitness(self):
        #print("AssessFitness" + str(self.id))
        self.fitness= Particle.AssessFitness(self)

    @property
    def informants(self): # To test
        return self._informants
    
    @informants.setter
    def informants(self,informants_data):  # To test    

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

        self._informants  = new_informants # ??? Might cause issues with de copy
        self.informants_fitness = new_informants_fitness
    '''
    @staticmethod
    def vector2ANN(vector):
            # Weight representation in the Layer class : for a given kth layer
        #
        #      \             Neuron1   Neuron2  ...        Neuron n_input
        # Weight 1              .          .                    .
        # Weight 2              .           .   ...             .           
        #   ...
        # Weight n_output       .           .                   .
        ANN_model = ANN(layer_sizes=Particle.ANN_structure, activations=["linear" for loop in range(len(Particle.ANN_structure)-1)])
        
        #assert(ANN_model.layer_sizes == self.ANN_model.layer_sizes) #  Error : ANN structure had been changed
        i = 0
        # for each layer
        for l in range(1,len(ANN_model.layers)) :                      
            # for each neuron of the layer
            for n in range(ANN_model.layers[l].n_neuron):    
                
                # (Neurons input weights .. Neurons biais)
                # for each input of the neuron               
                for inp in range(ANN_model.layers[l].n_input) :               
                    ANN_model.layers[l].W[inp,n]= vector[i]
                    i+=1
                 # bias
                ANN_model.layers[l].b[n] = vector[i]
                i+=1

        return ANN_model
    @staticmethod 
    def ANN2vector(ANN_model):
        
        i = 0
        vector = []
        for l in range(1,len(ANN_model.layers)) :                      
            # for each neuron of the layer
            for n in range(ANN_model.layers[l].n_neuron):    
                
                # (Neurons input weights .. Neurons biais)
                # for each input of the neuron               
                for inp in range(ANN_model.layers[l].n_input) :               
                    vector.append(ANN_model.layers[l].W[inp,n])
                    i+=1
                 # bias
                vector.append(ANN_model.layers[l].b[n])
                i+=1
                
        return np.array(vector)
    '''
    def export(self): #TODO : complete
        """
        Export the vector as a .txt file. 
        """
        pass

    

if __name__ == "__main__":
    
    from data import X_train, Y_train, X_test, Y_test
    from tools import * 

    Informants = randomParticleSet
    AssessFitness = inv_ANN_MSE
       
    Particle.ANN_structure = [8,5,1]
    Particle.AssessFitness = AssessFitness
    Particle.Informants = Informants
    Particle()

