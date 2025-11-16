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
from visualiser import compute_contributions,plot_contributions
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
        max_iteration_number, verbose = -1):
        
        #Remove Best in Particle class
        self.ANN_structure = ANNStructure  # given by an ANN instantiation
       
        Particle.particleNumber = 0 

        self.setInformants = setInformants
        self.informants_number = informants_number
        self.fitnessFunc = AssessFitness

        # == PSO parameters == 
        assert(swarmsize > 0 )
        self.swarmsize = swarmsize #           #10 -100                              [l1]
        
        self.alpha = alpha #         /! rule of thumb (Somme might be 4 )       [l2]
        self.beta = beta   #                                                     [l3]
        self.gamma = gamma  #                                                    [l4]
        self.delta = delta    #                                                  [l5]
        self.epsilon = epsi   #                                                  [l6]
        self.criteria = 1

        assert(max_iteration_number > 0)
        self.max_iteration_number = max_iteration_number 

        if informants_number == 0 : 
            # <==>
            self.gamma = 0

        # == results == 
        self.P = []                                                          
        self.Best       = None #                                                       [l10]
        self.BestANN    = None
        self.bestFitness = -np.inf

           
        self.score_train = None
        self.score_test = None 
        self.run_time = None

        #Stats
        self.verbose = verbose
        self.MaxDistance = []
        self.MinDistance = []
        self.AVGDistance = []
        self.BestSolutions = []
        self.BestFitnesses = [0]
        self.BestPaternityPerDecades = { decade :[0 for _ in range(self.swarmsize) ] for decade in [10,20,50,100,1000]}
       
        self.BestPaternityHistory = []


        self.GlobalSelfImprovementAVG = []
        self.GlobalSelfImprovementSTD = []

        self.ImprovementOfBest = [0]

    def train_step(self):
        pass

    def train(self,resume = False):

        self.P = []                                                             #[l7]
        for loop in range(self.swarmsize):                                      #[l8]
            p = Particle(self.ANN_structure)
            self.P.append(p)                                                    #[l9] # new random particle  

        t0 = time.time()
        it = 0 

        BestId = None


       
        for t in range(self.max_iteration_number): #                          [l11]
            # == Determination of the Best == 
            
            RelativeImprovements = []
            Distances = np.zeros((self.swarmsize,self.swarmsize))

            for x in self.P : #                                      [l12]
                
                x.assessFitness(self.fitnessFunc) #                              [l13]
                if  x.fitness > self.bestFitness: # [l14]
                    self.bestFitness = x.fitness
                    self.Best = x.vector.copy()#                                  [l15]
                    
                    BestId = x.id
                    
                    if type(self.BestANN) != ANN:
                         
                        layer_sizes = [  layerdim for i,layerdim in enumerate(self.ANN_structure) if i%2 == 0 ]
                        activations = [  layerdim for i,layerdim in enumerate(self.ANN_structure) if i%2 == 1 ]
                        self.BestANN =   ANN(layer_sizes=layer_sizes, activations=activations)

                    self.BestANN.set_params(self.Best)
                
                RelativeImprovements.append(x.improv_x)

            Distances = {}
            for i,p1 in enumerate(self.P) : 
                for j,p2 in enumerate(self.P):
                    if not(i == j) and not (j,i) in Distances :  # dist(xi,xi) is trivial
                        Distances[(i,j)]= np.linalg.norm(p1.vector-p2.vector,2)

          
        
            self.MaxDistance.append(np.max(list(Distances.values())))
            self.MinDistance.append(np.min(list(Distances.values()))) 
            self.AVGDistance.append(np.mean(list(Distances.values()))) 

            self.GlobalSelfImprovementAVG.append(np.mean(RelativeImprovements))
            self.GlobalSelfImprovementSTD.append(np.std(RelativeImprovements))
            self.BestPaternityHistory.append(BestId)
             
            self.BestSolutions.append(self.Best)
            self.BestFitnesses.append(self.bestFitness)
            self.ImprovementOfBest.append((self.BestFitnesses[-1]  - self.BestFitnesses[-2])/(self.BestFitnesses[-1]  + self.BestFitnesses[-2]))
                
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
                
                xmark = self.Best #                            [l19]
               
                #self.alpha = 0.4 + (0.4-0.9)*(t-self.max_iteration_number)/self.max_iteration_number # adaptative  wmax = 0.9 , wmin = 0.4 [Sangputa]

                
                for i in range(x.vector.shape[0]) : #                     [l20] 
                    b = random.random()* self.beta   #               [l21]
                    c = random.random() * self.gamma  #               [l22]
                    d = random.random() * self.delta  #               [l23]

                    #              self inertia     best version of x (~local fittest)  social term (informants)    global term                      
                    new_vel[i] = self.alpha*vel[i] +b* (xstar[i] - vector[i] )  +c* (xplus[i] - vector[i]) + d * (xmark[i] - vector[i])    # [l24]
                                    
                   
                x.velocity = new_vel                                                                                        

            # == Mutation ==   
            for x in self.P : #                                      [l25]
                vector = x.vector.copy()
                x.vector +=  (self.epsilon*x.velocity) #      [l26]
            
            #if Particle.best_fitness > criteria : break # [l27]    
           

       

        self.run_time = time.time() - t0
        self.score_train = MAE( Data.X_train, Data.Y_train,self.BestANN)
        self.score_test = MAE( Data.X_test, Data.Y_test,self.BestANN)

        
        fig1, axs1 = plt.subplots(1, 1, layout='constrained')
        axs1.plot(range(self.max_iteration_number),self.MaxDistance ,label ="Max")
        axs1.plot(range(self.max_iteration_number),self.MinDistance ,label ="Min")
        axs1.plot(range(self.max_iteration_number),self.AVGDistance ,label ="AVG")
        plt.legend()
        plt.plot()
        # == PLOT == 
        if self.verbose != -1 : 
            try : 
                print("== Plot == ")
                fig0, axs0 = plt.subplots(1, 1, layout='constrained')
                axs0.set_ylim([-1,1])
                axs0.plot(range(self.max_iteration_number),np.array(self.GlobalSelfImprovementAVG), label = "Global" ,capsize=3, fmt="+", c="blue")
                axs0.fill_between(range(self.max_iteration_number),np.array(self.GlobalSelfImprovementAVG)-np.array(self.GlobalSelfImprovementSTD), np.array(self.GlobalSelfImprovementAVG)+np.array(self.GlobalSelfImprovementSTD), color = "#AAAAAA60")
                axs0.plot(range(self.max_iteration_number),self.ImprovementOfBest[1:], "x", c = "red", label = "Improvement of the Best")
                axs0.legend()
               
             

                fig, axs = plt.subplots(self.swarmsize, 1, layout='constrained')
                for i,p in enumerate(self.P) : 
                    axs[i].set_ylim([-1,1])
                    axs[i].plot(p.improv_x_list, label = str(i+1))
                    axs[i].set_xlabel('Improvement')
                    axs[i].set_ylabel('id {}'.format(i+1))
                    axs[i].grid(True)
                    if i+1 == self.BestPaternityHistory[-1] : 
                        axs[i].yaxis.label.set_color('red')

                plt.figure()
                plt.title("Best solution particle id")
                plt.plot(self.BestPaternityHistory,"+")
                plt.grid(True)
                plt.yticks([i+1 for i in range(self.swarmsize)])
                plt.show()
                decades, contributions= compute_contributions(self.BestPaternityHistory,self.swarmsize,self.max_iteration_number)
                plot_contributions(contributions,decades)
     
            except : 
                print("Plot failed")
            finally :
                pass


        return self.Best, self.bestFitness, self.score_train, self.score_test , self.run_time 


if __name__ == "__main__" : 

    from data import Data 
    
    np.random.seed(42)
    random.seed(42)
    # %% Example 1 
    # Issue : the result doesnot evolve with the change of max_iter when swarmsize = 1
    Informants = randomParticleSet
    AssessFitness = inv_ANN_MAE
 
    ANNStructure = [8,'input',16,'relu',1,'sigmoid']
  
    swarmsize = 10
   
    alpha = 1 
    beta  = 1 
    gamma = 1
    delta = 1 
  
    epsi  = 0.3  
    informants_number = 3
    max_iteration_number = 30

    #== PSO == 

    if True : 
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


        pso.train()
    if True : 
        pso.max_iteration_number = 30
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


        pso.train()
    
    
  

  

Particle.reset()
#==================== pso.py  | END ==================#