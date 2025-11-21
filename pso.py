#==================== pso.py   ==============#
import time
import os
import random 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
from particle import Particle
from ANN_alone import * 
import tqdm
import xlrd
from tools import * 
from visualiser import compute_contributions,plot_contributions

# ==           PSO          == 
# AUTHOR : Maxime Vieillot 
# "A specific requirement for this task is that the comments in the code indicate the corresponding
# line numbers in the pseudocode [...]"
# LINE NUMBERS ARE GIVEN AS [l number_line] e.g "#    {some tabulations spaces}  [l1]" (the 1rst line of the pseudo code : setting of the swarmsize)

# Remark : all the '-- logs --..' / '-- {label} --' section are not essential.

class PSO : 
    """
    class PSO 

    That object compute and store all the PSO's variables (see method train).


    PSO's arguments : 
        alpha : float
        beta
        gamma
        delta
        epsi
        ANNStructure : list to instantiate a Particle object (see Particle documentation)
        AssessFitness : callable 
        informants_number
        setInformants : callable

        max_iteration_number, 

    other params :     
        verbose = -1
        path = '/temp/' : str : location to save plots by default

        # to turn on / off the monitoring :
        monitor = True 
        show = True
    
    """

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
        max_iteration_number, verbose = -1, path = '/temp/',monitor = True, show = True):
        
        
        self.ANN_structure = ANNStructure  # given by an ANN instantiation
        
        self.setInformants = setInformants
        self.informants_number = informants_number
        self.fitnessFunc = AssessFitness
        # == PSO parameters == 
        assert(swarmsize > 0 )
        self.swarmsize = swarmsize #           #10 -100                          [l1]
        
        self.alpha = alpha #                                                     [l2]
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
        self.Best       = None # will contain the vector of the best        [l10]                                           [l10]
        layer_sizes = [  layerdim for i,layerdim in enumerate(self.ANN_structure) if i%2 == 0 ]
        activations = [  layerdim for i,layerdim in enumerate(self.ANN_structure) if i%2 == 1 ]
        self.BestANN =   ANN(layer_sizes=layer_sizes, activations=activations)
        self.bestFitness = -np.inf
           
        self.score_train = None
        self.score_test = None 
        self.run_time = None

        # -- Stats  ---------------------------------------------------------------------------------------------------
        
        # ensure path ends with '/'
        if not path.endswith("/"):
            path = path + "/"
            # ensure directory exists
            os.makedirs(path, exist_ok=True)

        self.path = path
        print(path)
        self.show = show
        self.monitor = monitor
        self.verbose = verbose
        self.interDistances_t = {}
        self.MaxDistance = []
        self.MinDistance = []
        self.AVGDistance = []
        self.BestParticle  = []
        self.BestSolutions = []
        self.BestFitnesses = []
        self.BestPaternityPerDecades = { decade :[0 for _ in range(self.swarmsize) ] for decade in [10,20,50,100,1000]}
       
        self.BestPaternityHistory = []
        
        self.GlobalSelfImprovementAVG = []
        self.GlobalSelfImprovementSTD= []
        self.ImprovementOfBest = [0]

        # Accelerations

        self.globalVelMeanComponents = np.zeros((4,self.max_iteration_number))
        self.inertia_history = []
        self.local_history = []
        self.social_history = []
        self.global_history = []
        #----------------------------------------------------------------------------------------------------------------
       

    def compute_interDistances(self):

        self.interDistances_t = {}
        if self.swarmsize >1 : 
                for i,p1 in enumerate(self.P) : 
                    for j,p2 in enumerate(self.P):
                        if not(i == j) and not (j,i) in self.interDistances_t :  # dist(xi,xi) is trivial
                            self.interDistances_t[(i,j)] = np.linalg.norm(p1.vector-p2.vector,2)

                self.MaxDistance.append(np.max(list(self.interDistances_t.values())))
                self.MinDistance.append(np.min(list(self.interDistances_t.values()))) 
                self.AVGDistance.append(np.mean(list(self.interDistances_t.values()))) 
        else : 
            self.MaxDistance.append(0)
            self.MinDistance.append(0)
            self.AVGDistance.append(0) 
            
    def train_step(self):
        pass

    def train(self,resume = False):
        
        """
        
        PSO algorithm
        
        """

        #--------------------------------------------------
        Particle.reset() # reset of the particle Ids buffer
        #--------------------------------------------------

        self.P = []                                                             # [l7]
        for loop in range(self.swarmsize):                                      # [l8]
            p = Particle(self.ANN_structure)
            p.velocity = np.random.uniform(0,1, p.vector.shape[0])              # [l9] # new random particle  
            self.P.append(p)                                                    # [l9] # new random particle  

        t0 = time.time()
        it = 0 

        BestId = None                                                           

        with tqdm.tqdm(total = self.max_iteration_number ) as bar : 

            for t in range(self.max_iteration_number):                          # [l11]
                # == Determination of the Best == 
                
                # -- logs ---------------
                if (t+1)%10 == 0 : 
                    bar.update(t)
                
                RelativeImprovements_t = []
                #------------------------

                for x in self.P :                                                # [l12]
                    x.assessFitness(self.fitnessFunc)                           #  [l13]
                    if  x.fitness > self.bestFitness or self.Best == None :      # [l14]
                        BestId = x.id
                        self.BestParticle = x.copy()
                        self.bestFitness = x.fitness
                        
                        self.Best = x.vector.copy()                             # [l15]
                        self.BestANN.set_params(self.Best)
                    
                        # -- logs ----------------------------
                        
                    #-- logs ---------------------------------

                  
                    RelativeImprovements_t.append(x.improv_x)
                    # ------------------------------------

                # -- logs -----------------------------------------------------------------------------
            
                if self.monitor : 
                    particleVelComponentsMatrix_t = np.zeros((4,1)) 
                    particleVelMeanComponentsMatrix_t = np.zeros((4, x.vector.shape[0]))
                
                #---------------------------------------------------------------------------------------

                # == Determination of each velocities == 
                for j,x in enumerate(self.P) :              # [l16]
                    vel = x.velocity
                    vector = x.vector.copy()
                    new_vel =  x.velocity

                    # == Update of the fittest per catergory (x*,xplus, x!) vector type ====
                    xstar = x.best_x                        # [l17]
                    
                    # definition of the informants
                    x.x_informants = self.setInformants(x,self.P,self.informants_number) 

                    xplus = np.zeros_like(xstar)
                    if self.informants_number != 0 :
                        xplus = x.best_informant            # [l18]
                    
                    xmark = self.Best                       # [l19]
                

                    # -- logs -----------------------------------------------------
                    if self.monitor : x_xi_VelComponentsMatrix = np.zeros((4,x.vector.shape[0]))
                    
                    # [l21], [l22], [l23] 
                    bval , cval, dval = np.random.uniform(0,1,size = x.vector.shape[0]), np.random.uniform(0,1,size = x.vector.shape[0]), np.random.uniform(0,1,size = x.vector.shape[0])
                
                    B = np.diag(bval)*self.beta 
                    C = np.diag(cval)*self.gamma
                    D = np.diag(dval)*self.delta

                    Vinert, Vloc, Vinfo, Vglob = self.alpha * vel , B@(xstar - vector) , C@(xplus - vector) , D@(xmark - vector) #  [l24]
                    x.velocity =  Vinert + Vloc + Vinfo + Vglob                                                                  #  [l24]

                    # -- logs ----------------------------
                    # x_xi ... size (4, components)
                    # (4,number of components)
                  
                    if self.monitor : particleVelMeanComponentsMatrix_t[:,j] =  np.array([np.mean(abs(Vinert)),
                                                                                          np.mean(abs(Vloc)),
                                                                                          np.mean(abs(Vinfo)),
                                                                                          np.mean(abs(Vglob))])#np.mean(x_xi_VelComponentsMatrix, axis=1 ) 
                    #---------------------------------------
                
                # -- logs ----------------------------------------
                
                
                # == Mutation ==   
                for x in self.P :                               # [l25]
                    vector = x.vector.copy()
                    x.vector +=  (self.epsilon*x.velocity)      # [l26]
                        
                #if Particle.best_fitness > criteria : break # [l27]    

                # -- logs ----------------------------
                if self.monitor :
                    self.BestPaternityHistory.append(BestId)
                    self.BestSolutions.append(self.Best)
                    self.BestFitnesses.append(self.bestFitness)
                    self.ImprovementOfBest.append( (self.BestFitnesses[-1] - self.BestFitnesses[-2])/abs(self.BestFitnesses[-1] + self.BestFitnesses[-2 ]) if len(self.BestFitnesses)>=2 and abs(self.BestFitnesses[-1] + self.BestFitnesses[-2 ]) !=0 else 0 )

                    self.GlobalSelfImprovementAVG.append(np.mean(RelativeImprovements_t))
                    self.GlobalSelfImprovementSTD.append(np.std(RelativeImprovements_t))
                    self.globalVelMeanComponents[:,t] = np.mean(particleVelMeanComponentsMatrix_t, axis = 1)
                    self.compute_interDistances()
                # ----------------------------------------------------------------------------------------

        self.run_time = time.time() - t0
        self.score_train = MAE( Data.X_train, Data.Y_train,self.BestANN)
        self.score_test = MAE( Data.X_test, Data.Y_test,self.BestANN)

            
        # -- PLOT  ------------------------------------------------------------------------------------------ 

        if self.show : 
            print("== Plot == ")
            #self.plot()

            try : pass
            except Exception as e : print("Plot failed " ,e)
            finally :pass

        # [l28] : 
        return self.Best, self.bestFitness, self.score_train, self.score_test , self.run_time 


    

    def plot(self):
       
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 120
        plt.rcParams["savefig.dpi"] = 150

        # === 1 — Relative improvements of the SWARM ===
        fig0, ax0 = plt.subplots()
        ax0.set_title("Relative improvements of the SWARM")
        ax0.set_ylim([-0.5, 0.5])
        ax0.set_xlabel("iteration")
        ax0.set_ylabel("relative improvement")

        iterations = range(self.max_iteration_number)

        ax0.plot(iterations, self.GlobalSelfImprovementAVG, '+', c="#008fd5", label="Average")
        ax0.fill_between(iterations,
                        np.array(self.GlobalSelfImprovementAVG)-np.array(self.GlobalSelfImprovementSTD),
                        np.array(self.GlobalSelfImprovementAVG)+np.array(self.GlobalSelfImprovementSTD),
                        color="#BBE4F8",
                        label="Standard deviation")

        ax0.plot(iterations, self.ImprovementOfBest[1:], "+", c="#E4080A", label="Best particle")

        ax0.legend(loc="best")
        fig0.tight_layout(pad=2)
        fig0.savefig(os.path.join(self.path, "Relative_improvements.png"), bbox_inches='tight')

        # === 2 — Each particle's relative improvements ===
        fig, axs = plt.subplots(self.swarmsize, 1, figsize=(10, 2*self.swarmsize))
        if self.swarmsize == 1:
            axs = [axs]

        for i, (ax, p) in enumerate(zip(axs, self.P)):
            if i == 0:
                ax.set_title("Relative improvement per particle")

            ax.set_ylim([-0.5, 0.5])
            ax.plot(p.improv_x_list, linewidth=0.8)
            ax.set_ylabel(f"id {i+1}")
            ax.grid(True)

            if i+1 == self.BestPaternityHistory[-1]:
                ax.yaxis.label.set_color("red")

        axs[-1].set_xlabel("iteration")

        fig.subplots_adjust(hspace=0.5)
        fig.savefig(os.path.join(self.path, "Relative_improvements_particles.png"), bbox_inches='tight')

        # === 3 — Interparticle distances ===
        fig1, ax1 = plt.subplots()
        ax1.set_title("Interparticle distances")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("distance")
        ax1.plot(iterations, self.MaxDistance, label="max", linewidth= 0.8)
        ax1.plot(iterations, self.MinDistance, label="min", linewidth= 0.8)
        ax1.plot(iterations, self.AVGDistance, label="average", linewidth= 0.8)
        ax1.legend(loc="best")

        fig1.tight_layout(pad=2)
        fig1.savefig(os.path.join(self.path,"Interparticle_distances.png"), bbox_inches='tight')

        # === 4 — Velocities components ===
        fig2, ax2 = plt.subplots()
        ax2.set_title("Mean absolute velocity components")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("Average |component|")
        for c, component in enumerate(["inertia","local","social","global" ]) : 
            ax2.plot(iterations, self.globalVelMeanComponents[c,:], label= component, linewidth= 0.8)
        
        ax2.legend(loc="best")
        fig2.tight_layout(pad=2)
        fig2.savefig(os.path.join(self.path,"Velocity_components.png"), bbox_inches='tight')

        # === 5 — Best particle ID ===
        fig3, ax3 = plt.subplots()
        ax3.set_title("Id of the best particle")
        ax3.set_ylim([0, self.swarmsize + 1])
        ax3.set_xlabel("iteration")
        ax3.set_ylabel("id")
        ax3.plot(self.BestPaternityHistory, "s")
        ax3.grid(True)
        ax3.set_yticks([i+1 for i in range(self.swarmsize)])

        fig3.tight_layout(pad=2)
        fig3.savefig(os.path.join(self.path,"Best_particle_id.png"), bbox_inches='tight')

        #decades, contributions= compute_contributions(self.BestPaternityHistory,self.swarmsize,self.max_iteration_number)
        #plot_contributions(contributions,decades)
        
        plt.show()
                
    

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
    max_iteration_number = 1000

    #== PSO == 

    if True : 
        np.random.seed(42)
        random.seed(42)
        t0 = time.time()
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
            monitor = False) 
        pso.train()
        print(time.time()-t0)
        
    if True : 
        np.random.seed(42)
        random.seed(42)
        t0 = time.time()
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
            max_iteration_number = max_iteration_number, monitor = True) 
        

        pso.train()
    
        print(time.time()-t0)
  

  

Particle.reset()
#==================== pso.py  | END ==================#