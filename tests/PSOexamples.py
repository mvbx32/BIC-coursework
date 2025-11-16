
import sys
sys.path.insert(0, './')

from pso import *

if __name__ == "__main__":

    from data import Data

    Data.X_train = np.linspace(0,100,100).T
    Data.Y_train = Data.X_train

    from tools import * 
    
    # %% Example 1 - Linear case OK

    np.random.seed(42)
    random.seed(42)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('PSO trained ANN on basics function')


    Informants = randomParticleSet
    AssessFitness = inv_ANN_MAE
    ANNStructure = [1,1]
    ANN_activation = "linear"
    swarmsize = 100 # between 10 - 100
    alpha = 1 
    beta  = 1  
    gamma = 1 
    delta = 1 
    epsi  = 0.3 
    informants_number =1 #arbitrary ; ? 
    max_iteration_number = 10
    
    print(Particle.fittest_solution)

   
    #== PSO == 
    best_solution, best_fitness, score_train, score_test = PSO(swarmsize, 
        alpha, 
        beta, 
        gamma,
        delta,
        epsi, 
        ANNStructure,
        ANN_activation,
        AssessFitness, 
        informants_number, 
        Informants, 
        max_iteration_number = max_iteration_number, 
        verbose = -1) 

    
    axs[0,0].plot([Particle.bestANN.forward(np.array([i])) for i in range(10)], '+', label= "ANN")
    axs[0,0].plot([Y_train],"+",label= "data" )

    Particle.reset()
   
    ANN_activation = "relu"

    #== PSO == 
    best_solution, best_fitness, score_train, score_test = PSO(swarmsize, 
        alpha, 
        beta, 
        gamma,
        delta,
        epsi, 
        ANNStructure,
        ANN_activation,
        AssessFitness, 
        informants_number, 
        Informants, 
        max_iteration_number = max_iteration_number, 
        verbose = -1) 

    axs[0,1].plot([Particle.bestANN.forward(np.array([i])) for i in range(10)], '+', label= "ANN")
    axs[0,1].plot([Y_train],"+",label= "data" )


    #NEED TO RESET THE Particle BESTS
    Particle.reset()
   
    ANN_activation = "sigmoid"

    #== PSO == 
    best_solution, best_fitness, score_train, score_test = PSO(swarmsize, 
        alpha, 
        beta, 
        gamma,
        delta,
        epsi, 
        ANNStructure,
        ANN_activation,
        AssessFitness, 
        informants_number, 
        Informants, 
        max_iteration_number = max_iteration_number, 
        verbose = -1) 
    
    axs[1,1].plot([Particle.bestANN.forward(np.array([i])) for i in range(10)], '+', label= "ANN")
    axs[1,1].plot([Y_train],"+",label= "data" )
    fig.legend()
    plt.show()
    