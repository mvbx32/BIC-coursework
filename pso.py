import time
import random 
import numpy as np
import matplotlib.pyplot as plt
from particle import Particle
from ANN_alone import * 
import tqdm
import xlrd

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



def PSO(swarmsize, 
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
        max_iteration_number = 2000, 
        verbose = 1) : 
    
    
    # == Definition of the particle structure (ANN structure) ==
    Particle.ANN_structure = ANNStructure  # given by an ANN instantiation
    Particle.ANN_activation = ANN_activation
    Particle.AssessFitness = AssessFitness
    Particle.Informants = Informants
    Particle.informants_number = informants_number

    # == PSO parameters == 
    swarmsize = swarmsize #           #10 -100                              [l1]
    
    alpha = alpha #         /!\ rule of thumb (Somme might be 4 )       [l2]
    beta = beta   #                                                     [l3]
    gamma = gamma  #                                                    [l4]
    delta = delta    #                                                  [l5]
    epsilon = epsi   #                                                  [l6]
    criteria = 1

    P = []      #                                                       [l7]
    for loop in range(swarmsize):  #                                    [l8]
        p = Particle()
        P.append(p) #                                                   [l9] # new random particle  
       
    Best = None #                                                       [l10]

    t0 = time.time()
    it = 0 
    for loop in range(max_iteration_number): #                          [l11]
        print("Loop n°",loop)
    
        # == Determination of the Best == 
        for x in P : #                                      [l12]
            x.assessFitness() #                              [l13]
            if type(Best) != Particle or x.fitness > Best.fitness : # [l14]
                Best = x #                                  [l15]
            
           
        # == Determination of each velocities == 
        for x in P : # [l16]
            vel = x.velocity.copy()
            vector = x.vector.copy()
            new_vel =  x.velocity.copy()

            # == Update of the fittest per catergory (x*,xplus, x!) vector type ====
            xstar = x.best_x         #               [l17]
            # definition of the informants
            x.x_informants = Informants(x,P) 
            xplus = x.best_informant   #               [l18]
            xmark = Particle.fittest_solution #             [l19]

            for i in range(x.vector.shape[0]) : #           [l20] 
                b = random.random() * beta   #               [l21]
                c = random.random() * gamma  #               [l22]
                d = random.random() * delta  #               [l23]
                new_vel[i] = alpha*vel[i] + b* (xstar[i] - vector[i] ) + c* (xplus[i] - vector[i]) + d * (xmark[i] - vector[i]) # [l24]
            x.velocity = new_vel                                                                                          # [l24]
        # == Mutation ==   
        for x in P : #                                      [l25]
            vector = x.vector.copy()
            
            x.vector +=  (epsilon*x.velocity) #      [l26]

        Best = Particle.fittest_solution
           
        #if Particle.best_fitness > criteria : break # [l27]
        
        
        it +=1
    
    return Particle.fittest_solution, # Vector representation

if __name__ == "__main__" : 

    from data import Data 
    from tools import * 
    # %% Example 1 

    Informants = randomParticleSet
    AssessFitness = inv_ANN_MSE
 
    ANNStructure = [8,5,1]
    ANN_activation = "relu"
    swarmsize = 1 # between 10 - 100
    # Acceleration weights | Clue : sum = 4 
    alpha = 1 
    beta  = 1 # cognitive influence ; c1 = 1.49445  https://doi.org/10.1155/2020/8875922
    gamma = 1 # social influence ; c2 = 1.49445 
    delta = 1 
    # Jump size/ learning rate  | Clue : ? 
    epsi  = 0.3  # https://doi.org/10.1155/2020/8875922 0.3
    informants_number = 0 #arbitrary ; ? 
    max_iteration_number = 10 # research paper  # https://doi.org/10.1155/2020/8875922
    
    """
    X_train = np.linspace(0,100,10)
    Y_train = np.linspace(0,100,10)
    """
    print(Particle.fittest_solution)

    #== PSO == 
    best_solution = PSO(swarmsize, 
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
    
    

    # Warning a lot of assumptions : the ANN structure might be inefficient 

    #print("BEST SOlution at the end",  Particle.fittest_solution  )
    #== Test == 
    plt.figure()
    #p = Particle()
    p = Particle()
    ann_model = p.ANN_model
    ann_model.set_params( Particle.fittest_solution  )

    #ann_model.set_params( vect  )

    #ann_model = Particle.vector2ANN(Particle.fittest_solution   ) 
    #BestsolutionList.index(Particle.fittest_solution)
    #print("birth of the last best_sole" , ParticleBirthDate[str(list(best_solution)) ])
    #print(ParticleBirthDate)
    #print("birth of the last REAL best_sole" , ParticleBirthDate[str(list(Particle.fittest_solution)) ])
 
    mse = 0
    for k in range(Data.Y_train.shape[0]) :    
        err = Data.Y_train[k] - ann_model.forward(Data.X_train[k])#Particle.bestANN.forward(X_train[k])
        mse += np.abs(err)
    mse = mse  / Data.Y_train.shape[0]
    fit = 1/mse
    
    print("1/mse ",fit,"Global Fitness saved", "1/MSE", AssessFitness(ann_model), "1/MSE saved", Particle.best_fitness)
    
    X_train, Y_train  = Data.X_test, Data.Y_test  # to modify
    print(1/AssessFitness(Particle.bestANN))




