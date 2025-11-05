import time
import random 
import numpy as np
import matplotlib.pyplot as plt
from particle import Particle
from ANN_alone import * 
import tqdm
import xlrd

# TODO : 
# Issue with how the particle are referenced : pointer issue => use copy ??
# ISSUE : Fitness != mse when swarmsize > 1 but OK when swarmsize = 1 ; the good solution is well identified 
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

np.random.seed(42)

def PSO(swarmsize, 
        alpha, 
        beta, 
        gamma,
        delta,
        epsi, 
        ANNStructure, 
        AssessFitness, 
        informants_number, 
        Informants, 
        max_iteration_number = 2000, 
        verbose = 1) : 
    
    ParticleBirthDate = {}
    BestFitnessList = []
    BestsolutionList = []
    # == Definition of the particle structure (ANN structure) ==
    Particle.ANN_structure = ANNStructure  # given by an ANN instantiation
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
        ParticleBirthDate[str(list(p.vector))] = "init"
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

            for i in range(x.vector.shape[0]) : #           [l20] p.vector[i] = 1 works ? z
                b = random.random() * beta   #               [l21]
                c = random.random() * gamma  #               [l22]
                d = random.random() * delta  #               [l23]
                new_vel[i] = alpha*vel[i] + b* (xstar[i] - vector[i] ) + c* (xplus[i] - vector[i]) + d * (xmark[i] - vector[i]) # [l24]
            x.velocity = new_vel                                                                                          # [l24]
        # == Mutation ==   
        for x in P : #                                      [l25]
            vector = x.vector.copy()
            
            x.vector +=  (epsilon*x.velocity) #      [l26]

            ParticleBirthDate[str(list(x.vector))] = it
        Best = Particle.fittest_solution
           
        #if Particle.best_fitness > criteria : break # [l27]
        BestsolutionList.append(Particle)
        BestFitnessList.append(Particle.best_fitness)
        
        it +=1
    
    return Particle.fittest_solution, BestFitnessList, BestsolutionList, ParticleBirthDate # Vector representation

if __name__ == "__main__" : 

    from data import X_train, Y_train, X_test, Y_test
    from tools import * 
    # %% Example 1 

    Informants = randomParticleSet
    AssessFitness = inv_ANN_MSE
 
    ANNStructure = [8,5,1]
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
    best_solution, best_glob_fitness,  BestsolutionList,  ParticleBirthDate = PSO(swarmsize, 
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
    
    # How to check the correctness of the algorithm  ??
    # Warning a lot of assumptions : the ANN structure might be inefficient 

    #print("BEST SOlution at the end",  Particle.fittest_solution  )
    #== Test == 
    plt.figure()
    #p = Particle()
    p = Particle()
    ann_model = p.ANN_model
    ann_model.set_params( Particle.fittest_solution  )

    vect = np.array( [0.6394268,  0.02501076, 0.27502932, 0.22321074, 0.73647121, 0.67669949,
    0.89217957, 0.08693883 ,0.42192182, 0.02979722 ,0.21863797 ,0.50535529
    ,0.02653597 ,0.19883765 ,0.64988444 ,0.54494148 ,0.22044062 ,0.58926568
    ,0.80943046 ,0.00649876 ,0.80581925 ,0.69813939 ,0.34025052 ,0.1554795
    ,0.95721307 ,0.33659455 ,0.09274584 ,0.09671638 ,0.84749437 ,0.60372603
    ,0.80712827 ,0.72973179 ,0.53622809 ,0.97311576 ,0.37853438 ,0.55204063
    ,0.82940466 ,0.61851975 ,0.8617069  ,0.57735215 ,0.70457184 ,0.04582438
    ,0.22789828 ,0.28938796 ,0.07979198 ,0.23279089 ,0.10100143 ,0.2779736
    ,0.63568444 ,0.36483218 ,0.37018097])

    #ann_model.set_params( vect  )

    print(Particle.fittest_solution, vect)
    #ann_model = Particle.vector2ANN(Particle.fittest_solution   ) 
    #BestsolutionList.index(Particle.fittest_solution)
    #print("birth of the last best_sole" , ParticleBirthDate[str(list(best_solution)) ])
    #print(ParticleBirthDate)
    #print("birth of the last REAL best_sole" , ParticleBirthDate[str(list(Particle.fittest_solution)) ])
 
    mse = 0
    for k in range(Y_train.shape[0]) :    
        err = Y_train[k] - ann_model.forward(X_train[k])#Particle.bestANN.forward(X_train[k])
        mse += np.abs(err)
    mse = mse  / Y_train.shape[0]
    fit = 1/mse
    
    print("1/mse ",fit,"Global Fitness saved", best_glob_fitness[-1], "1/MSE", AssessFitness(ann_model), "1/MSE saved", Particle.best_fitness)
    

    X_train, Y_train  = X_test, Y_test 
    print(1/AssessFitness(Particle.bestANN))


    
    # BestFinestList NOK
    #print(best_glob_fitness)
    #print( BestsolutionList)
    # !!! ERROR 

   

   
    # Compare with a forward backward algorithm
    
    """ plt.title("Fitness evolution")
    plt.plot(best_glob_fitness,'*')
    plt.show()
    print(best_solution) """
    
print("\n=== Diagnostic ===")
print("Best fitness enregistré :", Particle.best_fitness)

# Recalcule le fitness depuis la même solution sauvegardée :
p_test = Particle()
p_test.ANN_model.set_params(Particle.fittest_solution)
f_test = AssessFitness(p_test)

print("Fitness recalculé sur la même solution :", f_test)
print("Différence :", Particle.best_fitness - f_test)
