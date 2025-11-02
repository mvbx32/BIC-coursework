
import time
import random 
import numpy as np
from particle import Particle

# TODO : 

#-      Clarrify the definition of the Fitness etc in the Particle Class

#-       Setup data importation
#- ADD ANN implementation
# Write example function for the Fitness 

# - TEST    Unittest (GPT)

# Find PSO examples / recommendation (course material)

# -  Parameters 
# Find similar work (search by parameter and define them)   
#   # social influence ; c2 = 1.49445 
# Set research Parameters as default

# -     Setup logs + Saving of intermediar / final models 


# ==            ANN         == 

#

#==             Data        == 

# Source : https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

#---Inputs
#Cement	Feature	Continuous		kg/m^3	no
#Blast Furnace Slag	Feature	Integer		kg/m^3	no
#Fly Ash	Feature	Continuous		kg/m^3	no
#Water	Feature	Continuous		kg/m^3	no
#Superplasticizer	Feature	Continuous		kg/m^3	no
#Coarse Aggregate	Feature	Continuous		kg/m^3	no
#Fine Aggregate	Feature	Continuous		kg/m^3	no
#Age	Feature	Integer		day	no

#---Outputs 
#Concrete compressive strength	Target	Continuous		MPa	no

# TODO 
# Advice : 70% for training, 30% for test 

train_data  = []
test_data = []


#==                 LOGS                ==
def logexport():
    pass



# ==           PSO          == 
# -- Inputs 
#       int swarmsize
#       
#       func ANN object / ANN structure ???
#       func ANN vector representation. (Could vary depending on whether activation functions are taken into account.)
#       The accelerations 
#       the step epsi
#       Informants list size (is needeed)
#       func Fitness Function
#       func Informant Function 
# The other variables will be considered as constant during the training process ???

# ==    Particule Class     ==
# We instantiated the Particule Class takes as input parameter the structure of the ANN 
# The constructor Particle() instantiates a 



def PSO(swarmsize, 
        alpha, 
        beta, 
        gamma,
        delta,
        epsi, 
        ANN, 
        ANN2Vector,
        AssessFitness, 
        informants_number, 
        Informants, 
        max_iteration_number = 2000, 
        verbose = 1) : 
    
    # == Definition of the particle structure (ANN structure) ==
    Particle.ANN_struture = None  # given by an ANN instantiation
    

    # == PSO parameters == 
    swarmsize = swarmsize #           #10 -100                              [l1]
    
    alpha = alpha #         /!\ rule of thumb (Somme might be 4 )    [l2]
    beta = beta   #                                            [l3]
    gamma = gamma  #                                            [l4]
    delta = delta    #                                          [l5]
    epsilon = epsilon   #                                         [l6]
    criteria = 1

    P = []      #                                           [l7]
    for loop in range(swarmsize):  #                        [l8]
        P.append(Particle()) #                              [l9] # new random particle  

    Best = None #                                           [l10]

    t0 = time.time()
    it = 0 
    while True : #                                          [l11]
        try : 
            # == Determination of the Best == 
            for x in P : #                                      [l12]
                AssessFitness(x) #                              [l13]
                if Best == None or x.fitness > Best.fitness : # [l14]
                    Best = x #                                  [l15]

            # == Determination of each velocities == 
            for x in P : # [l16]
                vel = x.velocity
                vector = x.vector
                new_vel = np.zeros_like(x.vector)

                # == Update of the fittest per catergory (x*,xplus, x!) ====
                xstar = x.best_x                #               [l17]
                # definition of the informants
                x.x_informants = Informants(x,P) 
                xplus = x.best_informant        #               [l18]
                xmark = Particle.fittest_solution #             [l19]

                for i in range(p.vector.shape[0]) : #           [l20] p.vector[i] = 1 works ? z
                    b = np.random(0,1) * beta   #               [l21]
                    c = np.random(0,1) * gamma  #               [l22]
                    d = np.random(0,1) * delta  #               [l23]
                    new_vel[i,:] = alpha*vel[i,:] + b* (xstar - vector[i] ) + c* (xplus - vector[i]) + d * (xmark - vector[i]) # [l24]

            # == Mutation ==   
            for x in P : #                                      [l25]
                x.vector = x.vector + epsilon*x.velocity #      [l26]

            if Particle.best_fitness > criteria or it <=max_iteration_number: break # [l27]

        except Exception as e:
            pass
        finally : 
            # Export logs 
            logexport()
            break
    
    return Particle.fittest_solution # Vector representation

if __name__ == "__main__" : 

    # %% Example 1 
    
    def ANN2Vector(ANNstructure):
        pass

    def Informants(x,P, informants_number):
        # x Particle 
        # P set of Particle

        # 2nd idea suggested in Lecture 7 : random subset of P
        return random.sample(P, informants_number)
    
    def AssessFitness(x): # funct input

       """  # MSE 
        y # known
        mse = 0
        for k in range() :    
            err = y - x.ANN(x)
            mse += np.abs(err)

        mse = mse  / N 

        return mse """
       
       return " "
    

    swarmsize = 10 # between 10 - 100

    # Acceleration weights | Clue : sum = 4 
    alpha = 1 
    beta  = 1 # cognitive influence ; c1 = 1.49445  https://doi.org/10.1155/2020/8875922
    gamma = 1 # social influence ; c2 = 1.49445 
    delta = 1 

    # Jump size/ learning rate  | Clue : ? 
    epsi  = 0.3  # https://doi.org/10.1155/2020/8875922
    informants_number = 1 #arbitrary ; ? 
    
    max_iteration_number = 1200 # research paper  # https://doi.org/10.1155/2020/8875922
 
    
    #== PSO == 


    #== Test == 

