#==================== main.py ============== #
from data import Data
from pso import * 
from tools import * 
from export_tools import *

# TODO : 
# Add the activation functions + structure
# Handle the fails : save step by step + indicate on the file if failed
# Rigourous class tests
# modify the script to run 10 times each config and compute the average and the standard deviation

#Same result as in PSO example if we set only one PSO to check

# AI generating rules : Please strictly respect the following rules
# 1.Don't modify the existing code without permission
# 2.Each piece of code generated has to be clearly tagged as 'AI generated' mentionning the Model used 


if __name__ == "__main__" : 

    # -- Experiment details ----------------------------------------
    # Name of the experiment eg 
    experiment_name = 'VanillaPSOBenchmark1'
    operator = "M" # / "A" : first letter of the name of the human supervisor 
    # Description eg.
    description = 'VanillaPSO ' 
    # Goal 
    # Variables of interest : which variables are going to be modified
    variables_of_interest = {}
    #-- END of experiment details ----------------------------------


    ###################                    PSO PARAMETERS                ###########################
    
    AssessFitness = minusMAE
    ANNStructure = [8,'input',5,"sigmoid",1,'linear']
    swarmsize = 40
    alpha = 1 # inertia # 
    beta  = 1 # local
    gamma = 1 # informant
    delta = 1 # global #when 0 : no evolution
    epsi  = 0.5
    informants_number = 5
    Informants = randomParticleSet
    max_iteration_number = 1000
    AttemptNumber = 1

    # ////////////// Params to increment //////////////////////
    max_iteration_numberList = [1000]
    swarmsizeList = [5]
    #__________________________________________________________________________________________________

    # -----------                   Creation of an Arborescence                  ---------------- #
    root_path, results_path = create_experiment_dir(experiment_name,operator= operator)
    save_experiment_details(root_path, experiment_name, operator,
                             "Evaluation of the impact of the number of iteration",
                             variables_of_interest)
    # -----------                                                                ---------------- #
    
    pso_id = 0 
    for iter in max_iteration_numberList : 
        for swarmsize in swarmsizeList : 
            
            # --  Monitoring ------------------------------------------------------
            print("pso id {} itermax {} swarm {}  ".format(pso_id, iter, swarmsize))
            pso_id +=1
            pso_dir,models_dir = create_pso_dir(root_path, pso_id)
            Fitness = []
            Train = []
            Test =  []
            Time =  []

            
            for attempt in range(AttemptNumber):

                # $ - - For Debugging only - - $
                #np.random.seed(42)
                #random.seed(42)
                # $----------------------------$

                #############################################################################

                verbose = -1
                if iter == 1000 and attempt == 0 :
                    verbose = 0

                if attempt in [5] : 
                    print("Attempt ", attempt)
                
                pso= PSO(swarmsize, 
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
                            verbose = 0) 
                
                
                # ////////////// Params to increment //////////////////////
                pso.max_iteration_number= iter
                pso.swarmsize = swarmsize
                #  == Results ==============================================================
                best_solution, best_fitness, score_train, score_test, run_time = pso.train()  

                #___________________________________________________________________________
                

                
                # -- Monitoring  ------------------------------------------------------------
                Fitness.append(best_fitness)
                Train.append(score_train)
                Test.append(score_test)
                Time.append(run_time)
                if attempt == 0 :
                    params = { 
                                "ANNStructure" :  str(ANNStructure) , "swarmsize": str(swarmsizeList),"AssessFitness" : str(AssessFitness.__doc__), "max_iteration_number" : str(max_iteration_numberList),
                                "alpha" : str(alpha),"beta"  :  str(beta), "gamma" : str(gamma),  "delta" : str(delta),   "epsi"  : str(epsi),   
                                "Informants" : str(Informants),
                                "informants_number" :str(informants_number)

                            }
                    save_pso_params(pso_dir, params)
                # -----------                                -------------                             ---------------- #


            # --  Exploitation  ----------------------------------------------------------------------------------------- #

            fitness_avg, score_train_avg,score_train_std,score_test_avg, score_test_std, time_avg =np.mean(Fitness), np.mean(Train),np.std(Train),np.mean(Test),np.std(Test),np.mean(Time)
            print(fitness_avg, score_train_avg,score_train_std,score_test_avg, score_test_std, time_avg)
            
            # --  Export        ----------------------------------------------------------------------------------------- #
            
            row_data = [pso_id,fitness_avg, 
                        score_train_avg,
                        score_train_std, 
                        score_test_avg,
                        score_test_std,
                        AttemptNumber,
                        pso.max_iteration_number,
                        run_time,
                        pso.swarmsize,
                        pso.informants_number,
                        alpha,beta,gamma,delta,epsi,
                        str(pso.ANN_structure).replace('[','').replace(']','').replace(',',' ')]
            row_data_str = [str(data) for data in row_data if (type(data)!= float) or (type(data)!= int)]
            
            append_results_to_excel(results_path, row_data)

            # == END of the attempts == #

        # == END of the evaluations  == #



# arborescence synthesis 

   #/experiments
    #   /task2_M_2003_05_07
    #       ExperimentsDetails.txt
    #       results.csv # comparison of the performances of all the PSOs : PSO id, parameters of interest value, best fitness, MAE on the train set, MAE on the test set (from evaluation), execution_time
    #       /PSO_1 # first PSO executed
    #        ...
    #       /PSO_n 
    #           PSOparams.txt # params + performance 
    #           PSOsolution.txt # vector of the final fittest solution
    #           /models
    #               vector0.txt #save the vector of the best solution each k step
    
    #               ...
    #               vectork.txt
    #               ...
    #               vectorkn.txt
    #  
Particle.reset()
#==================== main.py  | END ==============#