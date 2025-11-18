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

# This function can parse the arguments from the command line.
# It would be useful to run the training script from the command line and change
# the hyperparameters without modifying the code.

import argparse


# ===========================
# DEFAULT PARAMETERS
# ===========================


# ============= GPT 5 ================== #

DEFAULTS = { 
    "exp" : "",
    "swarmsize": 5,
    "alpha": 0.9,
    "beta": 1.25,
    "gamma": 1,
    "delta": 1,
    "epsi": 0.5,
    "informants_number": 5,
    "max_iteration_number": 10,
    "AttemptNumber": 3,
    "ANN": [8, "input", 5, "sigmoid", 1, "linear"],
    "IDE": True,     # <-- default: use the parameters from code
}


def parse_args():
    parser = argparse.ArgumentParser(description="PSO Experiment")
    parser.add_argument("--exp", type=str, default=DEFAULTS["exp"])
    parser.add_argument("--swarmsize", type=int, default=DEFAULTS["swarmsize"])
    parser.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    parser.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    parser.add_argument("--delta", type=float, default=DEFAULTS["delta"])
    parser.add_argument("--epsi", type=float, default=DEFAULTS["epsi"])
    parser.add_argument("--informants_number", type=int, default=DEFAULTS["informants_number"])
    parser.add_argument("--max_iteration_number", type=int, default=DEFAULTS["max_iteration_number"])
    parser.add_argument("--AttemptNumber", type=int, default=DEFAULTS["AttemptNumber"])

    parser.add_argument(
        "--ANN",
        type=str,
        default=list(DEFAULTS["ANN"]),
        help="ANN structure expressed as python list",
    )

    parser.add_argument(
        "--IDE",
        action="store_true",
        default=DEFAULTS["IDE"],
        help="If True, ignore CLI overrides and use in-code values",
    )
    return vars(parser.parse_args())

def format_value(v):
    """Convert floats to nice strings: 0.5 → 0_5"""
    if isinstance(v, float):
        return str(v).replace(".", "_")
    return str(v)

def build_experiment_label(params):
    label_parts = []
    for key, default_val in DEFAULTS.items():
        if key == "ANN" or key == "IDE":
            continue  # skip these

        val = params[key]
        if val != default_val:
            label_parts.append(f"{key.capitalize()}{format_value(val)}")

    if label_parts:
        return "_".join(label_parts)
    else:
        return "Default"

# ============= END GPT 5 ================== #

if __name__ == "__main__" : 

    # -- Experiment details ----------------------------------------
    # Name of the experiment eg 

    params = parse_args()
    experiment_name = 'PlotDebug' + build_experiment_label(params)
    operator = "M" # / "A" : first letter of the name of the h uman supervisor 
    # Description eg.
    description = 'VanillaPSO ' 
    # Goal 
    # Variables of interest : which variables are going to be modified
    variables_of_interest = {}
    #-- END of experiment details ----------------------------------
    print("=============",experiment_name,"======================")
   

    ###################                    PSO PARAMETERS                ###########################

    
    
    ANNStructure         = params["ANN"]
    swarmsize            = params["swarmsize"]
    alpha                = params["alpha"]       # inertia
    beta                 = params["beta"]        # local component
    gamma                = params["gamma"]       # informant component
    delta                = params["delta"]       # global component
    epsi                 = params["epsi"]        # random exploration / mutation factor
    informants_number    = params["informants_number"]

    max_iteration_number = params["max_iteration_number"]
    AttemptNumber        = params["AttemptNumber"]

    AssessFitness = minusMAE
    Informants = k_nearest_neighboors

    w  = alpha
    C1 = beta
    C2 = gamma + delta
    print((C1 + C2)/2  - 1)
    assert(1>w and w>((C1 + C2)/2  - 1) and (C1 + C2)/2  - 1 >=0 )

    # ////////////// Params to increment //////////////////////
    max_iteration_numberList = [max_iteration_number]
    swarmsizeList = [swarmsize]
    #__________________________________________________________________________________________________

    # -----------                   Creation of an Arborescence                  ---------------- #
    root_path, results_path = create_experiment_dir(experiment_name,operator= operator)
    save_experiment_details(root_path, experiment_name, operator,
                            "Evaluation of the impact of the number of iteration",
                            variables_of_interest)
    # -----------                                                                ---------------- #


    

    pso_id = 0 

    

    for iter in max_iteration_numberList : 
        PSO_number = len(swarmsizeList) * AttemptNumber
        PSOsGlobalVelMeanComponents = np.zeros((4,iter,PSO_number))
        PSOsGLobalDistance = np.zeros((3,iter,PSO_number))
        PSOsBestImprov = np.zeros((PSO_number,iter))
      
        PSOsBestsId = [] #np.zeros((PSO_number,iter))

        step = 0
        for s, swarmsize in enumerate(swarmsizeList) : 
            
            # --  Monitoring ------------------------------------------------------
            print("pso id {} itermax {} swarm {}  ".format(pso_id, iter, swarmsize))
            pso_id +=1
            pso_dir,models_dir = create_pso_dir(root_path, pso_id)
            Fitness = []
            Train = []
            Test =  []
            Time =  []

            
            for ai, attempt in enumerate(range(AttemptNumber)):

                # $ - - For Debugging only - - $
                #np.random.seed(42)
                #random.seed(42)
                # $----------------------------$

                #############################################################################

                verbose = -1
                if iter == max(max_iteration_numberList) and attempt == AttemptNumber-1 :
                    verbose = 0

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
                            max_iteration_number = iter, 
                            verbose = verbose, path = "/experiments/" + experiment_name, show = False ) 
                
                
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
                
                PSOsGlobalVelMeanComponents[:,:,step] = pso.globalVelMeanComponents.copy()
                PSOsGLobalDistance[:,:,step]  = np.array([pso.MinDistance, pso.AVGDistance, pso.MaxDistance])
                PSOsBestsId.append(pso.BestPaternityHistory)
                PSOsBestImprov[step,:] = pso.ImprovementOfBest[1:]
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
                print(s + ai, step)
                step +=1
                # == END of the attempts == #



        PSOsGlobalVelMeanComponentsAVG = np.mean(PSOsGlobalVelMeanComponents, axis = 2)
        PSOsGlobalVelMeanComponentsMAX = np.max(PSOsGlobalVelMeanComponents, axis = 2)
        PSOsGlobalVelMeanComponentsMIN = np.min(PSOsGlobalVelMeanComponents, axis = 2)
        
        PSOsBestImprovAVG = np.mean(PSOsBestImprov,axis = 0)
        PSOsBestImprovSTD = np.std(PSOsBestImprov,axis = 0)
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 120
        plt.rcParams["savefig.dpi"] = 150
        

         # === 1 — Relative improvements of the SWARM ===
        fig0, ax0 = plt.subplots()
        ax0.set_title("Relative improvements of the Bests")
        #ax0.set_ylim([-0.5, 0.5])
        ax0.set_xlabel("iteration")
        ax0.set_ylabel("relative improvement")

        iterations = range(max_iteration_number)

        ax0.plot(iterations,PSOsBestImprovAVG, 'o', c="#008fd5", label="Average")
        ax0.fill_between(iterations,
                        np.array(PSOsBestImprovAVG)-np.array(PSOsBestImprovSTD),
                        np.array(PSOsBestImprovAVG)+np.array(PSOsBestImprovSTD),
                        color="#BBE4F8",
                        label="Standard deviation")
        for ps in range(PSO_number) : 
            line = ax0.plot(iterations,PSOsBestImprov[ps,:], '.')[0]
            ax0.plot(iterations,PSOsBestImprov[ps,:], linestyle = 'solid',linewidth = 0.3,color =  line.get_color())
        ax0.legend(loc="best")
        fig0.tight_layout(pad=2)
        fig0.savefig( "Relative improvements of the Bests.png", bbox_inches='tight')

        # === 3 — Interparticle distances ===
        fig1, ax1 = plt.subplots()
        ax1.set_title("Interparticle distances")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("distance")
        ax1.plot(range(iter),np.max(PSOsGLobalDistance, axis = 2)[2,:] , label="max", linewidth= 0.8)
        ax1.plot(range(iter),np.mean(PSOsGLobalDistance, axis = 2)[0,:], label="average", linewidth= 0.8)
        ax1.plot(range(iter),np.min(PSOsGLobalDistance, axis = 2)[0,:], label="min", linewidth= 0.8)
        ax1.legend(loc="best")

        fig1.tight_layout(pad=2)
        fig1.savefig("Interparticle_distances.png", bbox_inches='tight')

        fig2, ax2 = plt.subplots(4,1)

        for c, component in enumerate(["inertial","local","social","global" ]) :
            ax2[c].title.set_fontsize(0.25 + 2)
            ax2[c].xaxis.label.set_fontsize(0.25)
            ax2[c].yaxis.label.set_fontsize(0.25)
            ax2[c].set_title("Mean(|{} velocity |)".format(component))
            ax2[c].plot(range(iter), PSOsGlobalVelMeanComponentsAVG[c,:], label="MEAN", linewidth=0.8)
            ax2[c].plot(range(iter), PSOsGlobalVelMeanComponentsMIN[c,:], label="MIN", linewidth=0.8)
            ax2[c].plot(range(iter), PSOsGlobalVelMeanComponentsMAX[c,:], label="MAX", linewidth=0.8)
        ax2[c].set_xlabel("iteration")
        ax2[c].legend(loc="best")
        fig2.tight_layout(pad=2)
        # --- SAVE IN THE EXPERIMENT FOLDER ---
        fig2.savefig("PSOSVelocity_components.png",
                    bbox_inches='tight')

        


         # === 5 — Best particle ID ===
        fig3, ax3 = plt.subplots()
        ax3.set_title("Id of the best particle")
        ax3.set_ylim([0, swarmsize + 1])
        ax3.set_xlabel("iteration")
        ax3.set_ylabel("id")

        for ps in range(PSO_number):
            line = ax3.plot(range(iter),PSOsBestsId[ps], "s")
            #ax3.plot(range(iter),PSOsBestsId[ps], linestyle ="solid", linewidth = 0.8, c = line[0].get_color())
            ax3.grid(True)
            ax3.set_yticks([i+1 for i in range(swarmsize)])

        fig3.tight_layout(pad=2)
        fig3.savefig("Best_particle_id.png", bbox_inches='tight')

        plt.show()
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