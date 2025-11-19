#==================== main.py ============== #
from data import Data
from pso import * 
from tools import * 
from export_tools import *

# TODO : 



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
    "exp" : "TestNPY",
    "swarmsize": 10,
    "alpha": 0.9,
    "beta": 1.25,
    "gamma": 1.25,
    "delta": 1,
    "epsi": 0.5,
    "informants_number": 5,
    "max_iteration_number": 100,
    "AttemptNumber": 2,
    "ANN": [8, "input", 10, "sigmoid", 1, "linear"],
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
        type = float,
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


plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150


# === 0 — Best fitnesses ===
fig0, ax0 = plt.subplots()
ax0.set_title("SWARM best fitness")
#ax0.set_ylim([-0.5, 0.5])
ax0.set_xlabel("iteration")
ax0.set_ylabel("fitness")

figim, axim = plt.subplots()
axim.set_title("Relative improvement of the Best")
#ax0.set_ylim([-0.5, 0.5])
axim.set_xlabel("iteration")
axim.set_ylabel("relative improvement")

fig1, ax1 = plt.subplots()
ax1.set_title("Interparticle average distance")
ax1.set_xlabel("iteration")
ax1.set_ylabel("distance")

fig2, ax2 = plt.subplots(4,1)

"""
fig3, ax3 = plt.subplots()
ax3.set_title("Id of the best particle")
ax3.set_ylim([0, swarmsize + 1])
ax3.set_xlabel("iteration")
ax3.set_ylabel("id")

"""

if __name__ == "__main__" : 

    # -- Experiment details ----------------------------------------
    # Name of the experiment eg 

    params = parse_args()
    experiment_name = DEFAULTS["exp"] + build_experiment_label(params)
    operator = "M" # / "A" : first letter of the name of the h uman supervisor 
    # Description eg.
    description = 'VanillaPSO ' 
    # Goal 
    # Variables of interest : which variables are going to be modified
    variables_of_interest = {}
    #-- END of experiment details ----------------------------------
    print("=============",experiment_name,"======================")
   

    ###################                    PSO PARAMETERS                ###########################

    
    print(params)
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
    IDE                  = bool(params["IDE"])

    print(IDE) 
    AssessFitness = minusMAE
    Informants = k_nearest_neighboors

    w  = alpha
    C1 = beta
    C2 = gamma + delta
    #print((C1 + C2)/2  - 1)
    assert(1>w and w>((C1 + C2)/2  - 1) and (C1 + C2)/2  - 1 >=0 )

    # ////////////// Params to increment //////////////////////
    max_iteration_numberList = [max_iteration_number]
    swarmsizeList = [swarmsize]

    paramsList = [1,0.5,0.25,0.1]
    #__________________________________________________________________________________________________

    # -----------                   Creation of an Arborescence                  ---------------- #
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_name = f"{experiment_name.replace(' ', '_')}_{operator}_{now}"
    exp_path = os.path.join("experiments",experiment_name)
  

    print(exp_path)
    root_path, results_path = create_experiment_dir(experiment_name,operator= operator)
    save_experiment_details(root_path, experiment_name, operator,
                            "Evaluation of the impact of the number of iteration",
                            variables_of_interest)
    
   
    # -----------                                                                ---------------- #

    

    

    pso_id = 0 

   



    PSO_number = len(paramsList) * AttemptNumber
    PSOsGlobalVelMeanComponents = np.zeros((4,max_iteration_number,PSO_number))
    PSOsGLobalDistance = np.zeros((3,max_iteration_number,PSO_number))
    PSOsBestImprov = np.zeros((PSO_number,max_iteration_number))
    PSOsBestFitnesses = np.zeros((PSO_number,max_iteration_number))
    
    PSOsBestsId = [] #np.zeros((PSO_number,iter))

    step = 0
    for s, param2change in enumerate(paramsList) : 
        
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
            if  attempt == AttemptNumber-1 :
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
                        max_iteration_number = max_iteration_number, 
                        verbose = verbose, path = "/experiments/" + experiment_name, show = False ) 
            
            
            # ////////////// Params to increment //////////////////////
            pso.max_iteration_number= max_iteration_number
            pso.swarmsize = swarmsize
            pso.delta = param2change
            #  == Results ==============================================================
            best_solution, best_fitness, score_train, score_test, run_time = pso.train()  
            
            

            #___________________________________________________________________________
            

            
            # -- Monitoring  ------------------------------------------------------------
            Fitness.append(best_fitness)
            Train.append(score_train)
            Test.append(score_test)
            Time.append(run_time)

            """
            
                if attempt == 0 :
                params = { 
                            "ANNStructure" :  str(ANNStructure) , "swarmsize": str(swarmsizeList),"AssessFitness" : str(AssessFitness.__doc__), "max_iteration_number" : str(max_iteration_numberList),
                            "alpha" : str(alpha),"beta"  :  str(beta), "gamma" : str(gamma),  "delta" : str(delta),   "epsi"  : str(epsi),   
                            "Informants" : str(Informants),
                            "informants_number" :str(informants_number)}
                save_pso_params(pso_dir, params)
            
            """

            

            # -----------                                -------------                             ---------------- #

            # --  Exploitation  ----------------------------------------------------------------------------------------- #

            fitness_avg, score_train_avg,score_train_std,score_test_avg, score_test_std, time_avg =np.mean(Fitness), np.mean(Train),np.std(Train),np.mean(Test),np.std(Test),np.mean(Time)
            print(fitness_avg, score_train_avg,score_train_std,score_test_avg, score_test_std, time_avg)
            
            PSOsGlobalVelMeanComponents[:,:,step] = pso.globalVelMeanComponents.copy()
            PSOsGLobalDistance[:,:,step]  = np.array([pso.MinDistance, pso.AVGDistance, pso.MaxDistance])
            PSOsBestsId.append(pso.BestPaternityHistory)
            PSOsBestImprov[step,:] = pso.ImprovementOfBest[1:]
            PSOsBestFitnesses[step,:] = pso.BestFitnesses

         
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
                        pso.alpha,pso.beta,pso.gamma,pso.delta,pso.epsilon,
                        str(pso.ANN_structure).replace('[','').replace(']','').replace(',',' ')]
            row_data_str = [str(data) for data in row_data if (type(data)!= float) or (type(data)!= int)]
            
            append_results_to_excel(results_path, row_data)
            
            step +=1
            # == END of the attempts == #

        PSOsGlobalVelMeanComponentsAVG = np.mean(PSOsGlobalVelMeanComponents[:,:,s:s+AttemptNumber], axis = 2) # (compo, time, param1[attempt1, ..., attempt10] ... paramN[..] ) 
        PSOsGlobalVelMeanComponentsMAX = np.max(PSOsGlobalVelMeanComponents[:,:,s:s+AttemptNumber], axis = 2)
        PSOsGlobalVelMeanComponentsMIN = np.min(PSOsGlobalVelMeanComponents[:,:,s:s+AttemptNumber], axis = 2)

        PSOsBestFitnessesAVG = np.mean(PSOsBestFitnesses[s:s+AttemptNumber,:],axis = 0)
        PSOsBestFitnessesSTD = np.std(PSOsBestFitnesses[s:s+AttemptNumber,:],axis = 0)

        PSOsBestImprovAVG = np.mean(PSOsBestImprov[s:s+AttemptNumber,:],axis = 0)
        PSOsBestImprovSTD = np.std(PSOsBestImprov[s:s+AttemptNumber,:],axis = 0)
        iterations = range(max_iteration_number)
        ax0.plot(iterations,PSOsBestFitnessesAVG, linestyle ="solid", label=str(param2change) if len(paramsList)>0 else "") #


        axim.plot(iterations,PSOsBestImprovAVG, linestyle ="solid", label=str(param2change) if len(paramsList)>0 else "") #
        """
        axim.fill_between(iterations,
            np.array(PSOsBestImprovAVG)-np.array(PSOsBestImprovSTD),
            np.array(PSOsBestImprovAVG)+np.array(PSOsBestImprovSTD),
            color="#BBE4F8",
            label="Standard deviation")

        """
        

        #ax1.plot(range(max_iteration_number),np.max(PSOsGLobalDistance, axis = 2)[2,:] , label="max", linewidth= 0.8)
        ax1.plot(range(max_iteration_number),np.mean(PSOsGLobalDistance[:,:,s:s+AttemptNumber], axis = 2)[1,:], label=str(param2change) if len(paramsList)>0 else "", linewidth= 0.8) ; 
        #ax1.plot(range(max_iteration_number),np.min(PSOsGLobalDistance, axis = 2)[0,:], label="min", linewidth= 0.8)

        for c, component in enumerate(["inertial","local","social","global" ]) :
            ax2[c].title.set_fontsize(0.25 + 2)
            ax2[c].xaxis.label.set_fontsize(0.25)
            ax2[c].yaxis.label.set_fontsize(0.25)
            ax2[c].set_title("mean(|{} velocity|)".format(component))
            ax2[c].plot(range(max_iteration_number), PSOsGlobalVelMeanComponentsAVG[c,:], label=str(param2change) if len(paramsList)>0 else "", linewidth=0.8)
            #ax2[c].plot(range(max_iteration_number), PSOsGlobalVelMeanComponentsMIN[c,:], label="MIN", linewidth=0.8)
            #ax2[c].plot(range(max_iteration_number), PSOsGlobalVelMeanComponentsMAX[c,:], label="MAX", linewidth=0.8)

        """for ps in range(PSO_number):
            line = ax3.plot(range(max_iteration_number),PSOsBestsId[ps], "s")
            #ax3.plot(range(iter),PSOsBestsId[ps], linestyle ="solid", linewidth = 0.8, c = line[0].get_color())
            ax3.grid(True)
            ax3.set_yticks([i+1 for i in range(swarmsize)])
        """
        try : pass
            
            
            
        except Exception as e : 
            print("plot failed {}".format(e))
 

with open(os.path.join(root_path,"params.txt"),"w") as f : f.write("Params" +  str(paramsList) + "\n" + "Attempts " + str(AttemptNumber))
np.save(os.path.join(root_path,"{}PSOsGlobalVelMeanComponents".format(len(paramsList) if len(paramsList)>1 else "" ) ), PSOsGlobalVelMeanComponents , allow_pickle=True)
np.save(os.path.join(root_path,"{}PSOsGLobalDistance".format(len(paramsList) if len(paramsList)>1 else "" ) ), PSOsGLobalDistance , allow_pickle=True)   
np.save(os.path.join(root_path,"{}PSOsBestFitnesses".format(len(paramsList) if len(paramsList)>1 else "" ) ), PSOsBestFitnesses , allow_pickle=True)   
np.save(os.path.join(root_path,"{}PSOsBestImprov".format(len(paramsList) if len(paramsList)>1 else "" ) ), PSOsBestImprov , allow_pickle=True)  
np.save(os.path.join(root_path,"{}PSOsBestImprov".format(len(paramsList) if len(paramsList)>1 else "" ) ), PSOsBestImprov , allow_pickle=True)  

 




try : 
    
    if len(paramsList)> 0 : 
        ax0.legend(loc="best")
        ax1.legend(loc="best")
        ax2[c].legend(loc="best")
    fig0.tight_layout(pad=2)
    fig0.savefig( os.path.join(root_path, "Average of the best fitnesses.png"), bbox_inches='tight')
    # === 1 — Relative improvements of the SWARM ===

    axim.legend(loc="best")
    figim.tight_layout(pad=2)
    figim.savefig( os.path.join(root_path, "Relative improvements of the Bests.png"), bbox_inches='tight')

    # === 3 — Interparticle distances ===


    

    fig1.tight_layout(pad=2)
    fig1.savefig(os.path.join(root_path,"Interparticle_distances.png"), bbox_inches='tight')

    # == Velocities

    ax2[c].set_xlabel("iteration")
    
    fig2.tight_layout(pad=2)
    # --- SAVE IN THE EXPERIMENT FOLDER ---
    fig2.savefig(os.path.join(root_path,"PSOSVelocity_components.png"),
                bbox_inches='tight')

    # === 5 — Best particle ID ===
    """ fig3.tight_layout(pad=2)
    fig3.savefig(os.path.join(root_path,"Best_particle_id.png"), bbox_inches='tight')"""
   

    if bool(IDE) == True :
        
        plt.show()
   
except Exception as e : 
    print("plot failed {}".format(e))
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