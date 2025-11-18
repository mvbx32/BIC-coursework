import os
import time 
import datetime

alpha =  0.9
beta =1.25
gamma = 1.25
delta = 1


now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
with open("launch_stats{}.txt".format(now),"w") as f :
    # Parameter 

    f.write("Try to delay the final convergence by reducing the global component")
    f.write("\n")
    for d_delta in [0.1,0.2,0.3,0.4,0.5] : 
        t0 = time.time()
        cmd = "py main.py  --IDE 0 --exp PierroInformantsLocalVsGlob --delta {} --gamma {} --beta {}".format(delta - d_delta, gamma + d_delta/2, beta + d_delta/2)
        f.write(cmd)
        f.write("\n")
        os.system(cmd)
        f.write(str(time.time()-t0))
        f.write("\n")
    # Parameter 
    for n_infor in [2,3,4,5,10] : 
        t0 = time.time()

        cmd = "py main.py --IDE 0 --exp PierroInformantsNum --informants_number {}".format(n_infor)

        f.write(cmd)
        f.write("\n")
        os.system(cmd)
        f.write(str(time.time()-t0))
        f.write("\n")