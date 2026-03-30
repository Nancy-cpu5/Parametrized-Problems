import random

def params_builder(nsample, seed_par):
    #generate the parameters, both training and validation parameters. nsample is a name to indicate either training or validation parameters
    mux_min=-0.9
    mux_max=0.9
    muy_min=-0.9
    muy_max=0.9
    mux_sampled=list()
    muy_sampled=list()
    random.seed(seed_par)# for reproducibility of samples
    for ii in range(nsample):
            mux_sampled.append(random.uniform(mux_min,mux_max))
            muy_sampled.append(random.uniform(mux_min,mux_max))
    
    return mux_sampled, muy_sampled


