import numpy as np
import matplotlib.pyplot as plt

def pod(u_hf_train,nmax,sigma):
    #u_hf_train is the matrix of the high-fidelity (finite element) solutions for all the parameters in the training set
    Ntrain=u_hf_train.shape[1]
    # Compute covariance matrix via SVD (preferred for numerical stability)
    U, S, Vt = np.linalg.svd(u_hf_train, full_matrices=False)
    Z = U[:, :nmax]
    # S contains the singular values; 
    lambda_pod = S**2

    return Z, lambda_pod