#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nancy Mokh
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
from plots.fe_solutions import plot_solution
from fenics import *
import matplotlib.pyplot as plt
#In the following code, we are doing the following:
#generate parameters, solve the Poisson 2D with Guassian source term for diff generated ransom fixed params
#get the matrix of solution for both valiudation and training for the two diff values of sigma
#Plot the solution and the solution profile
#Plot the decay of eigen values for the RBM
#Conclude


#create the parameters
from params.params_builder import params_builder
from FE.FE_solver import solve_pde

ntrain=200
nvalid=20

#training parameters to generate
seed_train=3
mux_train,muy_train=params_builder(ntrain, seed_train)

seed_valid=seed_train+1
mux_valid,muy_valid=params_builder(nvalid, seed_valid)
mesh = RectangleMesh(Point(-1, -1), Point(1, 1),80 , 80
                     )
plot(mesh)
plt.show()


def FE_run(base_folder, nsample, mux_sample, muy_sample,mesh,sigma):#nsample=ntrain or nvalid
    #Initializations
    u_hf=list()
    print(f"\n--- Solving for sigma = {sigma} ---")
    #create a folder for this specific sigma
    sigma_folder = os.path.join(base_folder, f'sigma_{sigma:.3f}')
    os.makedirs(sigma_folder, exist_ok=True)
    # Loop over all training parameters
    for i in range(nsample):
        mux = mux_sample[i]
        muy = muy_sample[i]
        print(f"Solving for mu ({mux:.4f}, {muy:.4f})")
        # Call solver with current mu and sigma
        uh = solve_pde(mesh,mux=mux, muy=muy, sigma=sigma, K=1)
        # Append solution vector
        uh.vector().get_local() 
        u_hf.append(uh)
            
        # Convert list of arrays to a snapshot matrix: (n_dofs x ntrain)
    u_hf_matrix = np.column_stack([u.vector().get_local() for u in u_hf])


        # Save parameters and solutions
    with open(sigma_folder+'mux_file.pickle', 'wb') as handle:
        pickle.dump(mux_sample, handle)

    with open(sigma_folder+'muy_file.pickle', 'wb') as handle:
        pickle.dump(muy_sample, handle)

    with open(sigma_folder+'u_hf.pickle', 'wb') as handle:
        pickle.dump(u_hf_matrix, handle)
    
    return u_hf_matrix,u_hf


# solve the pde for ntrain and store the uh train for sigma1 and sigma2
sigma1=0.5
sigma2=0.005
u_hf_train_sigma1,u_fenics_sigma1=FE_run('data_train_sigma1',ntrain, mux_train, muy_train,mesh,sigma1)
u_hf_train_sigma2, u_fenics_sigma2=FE_run('data_train_sigma2',ntrain, mux_train, muy_train,mesh,sigma2)

# solve the pde for nvalid and store the uh nvalid
u_hf_valid_sigma1=FE_run('data_valid_sigma1',nvalid, mux_valid, muy_valid,mesh,sigma1)
u_hf_valid_sigma2=FE_run('data_valid_sigma2',nvalid, mux_valid, muy_valid,mesh,sigma2)
#plot two solutions for the two values of sigma we have for a fixed mux and muy
indx_plt=0
plot_solution(u_fenics_sigma1[indx_plt],mesh,sigma1)
plot_solution(u_fenics_sigma2[indx_plt],mesh,sigma2)
#plot the solution profile for diff values of sigma for fixed mux and muy
#In order to plot the profile, lets get the ugamma and gamma point a fixed yi=muy
def get_profile(uh,mesh,muy_fixed):
    # Define resolution of sampling
    resolution = 500
    x_values = np.linspace(-1.0, 1.0, resolution)
    
    # Lists to store results
    u_Gamma = []
    Gamma_points = []

    # Evaluate solution at each x in [-1, 1]
    for x in x_values:
        try:
            value = uh(Point(x, muy_fixed))
            u_Gamma.append(value)
            Gamma_points.append(x)
        except RuntimeError:
            # Skip points outside the domain
            pass

    return u_Gamma, Gamma_points
# Get the fixed muy from sample
muy_fixed =muy_train[0]

# Evaluate the solution along y = muy_fixed
u_Gamma_sigma1, Gamma_points_sigma1 = get_profile(u_fenics_sigma1[0], mesh, muy_fixed=muy_fixed)
u_Gamma_sigma2, Gamma_points_sigma2 = get_profile(u_fenics_sigma2[0], mesh, muy_fixed=muy_fixed)

# Plot solution profile for sigma1
from plots.fe_solutions import plot_profile
plt.figure(figsize=(8, 4))
plot_profile(u_Gamma_sigma1, Gamma_points_sigma1, sigma=sigma1)
plt.title(f"Solution Profile (sigma = {sigma1:.3f})")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.tight_layout()
plt.show()
#plot solution profile for sigma2
plt.figure(figsize=(8, 4))
plot_profile(u_Gamma_sigma2, Gamma_points_sigma2, sigma=sigma2)
plt.title(f"Solution Profile (sigma = {sigma2:.3f})")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.tight_layout()
plt.show()



# Read the pickle files for both training and validation solutions
#with open('data_train/u_file.pickle', 'rb') as handle:
#    u_all_train = pickle.load(handle)  
#with open('data_valid/u_file.pickle', 'rb') as handle:
#    u_all_valid = pickle.load(handle)  



from RB.reduced_basis_builder import pod
Z_sigma1,lambda_sigma1=pod(u_hf_train_sigma1,ntrain,sigma1)
Z_sigma2,lambda_sigma2=pod(u_hf_train_sigma2,ntrain,sigma2)
#lets compute the relative sq error at sigma1
print("\n--- Computing Relative Squared Error for sigma1 ---")
#Compute the relative sq error of sigma1

# Pick one test solution from training set
uh_test = u_fenics_sigma1[0]  # First FEniCS solution
uh_test_vec = uh_test.vector().get_local()  # Convert to NumPy array

# Project onto reduced basis (L2 projection / POD reconstruction)
u_rbm_vec = Z_sigma1 @ (Z_sigma1.T @ uh_test_vec)

# Compute error vector
error_vec = uh_test_vec - u_rbm_vec

# Compute L2 norms (squared)
abs_error_L2_squared = np.dot(error_vec, error_vec)
exact_norm_L2_squared = np.dot(uh_test_vec, uh_test_vec)

# Compute relative squared error
rel_sq_error_sigma1 = abs_error_L2_squared / max(exact_norm_L2_squared, 1e-8)

print(f"Relative Squared Error (sigma1) = {rel_sq_error_sigma1:.6f}")
#Compute the relative sq error of sigma2

# Pick one test solution from training set
uh_test_sigma2 = u_fenics_sigma2[0]  # First FEniCS solution
uh_test_vec_sigma2 = uh_test.vector().get_local()  # Convert to NumPy array

# Project onto reduced basis (L2 projection / POD reconstruction)
u_rbm_vec_sigma2 = Z_sigma2 @ (Z_sigma2.T @ uh_test_vec_sigma2)

# Compute error vector
error_vec2 = uh_test_vec_sigma2 - u_rbm_vec_sigma2

# Compute L2 norms (squared)
abs_error_L2_squared_sigma2 = np.dot(error_vec2, error_vec2)
exact_norm_L2_squared_sigma2 = np.dot(uh_test_vec_sigma2, uh_test_vec_sigma2)

# Compute relative squared error
rel_sq_error_sigma2 = abs_error_L2_squared_sigma2 / max(exact_norm_L2_squared_sigma2, 1e-8)

print(f"Relative Squared Error (sigma2) = {rel_sq_error_sigma2:.6f}")
# Plot the decay of eigenvalues
# Plot the decay of eigenvalues
plt.figure(figsize=(8, 5))

plt.semilogy(lambda_sigma1/lambda_sigma1[0], 'o-', label=r'$\sigma=0.5$', linewidth=2, markersize=6)
plt.semilogy(lambda_sigma2/lambda_sigma2[0], 's-', label=r'$\sigma=0.005$', linewidth=2, markersize=6)

# Bigger title and labels
plt.title('POD Eigenvalue Decay for $\sigma$', fontsize=18, fontweight='bold')
plt.xlabel('Mode Number', fontsize=14)
plt.ylabel('Normalized Eigenvalue (log scale)', fontsize=14)

# Larger tick labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Grid and legend
plt.grid(True, which="both", linewidth=0.7)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()


#plt.savefig('figures/'+'pod'+'.pdf',bbox_inches='tight')
plt.show()
#Now we want to check the behaviour of  the square error for the two sigmas, while increasing the nmax

total_energy_sigma1 = np.sum(lambda_sigma1)
total_energy_sigma2 = np.sum(lambda_sigma2)

e_k_sigma1 = np.array([
    np.sum(lambda_sigma1[k:]) / total_energy_sigma1
    for k in range(ntrain)
])

e_k_sigma2 = np.array([
    np.sum(lambda_sigma2[k:]) / total_energy_sigma2
    for k in range(ntrain)
])

# Plot the decay of e_k (tail error)
plt.figure(figsize=(8, 5))

# Plot with thicker lines and bigger markers
plt.semilogy(range(1, ntrain + 1), e_k_sigma1, 'o-', 
             label=fr'$\sigma = {sigma1}$', linewidth=2, markersize=6)
plt.semilogy(range(1, ntrain + 1), e_k_sigma2, 's-', 
             label=fr'$\sigma = {sigma2}$', linewidth=2, markersize=6)

# Bigger fonts
plt.xlabel("Number of POD Modes $k$", fontsize=14)
plt.ylabel(r"Square error $e_k = \frac{\sum_{j \geq k} \lambda_j}{\sum_j \lambda_j}$", fontsize=14)
plt.title("Square Error Decay (POD Error) vs. Number of Modes", fontsize=18, fontweight="bold")

# Bigger tick labels
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

# Grid and legend
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()







