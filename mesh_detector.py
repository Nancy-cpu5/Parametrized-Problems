import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
from fenics import *
from scipy.integrate import dblquad

from FE.FE_solver import solve_pde
mux = 0.5
muy = 0.5
sigma=0.1

mesh1 = RectangleMesh(Point(-1, -1), Point(1, 1), 1000,1000 )
V = FunctionSpace(mesh1, 'P', 1)
u_exact=solve_pde(mesh1, mux=mux, muy=muy, sigma=sigma, K=1, N=1000)
mesh2=RectangleMesh(Point(-1, -1), Point(1, 1), 20,20)
V = FunctionSpace(mesh2, 'P', 1)
uh = solve_pde(mesh2, mux=mux, muy=muy, sigma=sigma, K=1, N=20)
test_points = [
    (0.5, 0.5),   # Center – peak
    (0.6, 0.5),   # Right of center
    (0.5, 0.6),   # Above center
    (0.4, 0.5),   # Left of center
    (0.5, 0.4),   # Below center
    (0.0, 0.0),   # Origin – smooth region
    (1.0, 0.0),   # Boundary – tests BC enforcement
    (0.0, 1.0),   
]
relative_errors = []
for x_choice,y_choice in test_points:
    u_exact_pointwise=u_exact(Point(x_choice,y_choice))
    uh_pointwise=uh(Point(x_choice,y_choice))
    rel_error = abs(u_exact_pointwise - uh_pointwise) / max(abs(u_exact_pointwise), 1e-8)
    relative_errors.append(rel_error)

print(f"Relative error = {rel_error:.4f}")
max_rel_error = max(relative_errors)
print(f"\nMaximum relative error: {max_rel_error:.4f}")







            