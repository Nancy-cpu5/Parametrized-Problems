from fenics import *
from dolfin import (
    FunctionSpace, Constant, DirichletBC, Expression,
    TrialFunction, TestFunction, Function, solve, dot, grad, dx
)
import numpy as np

def solve_pde(mesh,mux,muy,sigma,K):
    V = FunctionSpace(mesh, 'P', 1)

    # Boundary condition
    u_D = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u_D, boundary)

    # Source term
    f = Expression( 
        '1.0/sigma * exp(-(pow(x[0]-mux,2) + pow(x[1]-muy,2))/pow(sigma,2))',
        sigma=sigma, mux=mux, muy=muy, degree=1)

    # Define variational problem with K
    u = TrialFunction(V)
    v = TestFunction(V)
    a = K * dot(grad(u), grad(v)) * dx  #  K included here
    L = f * v * dx

    u_sol = Function(V)
    solve(a == L, u_sol, bc)

    return u_sol