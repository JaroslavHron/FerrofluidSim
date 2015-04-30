#
# Navier-Stokes solution
# implementation of Advection solver for jump Level-set
#
# Author: Michal Habera 2015 <michal.habera@gmail.com>

from dolfin import *
import advsolver
import numpy as np
from termcolor import colored
import time

# Mesh and function spaces
# mesh dimension
m = 100
mesh = UnitSquareMesh(m, m)
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

# advection constants
dt = 0.02
T = 2
# time independent vector field
r = Expression(("sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[1])", "-sin(pi*x[1])*sin(pi*x[1])*sin(2*pi*x[0])"))

# reinitialization constants
d = 0
dtau = pow(1/float(m), 1+d)/2  # Olsson Kreiss --> dtau = ((dx)^(1+d))/2
eps = pow(1/float(m), 1-d)/2  # Olsson Kreiss --> eps = ((dx)^(1-d))/2
Tau = 3
# stationary state of LS, definition
norm_eps = 0.00001


# boundaries
def bottom_boundary(x):
    return np.isclose(x[0], 0.0)


def top_boundary(x):
    return np.isclose(x[0], 1.0)


def left_boundary(x):
    return np.isclose(x[1], 0.0)


def right_boundary(x):
    return np.isclose(x[1], 1.0)


# boundary conditions
bc_bottom = DirichletBC(V, Constant(0.0), bottom_boundary)
bc_top = DirichletBC(V, Constant(0.0), top_boundary)
bc_left = DirichletBC(V, Constant(0.0), left_boundary)
bc_right = DirichletBC(V, Constant(0.0), right_boundary)

phi0 = Function(V)

# initial condition
print colored("Projecting initial level set...", "red")
start_ls_init = time.time()
phi_init = Expression("1/( 1+exp((sqrt((x[0]-0.3)*(x[0]-0.3)+(x[1]-0.3)*(x[1]-0.3))-0.2)/{0}))".format(eps))
phi0.assign(interpolate(phi_init, V))
print colored("Projection time: {:.3f}s.".format(time.time()-start_ls_init), "blue")

t = dt
while t < T:
    phi1 = advsolver.advsolve(mesh, V, W, m, r, phi0, _dt=dt, _t_end=dt,
                              _dtau=dtau, _tau_end=Tau, _norm_eps=norm_eps, _eps=eps,
                              _adv_scheme="implicit_euler",
                              _bcs=[bc_bottom, bc_right, bc_top, bc_left])
    phi0.assign(phi1)
    t += dt