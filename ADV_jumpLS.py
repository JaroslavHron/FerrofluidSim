#
# Advection equation
#
# Author michal.habera@gmail.com
#

from dolfin import *
import numpy as np
import time
from termcolor import colored

start_global = time.time()

lsfile = File("results/ADV_jumpLS_ls.pvd")

# Mesh and function spaces
# mesh dimension
m = 30
mesh = UnitSquareMesh(m, m)
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

# advection constants
dt = 0.02
T = 1
# time independent vector field
r = Expression(("sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[1])", "-sin(pi*x[1])*sin(pi*x[1])*sin(2*pi*x[0])"))

# reinitialization constants
d = 0
dtau = pow(1/float(m), 1+d)/2  # Olsson Kreiss --> dtau = ((dx)^(1+d))/2
eps = pow(1/float(m), 1-d)/2  # Olsson Kreiss --> eps = ((dx)^(1-d))/2
Tau = 2
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

# trial and test functions
# level-set
phi = TrialFunction(V)
phi_t = TestFunction(V)

# normal field
n = Function(W)

# results functions
phi0 = Function(V)
phi1 = Function(V)
phi_reinit = Function(V)


# initial condition
print colored("Projecting initial level set...", "red")
start_ls_init = time.time()
phi_init = Expression("1/( 1+exp((sqrt((x[0]-0.3)*(x[0]-0.3)+(x[1]-0.3)*(x[1]-0.3))-0.2)/{0}))".format(eps))
phi0.assign(interpolate(phi_init, V))
print colored("Projection time: {:.3f}s.".format(time.time()-start_ls_init), "blue")


def get_adv_forms(_phi, _phi_t, _phi0, _schema="implicit_euler", _dt=0.1, _r=Constant((0, 0))):
    """ Returns tuple of multilinear forms for advection equation.
    Bilinear and linear form respectively.
    """
    if _schema == "implicit_euler":
        return (
            _dt*inner(dot(_r, grad(_phi)), _phi_t)*dx + inner(_phi, _phi_t)*dx,
            inner(_phi0, _phi_t)*dx
        )
    elif _schema == "explicit_euler":
        return (
            inner(_phi, _phi_t)*dx,
            inner(_phi0, _phi_t)*dx+_dt*inner(_phi0, dot(grad(_phi_t), _r))*dx
        )
    elif _schema == "crank_nicholson":
        return (
            inner(_phi, _phi_t)*dx-_dt/2.0*inner(_phi, dot(grad(_phi_t), _r))*dx,
            inner(_phi0, _phi_t)*dx+_dt/2.0*inner(_phi0, dot(grad(_phi_t), _r))*dx
        )
    else:
        raise ValueError("Unknown numerical scheme {scheme_name}!".format(scheme_name=_schema))


def get_reinit_forms(_phi, _phi_t, _phi0, _schema="implicit_euler", _dt=0.1, _eps=0.1, _n=Constant((0, 0))):
    """ Returns multilinear form for reinitialization equation.
    Nonlinear form F, solved by F == 0.
    """
    if _schema == "implicit_euler":
        return (
            1/_dt*inner(_phi-_phi0, _phi_t)*dx -
            inner(_phi*(1-_phi), dot(grad(_phi_t), _n))*dx +
            _eps*inner(grad(_phi), grad(_phi_t))*dx
        )
    else:
        raise ValueError("Unknown numerical scheme {scheme_name}!".format(scheme_name=_schema))


# bilinear form, time independent
(a, L) = get_adv_forms(phi, phi_t, phi0, "implicit_euler", dt, r)
A = assemble(a)

t = dt
while t < T + DOLFIN_EPS:
    print colored("Computing advection... , t = {}".format(t), "red")
    b = assemble(L)

    [bc.apply(A, b) for bc in [bc_bottom, bc_left, bc_top, bc_right]]
    solve(A, phi1.vector(), b)

    phi0.assign(phi1)

    print colored("Computing unit normal...", "red")
    grad_phi = grad(phi0)
    start = time.time()
    n.assign(project(grad_phi/sqrt(pow(norm_eps, 2)+dot(grad_phi, grad_phi)), W))
    print colored("Normal computation time: {:.3f}s.".format(time.time()-start), "blue")

    # reinitialization
    tau = dtau
    while tau < Tau + DOLFIN_EPS:
        start_reinit = time.time()
        print colored("Computing reinitialization, tau = {0}...".format(tau), "red")

        F = get_reinit_forms(phi_reinit, phi_t, phi0, "implicit_euler", dtau, eps, n)
        solve(F == 0, phi_reinit, [])

        phi0.assign(phi_reinit)

        print "Norm {0}".format(norm(assemble(F), "L2"))

        tau += dtau
        plot(phi_reinit)

        if norm(assemble(F), "L2") < 0.001:
            print colored("Reinitialization computation time: {:.3f}s".format(time.time()-start_reinit), "blue")
            break

    t += dt
    lsfile << phi0

print colored("***\n Total computational time: {:.2f}s \n***".format(time.time()-start_global), "red", "on_white")


