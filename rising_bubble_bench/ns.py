#
# Rising bubble bench
# geometry and phys params inspired by featflow.de
#
# @author: Michal Habera 2015

from dolfin import *
import advsolver as advsolver
import nssolver as nssolver
import numpy as np
import time

start_time = time.time()

# import mesh
d_ref = 20
mesh = Mesh("mesh.xml")

# create results files
ufile = File("results/velocity.xdmf")
pfile = File("results/pressure.xdmf")
lsfile = File("results/ls.xdmf")

# function spaces
# pressure
P = FunctionSpace(mesh, "CG", 1)
# velocity
U = VectorFunctionSpace(mesh, "CG", 2)
# level set
LS = FunctionSpace(mesh, "CG", 1)
# normal field
N = VectorFunctionSpace(mesh, "CG", 1)

# trial and test fctions
# velocity
u = TrialFunction(U)
u_t = TestFunction(U)
# pressure
p = TrialFunction(P)
p_t = TestFunction(P)
# level set
ls = TrialFunction(LS)
ls_t = TestFunction(LS)
# psi
psi = TrialFunction(LS)
psi_t = TestFunction(LS)

# result fctions
# velocity
u0 = Function(U)
u1 = Function(U)
# pressure
p0 = Function(P)
p1 = Function(P)
# level set
ls0 = Function(LS)
ls1 = Function(LS)
# psi
psi0 = Function(LS)
psi1 = Function(LS)

# boundaries
def bottom_boundary(x):
    return np.isclose(x[1], 0.0)


def top_boundary(x):
    return np.isclose(x[1], 2.0)


def left_boundary(x):
    return np.isclose(x[0], 0.0)


def right_boundary(x):
    return np.isclose(x[0], 1.0)


# constants
# dimensional ref values
rhoref = 1000.0  # kg/m3
nuref = 10.0  # kg/m3
lref = 0.5  # m
grav = 0.98 # m*s^-2
uref = sqrt(grav*lref)  # m/s

# nondim
# dyn viscosity
# out LS
nu1 = 10.0 / nuref
# in LS 
nu2 = 0.1 / nuref
# density
# out LS
rho1 = 1000.0 / rhoref
# in LS
rho2 = 1.0 / rhoref
# surface tension
sigma = 1.96 
# min of dens. GUERMOND MODEL
chi = 0.01 / rhoref

def rho(_ls):
    return rho1 + (rho2 - rho1) * _ls


def nu(_ls):
    return nu1 + (nu2 - nu1) * _ls

reynolds = rhoref * uref * lref / nuref
froude = uref / sqrt(lref * grav)
weber = rhoref * uref * uref * lref / sigma

print "########################"
print "Nondimensional numbers:"
print " Re={:.3f},\n We={:.3f},\n Fr={:.3f}".format(reynolds, weber, froude)
print "########################"


# N-S iteration
T = 3.0
dt = 0.01

# reinit
d = 0.08
dtau = pow(1 / float(d_ref), 1 + d) / 2.0# Olsson Kreiss --> dtau = ((dx)^(1+d))/2
eps = pow(1 / float(d_ref), 1 - d) / 2.0# Olsson Kreiss --> eps = ((dx)^(1-d))/2

# boundary conds
# merge bcs
# velocity
bcu = [DirichletBC(U, Constant((0.0, 0.0)), top_boundary), 
       DirichletBC(U, Constant((0.0, 0.0)), bottom_boundary),
       DirichletBC(U.sub(0), 0.0, left_boundary),
       DirichletBC(U.sub(0), 0.0, right_boundary)]
# pressure
bcp = []


# initial level-set
radius = 0.25
initx = 0.5
inity = 0.5
phiinit = Expression(
    "1/( 1+exp((sqrt((x[0]-{0})*(x[0]-{0})+(x[1]-{1})*(x[1]-{1}))-{2})/{3}))".format(initx, inity, radius, eps))
ls0.assign(interpolate(phiinit, LS))

# LS initialization
ls0, n = advsolver.advsolve(mesh, LS, N, d_ref, u0, ls0, _dtau=dtau, _eps=eps,
                            _dt=dt, _t_end=dt, _bcs=[],
                            _adv_scheme="implicit_euler")

# velocity initial
u0_init = Constant((0.0, 0.0))
u0.assign(project(u0_init, U))

# GUERMOND MODEL
psi0.assign(project(Constant(0.0), LS))

dt_ = Constant(dt)
# N-S time step
t = dt
while t <= T + DOLFIN_EPS:

    # advance LS
    ls1, n = advsolver.advsolve(mesh, LS, N, d_ref, u0, ls0,
                                _dtau=dtau, _eps=eps, 
                                _dt=dt, _t_end=dt, _bcs=[],
                                _adv_scheme="implicit_euler", _break_norm=0.001)
    
    u1, p1 = nssolver.nssolve(mesh, P, U, ls0, ls1, "olsson", froude, reynolds, weber,
                              rho2, rho1, nu2, nu1, sigma, u0, p0, n, dt_, bcu, bcp)

    ### Fenics demo Chorin
    # # Tentative velocity step
    # F1 = (1.0/dt)*rho(ls1)*inner(u - u0, u_t)*dx + \
    #     rho(ls0)*inner(grad(u0)*u0, u_t)*dx + \
    #     nu(ls1)/reynolds*inner(grad(u), grad(u_t))*dx - \
    #     20.0*inner(f, u_t)*dx + \
    #     1.0 / weber * sigma * inner(Ttens, grad(u_t)) * dx
    # a1 = lhs(F1)
    # L1 = rhs(F1)
    # 
    # # Pressure update
    # a2 = inner(grad(p), grad(p_t))*dx
    # L2 = -(1/dt)*div(u1)*p_t*dx
    # 
    # # Velocity update
    # a3 = inner(u, u_t)*dx
    # L3 = inner(u1, u_t)*dx - dt*inner(grad(p1), u_t)*dx
    # 
    # # Assemble matrices
    # A1 = assemble(a1)
    # A2 = assemble(a2)
    # A3 = assemble(a3)
    # 
    # # Compute tentative velocity step
    # begin("Computing tentative velocity")
    # b1 = assemble(L1)
    # [bc.apply(A1, b1) for bc in bcu]
    # solve(A1, u1.vector(), b1)
    # end()
    # 
    # # Pressure correction
    # begin("Computing pressure correction")
    # b2 = assemble(L2)
    # [bc.apply(A2, b2) for bc in bcp]
    # solve(A2, p1.vector(), b2)
    # end()
    # 
    # # Velocity correction
    # begin("Computing velocity correction")
    # b3 = assemble(L3)
    # [bc.apply(A3, b3) for bc in bcu]
    # solve(A3, u1.vector(), b3)
    # end()
    # Fenics demo Chorin END

    ### Guermond 2008
    # F1 = 1/dt*inner(0.5*(rho(ls0)+rho(ls1))*u, u_t)*dx \
    #     + 1/reynolds*nu(ls1)*inner(grad(u), grad(u_t))*dx \
    #     + inner(dot(rho(ls1)*u0, grad(u)), u_t)*dx \
    #     + 0.5*inner(div(rho(ls1)*u0)*u, u_t)*dx \
    #     + inner(grad(p0 + psi0), u_t)*dx \
    #     - 0.0*inner(f, u_t)*dx \
    #     - 1/dt*inner(rho(ls0)*u0, u_t)*dx \
    #     + 1.0 / weber * sigma * inner(Ttens, grad(u_t)) * dx
    # a1 = lhs(F1)
    # L1 = rhs(F1)
    # 
    # A1 = assemble(a1)
    # b1 = assemble(L1)
    # [bc.apply(A1, b1) for bc in bcu]
    # solve(A1, u1.vector(), b1, "gmres", "default")
    # 
    # F2 = inner(grad(psi), grad(psi_t))*dx \
    #     - chi/dt*inner(u1, grad(psi_t))*dx
    # 
    # a2 = lhs(F2)
    # L2 = rhs(F2)
    # 
    # A2 = assemble(a2)
    # b2 = assemble(L2)
    # [bc.apply(A2, b2) for bc in []]
    # solve(A2, psi1.vector(), b2, "gmres", "default")
    # 
    # p1.assign(project(p0 + psi1, P))
		

    ### Guermond 2008 END
    
    # advance LS
    # ls1, n = advsolver.advsolve(mesh, LS, N, d_ref, u1, ls0,
    #                            _dtau=dtau, _eps=eps, 
    #                            _dt=dt, _t_end=dt, _bcs=[],
    #                            _adv_scheme="implicit_euler")

    u1.rename("u", "velocity")
    p1.rename("p", "pressure")
    ls1.rename("ls", "level-set")

    lsfile << ls1
    ufile << u1
    pfile << p1
    
    u0.assign(u1)
    p0.assign(p1)
    ls0.assign(ls1)
    psi0.assign(psi1)
    plot(u0, interactive=False, key="u0", title="Velocity at time {}".format(t))
    # plot(p0, interactive=False)
    plot(ls0, key="ls0", title="Level-set at time {}".format(t))
    t += dt

    print "t = ", t, " \n"

print "Total time: {}".format(start_time - time.time())
