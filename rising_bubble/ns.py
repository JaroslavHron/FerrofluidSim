#
# Rising bubble
#
# @author: Michal Habera 2015

from dolfin import *
import advsolver as advsolver
import numpy as np
import time

start_time = time.time()

# import mesh
d_ref = 15
mesh = Mesh("mesh.xml")

# create results files
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")
lsfile = File("results/ls.pvd")

# function spaces
# pressure
P = FunctionSpace(mesh, "CG", 1)
# velocity
U = VectorFunctionSpace(mesh, "CG", 2)
# level set
LS = FunctionSpace(mesh, "CG", 1)

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
nuref = 0.001  # kg/m3
lref = 0.1  # m
uref = 0.5  # m/s
# atmospheric pressure
patm = 0.0  # Pa

# nondim
# dyn viscosity
# out LS
nu1 = 0.001 / nuref
# in LS 
nu2 = 1.78 * pow(10, -5) / nuref
# density
# out LS
rho1 = 1000.0 / rhoref
# in LS
rho2 = 1.0 / rhoref
# surface tension
sigma = 0.07275
# grav accel
grav = 9.81


def rho(_ls):
    return rho1 + (rho2 - rho1) * _ls


def nu(_ls):
    return nu1 + (nu2 - nu1) * _ls

# pressure scale
pscale = lref / (rhoref * uref * uref)
# non dim numbers
reynolds = rhoref * uref * lref / nuref
froude = uref / sqrt(lref * grav)
weber = rhoref * uref * uref * lref / sigma

print "########################"
print "Nondimensional numbers:"
print " Re={:.3f},\n We={:.3f},\n Fr={:.3f}".format(reynolds, weber, froude)
print "########################"


# N-S iteration
T = 10.0
dt = 0.01

# reinit
d = 0
dtau = pow(1 / float(d_ref), 1 + d) / 2  # Olsson Kreiss --> dtau = ((dx)^(1+d))/2
eps = pow(1 / float(d_ref), 1 - d) / 2  # Olsson Kreiss --> eps = ((dx)^(1-d))/2

# boundary conditions
# bc_bottom = DirichletBC(S, Constant(0.0), bottom_boundary)
# bc_top = DirichletBC(S, Constant(0.0), top_boundary)
#bc_left = DirichletBC(S, Constant(0.0), left_boundary)
#bc_right = DirichletBC(S, Constant(0.0), right_boundary)

# boundary conds
# velocity
freeslipleft = DirichletBC(U.sub(0), 0.0, left_boundary)
freeslipright = DirichletBC(U.sub(0), 0.0, right_boundary)
freeslipbottom = DirichletBC(U.sub(1), 0.0, bottom_boundary)

bottom = DirichletBC(P, 0.0, bottom_boundary)
top = DirichletBC(P, patm, top_boundary)

# merge bcs
bcu = [freeslipleft, freeslipright, freeslipbottom]
bcp = [top]

radius = 0.3
initx = 0.5
inity = 0.5
phiinit = Expression(
    "1/( 1+exp((sqrt((x[0]-{0})*(x[0]-{0})+(x[1]-{1})*(x[1]-{1}))-{2})/{3}))".format(initx, inity, radius, eps))
ls0.assign(interpolate(phiinit, LS))

# LS init
ls0, n = advsolver.advsolve(mesh, LS, VectorFunctionSpace(mesh, "CG", 1), d_ref, u0, ls0, _dtau=dtau, _eps=eps,
                            _norm_eps=0.00001,
                            _dt=dt, _t_end=dt, _bcs=[],
                            _adv_scheme="implicit_euler")

# pressure init
pinit1 = Expression("6.5 - 3.25*x[1]")
pinit2 = pinit1
p_init = pinit1 + (pinit2 - pinit1) * ls0
p0.assign(project(p_init, P))

plot(p0, interactive=True)

dt_ = Constant(dt)
# N-S time step
t = dt
lsp = plot(ls0)
while t < T + DOLFIN_EPS:
    # advance LS
    ls1, n = advsolver.advsolve(mesh, FunctionSpace(mesh, "CG", 1), VectorFunctionSpace(mesh, "CG", 1), d_ref, u0, ls0,
                                _dtau=dtau, _eps=eps, _norm_eps=0.00001,
                                _dt=dt, _t_end=dt, _bcs=[],
                                _adv_scheme="implicit_euler")
    Ttens = (Identity(2) - outer(n, n)) * sqrt(pow(0.00001, 2) + dot(grad(ls1), grad(ls1)))

    # tentative velocity
    f = pow(1.0 / froude, 2) * rho(ls1) * Constant((0, -1.0))
    F1 = (1.0 / dt_) * inner(rho(ls1) * u - rho(ls0) * u0, u_t) * dx \
        - inner(dot(grad(u_t), u0), rho(ls1) * u) * dx \
        - inner(div(u_t), p0) * dx \
        + 1.0 / reynolds * inner(nu(ls1) * sym(grad(u)), grad(u_t)) * dx \
        + 1.0 / weber * sigma * inner(Ttens, grad(u_t)) * dx \
        - inner(f, u_t) * dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # pressure corr
    a2 = 1.0 / rho(ls1) * inner(grad(p), grad(p_t)) * dx
    L2 = -(1.0 / dt_) * inner(div(u1), p_t) * dx + 1.0 / rho(ls1) * inner(grad(p0), grad(p_t)) * dx

    # velocity corr
    a3 = inner(u, u_t) * dx
    L3 = inner(u1, u_t) * dx - dt_ * inner(grad(p1 - p0), u_t) / (rho(ls1)) * dx

    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    print "Computing tentative velocity..."
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, 'mumps')

    print "Computing pressure correction..."
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, 'mumps')

    print "Computing velocity correction..."
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, 'mumps')

    lsfile << ls0

    u0.assign(u1)
    p0.assign(p1)
    ls0.assign(ls1)
    plot(p0, interactive=False)
    lsp.plot(ls1)
    t += dt
    print "t = ", t, " \n"

print "Total time: {}".format(start_time - time.time())







