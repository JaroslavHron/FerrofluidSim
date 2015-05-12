#
# Rising bubble
#
# @author: Michal Habera 2015
#

from dolfin import *
import advsolver as advsolver
import numpy as np
import time

start_time = time.time()

# vytvorim/importujem siet
d_ref = 20
mesh = UnitSquareMesh(d_ref, d_ref, "crossed")

mesh = Mesh("mesh.xml")
plot(mesh, interactive=True)

# vytvorim subory na vysledky
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")
lsfile = File("results/ls.pvd")

# priestory funkcii
# skalarny
S = FunctionSpace(mesh, "CG", 1)
# vektorovy, 2. radu
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# trial a test funnkcie
# rychlostne pole
u = TrialFunction(V)
u_t = TestFunction(V)
# tlak
p = TrialFunction(S)
p_t = TestFunction(S)
# level set
ls = TrialFunction(S)
ls_t = TestFunction(S)

# funkcie vysledkov
# rychlost
u0 = Function(V)
u1 = Function(V)
# tlak
p0 = Function(S)
p1 = Function(S)
# level set
ls0 = Function(S)
ls1 = Function(S)

# boundaries
def bottom_boundary(x):
    return np.isclose(x[1], 0.0)


def top_boundary(x):
    return np.isclose(x[1], 1.0)


def left_boundary(x):
    return np.isclose(x[0], 0.0)


def right_boundary(x):
    return np.isclose(x[0], 0.0)


# Hranicne podmienky
#noslip = DirichletBC(V, (0, 0), "(x[0] > (1.0-DOLFIN_EPS) | x[0] < DOLFIN_EPS | x[1] < (DOLFIN_EPS)   )")
freeslip = DirichletBC(V.sub(0), 0.0, "(x[0] > (1.0-DOLFIN_EPS) || x[0] < DOLFIN_EPS )")
freeslip2 = DirichletBC(V.sub(1), 0.0, "(x[1] < (DOLFIN_EPS) )")
# casovo premenny vtok
p_in = Expression("0", t=0.0)
p_init = Constant(0)
p0.assign(project(p_init, S))
bottom = DirichletBC(S, p_in, "x[1] < DOLFIN_EPS")

top = DirichletBC(S, 0, "x[1] > (1.0-DOLFIN_EPS)")

# spojim hranicne podmienky
bcu = [freeslip, freeslip2]
bcp = [top]

# Konstanty

rhoref = 1.0
nuref = 0.001
lref = 0.1
uref = 0.1

# fyzikalne
# viskozita okolia
nu2 = 0.001/nuref*1000
# viskozita v	nutri LS
nu1 = 2.2*pow(10, -5)/nuref*1000
# hustota okolia
rho2 = 1000.0/rhoref
# hustota vnutri LS
rho1 = 1.0/rhoref
# povrchove napatie
sigma = 0.07275
# gravitacne zrychlenie
grav = 9.81

reynolds = rhoref*uref*lref/nuref
froude = uref/sqrt(lref*grav)
weber = rhoref*uref*uref*lref/sigma

print "Re={}, We={}, Fr={}".format(reynolds, weber, froude)


# casove
# N-S cyklus
T = 10
dt = 0.01


d = 0
dtau = pow(1/float(d_ref), 1+d)/2  # Olsson Kreiss --> dtau = ((dx)^(1+d))/2
eps = pow(1/float(d_ref), 1-d)/2  # Olsson Kreiss --> eps = ((dx)^(1-d))/2

# boundary conditions
bc_bottom = DirichletBC(S, Constant(0.0), bottom_boundary)
bc_top = DirichletBC(S, Constant(0.0), top_boundary)
bc_left = DirichletBC(S, Constant(0.0), left_boundary)
bc_right = DirichletBC(S, Constant(0.0), right_boundary)

phi_init = Expression("1/( 1+exp((sqrt((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5))-0.25)/{0}))".format(eps))
ls0.assign(interpolate(phi_init, S))

# inicializacia
ls0, n = advsolver.advsolve(mesh, FunctionSpace(mesh, "CG", 1), VectorFunctionSpace(mesh, "CG", 1), d_ref, u0, ls0, _dtau=dtau, _eps=eps, _norm_eps=0.00001,
                            _dt=dt, _t_end=dt, _bcs=[bc_bottom, bc_left, bc_top, bc_right],
                            _adv_scheme="implicit_euler")


dt_ = Constant(dt)

def rho(_ls):
    return (rho1 + (rho2 - rho1)*_ls)

def nu(_ls):
    return (nu1 + (nu2 - nu1)*_ls)

## Casovy krok N-S rovnice
t = dt
lsp = plot(ls0)
Fsigma = Function(V)
while t < T + DOLFIN_EPS:
    # aktualizujem hranicnu podmienku tlaku v novom case
    p_in.t = t

    # ziskam LS v novom case
    ls1, n = advsolver.advsolve(mesh, FunctionSpace(mesh, "CG", 1), VectorFunctionSpace(mesh, "CG", 1), d_ref, u0, ls0, _dtau=dtau, _eps=eps, _norm_eps=0.00001,
                                _dt=dt, _t_end=dt, _bcs=[],
                                _adv_scheme="implicit_euler")
    # plot(n, interactive=True)
    Ttens = (Identity(2)-outer(n, n))*sqrt(pow(0.00001, 2) + dot(grad(ls1), grad(ls1)))



    # Definujem bilinearne formy
    # predbezne rychlostne pole
    dt_ = Constant(dt)
    #f = Fsigma/weber + rho(ls1)*Constant((0, 1))/pow(froude, 2)
    f = pow(1.0/froude, 2)*rho(ls1)*Constant((0, -1.0))
    #f = Constant((0, 0))
    F1 = (1.0 / dt_)* inner(rho(ls1)*u - rho(ls0)*u0, u_t) * dx - inner(dot(grad(u_t), u0), rho(ls1)*u)*dx - inner(div(u_t), p0)*dx + 1.0/reynolds * inner(nu(ls1)*sym(grad(u)), grad(u_t)) * dx - inner(f, u_t) * dx + 1.0/weber*sigma*inner(Ttens, grad(u_t))*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # update tlaku
    a2 = 1.0/rho(ls1)*inner(grad(p), grad(p_t)) * dx
    L2 = -(1.0 / dt_) * inner(div(u1), p_t) * dx + 1.0/rho(ls1)*inner(grad(p0), grad(p_t))*dx

    # update rychlostneho pola
    a3 = inner(u, u_t) * dx
    L3 = inner(u1, u_t) * dx - dt_ * inner(grad(p1 - p0), u_t)/(rho(ls1)) * dx
    
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # pocitanie predbezneho rychlostneho pola
    print "Pocitam predbezne rychlostne pole"
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, 'mumps')

    # korekcia tlaku
    print "Pocitam tlakovu korekciu..."
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, 'mumps')

    # korekcia rychlosti
    print "Pocitam rychlostnu korekciu..."
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, 'mumps')

    # ulozim vysledky
    ufile << u1
    pfile << p1
    lsfile << ls0

    # posuniem sa na dalsi casovy krok
    u0.assign(u1)
    p0.assign(p1)
    ls0.assign(ls1)
    plot(u0, interactive=False)
    lsp.plot(ls1)
    t += dt
    print "t = ", t, " \n"


print "Total time: {}".format(start_time-time.time())







