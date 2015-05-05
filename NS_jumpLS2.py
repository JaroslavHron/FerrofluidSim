#
# Numericke simulacie ferrokvapalin
#
# @author: Michal Habera 2015

from dolfin import *
import advsolver
import numpy as np

# vytvorim/importujem siet
d_ref = 32
mesh = UnitSquareMesh(d_ref, d_ref,"crossed")

# vytvorim subory na vysledky
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")
lsfile = File("results/ls.pvd")

# priestory funkcii
# skalarny
S = FunctionSpace(mesh, "Lagrange", 1)
# vektorovy, 2. radu
V = VectorFunctionSpace(mesh, "Lagrange", 2)
N = VectorFunctionSpace(mesh, "CG", 1)

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
p1 = Function(S)
# level set
ls0 = Function(S)
ls1 = Function(S)


# Hranicne podmienky
noslip = DirichletBC(V, (0, 0), "on_boundary && (x[0] > (1.0-DOLFIN_EPS) | x[1] < DOLFIN_EPS  )")

# casovo premenny vtok
p_in = Expression("sin(1.0*t)*x[0]*5", t=0.0)
inflow = DirichletBC(S, p_in, "x[1] > (1.0-DOLFIN_EPS)")

outflow = DirichletBC(S, 0, "x[1] < DOLFIN_EPS")

# spojim hranicne podmienky
bcu = [noslip]
bcp = [inflow, outflow]

# Konstanty

# fyzikalne
# viskozita
nu = Constant(0.05)

# casove
# N-S cyklus
T = 6
dt = 0.01


# Definujem bilinearne formy
# predbezne rychlostne pole
dt_ = Constant(dt)
f = Constant((0, 0))
F1 = (1 / dt_) * inner(u - u0, u_t) * dx + inner(grad(u0) * u0, u_t) * dx + nu * inner(grad(u), grad(u_t)) * dx - inner(
    f, u_t) * dx
a1 = lhs(F1)
L1 = rhs(F1)

# update tlaku
a2 = inner(grad(p), grad(p_t)) * dx
L2 = -(1 / dt_) * div(u1) * p_t * dx

# update rychlostneho pola
a3 = inner(u, u_t) * dx
L3 = inner(u1, u_t) * dx - dt_ * inner(grad(p1), u_t) * dx

# zostavim matice bil. foriem
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)


# boundaries
def bottom_boundary(x):
    return np.isclose(x[0], 0.0)


def top_boundary(x):
    return np.isclose(x[0], 1.0)


def left_boundary(x):
    return np.isclose(x[1], 0.0)


def right_boundary(x):
    return np.isclose(x[1], 1.0)

d = 0
dtau = pow(1/float(d_ref), 1+d)/2  # Olsson Kreiss --> dtau = ((dx)^(1+d))/2
eps = pow(1/float(d_ref), 1-d)/2  # Olsson Kreiss --> eps = ((dx)^(1-d))/2

# boundary conditions
bc_bottom = DirichletBC(S, Constant(0.0), bottom_boundary)
bc_top = DirichletBC(S, Constant(0.0), top_boundary)
bc_left = DirichletBC(S, Constant(0.0), left_boundary)
bc_right = DirichletBC(S, Constant(0.0), right_boundary)

phi_init = Expression("1/( 1+exp((sqrt((x[0]-0.7)*(x[0]-0.7)+(x[1]-0.7)*(x[1]-0.7))-0.2)/{0}))".format(eps))
ls0.assign(interpolate(phi_init, S))

## Casovy krok N-S rovnice
t = dt
lsp = plot(ls0)
while t < T + DOLFIN_EPS:
    # aktualizujem hranicnu podmienku tlaku v novom case
    p_in.t = t

    # pocitanie predbezneho rychlostneho pola
    begin("Pocitam predbezne rychlostne pole")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1)
    end()

    # korekcia tlaku
    begin("Pocitam tlakovu korekciu...")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2)
    end()

    # korekcia rychlosti
    begin("Pocitam rychlostnu korekciu...")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3)
    end()

    ls1 = advsolver.advsolve(mesh, S, N, d_ref, u1, ls0, _dtau=dtau, _norm_eps=0.000001,
                             _dt=dt, _t_end=dt, _bcs=[bc_bottom, bc_left, bc_top, bc_right])

    # ulozim vysledky
    ufile << u1
    pfile << p1
    lsfile << ls0

    # posuniem sa na dalsi casovy krok
    u0.assign(u1)
    ls0.assign(ls1)

    lsp.plot(ls1)
    plot(u1)
    t += dt
    print "t = ", t, " \n \n \n "










