#
# Advekcna rovnica
#
# Author michal.habera@gmail.com
#
# TODO normaly cez FEM prenasobenim grad(phi)

from dolfin import *
import numpy as np

lsfile = File("results/ADV_jumpLS_ls.pvd")

# Mriezka a priestory funkcii
# rozmer mriezky
m = 100
mesh = UnitSquareMesh(m, m)
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, 'CG', 1)


# Konstanty advekcie
dt = 0.02
T = 1
# konstantne vektorove pole
r = Expression(("sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[1])", "-sin(pi*x[1])*sin(pi*x[1])*sin(2*pi*x[0])"))

# Konstanty reinicializacie
d = 0
dtau = pow(1/float(m), 1+d)/2  # Olsson Kreiss --> dtau = ((dx)^(1+d))/2
eps =  pow(1/float(m), 1-d)/2  # Olsson Kreiss --> eps = ((dx)^(1-d))/2
eps_init = 0.02
Tau = 2
norm_eps = 0.00001


# Hranice vypoc. oblasti
def bottom_boundary(x):
    return np.isclose(x[0], 0.0)


def top_boundary(x):
    return np.isclose(x[0], 1.0)


def left_boundary(x):
    return np.isclose(x[1], 0.0)


def right_boundary(x):
    return np.isclose(x[1], 1.0)


# Hranicne podmienky
bc_bottom = DirichletBC(V, Constant(0.0), bottom_boundary)
bc_top = DirichletBC(V, Constant(0.0), top_boundary)
bc_left = DirichletBC(V, Constant(0.0), left_boundary)
bc_right = DirichletBC(V, Constant(0.0), right_boundary)

# trial a test funkcie
# level-set
phi = TrialFunction(V)
phi_t = TestFunction(V)
phi_reinit = Function(V)

# normalove pole na priestore vektorov W
n = Function(W)

# funkcie vysledkov
phi0 = Function(V)
phi1 = Function(V)

# Pociatocna podmienka
print "Projekcia pociatocneho level setu"
phi_init = Expression("1/( 1+exp((sqrt((x[0]-0.3)*(x[0]-0.3)+(x[1]-0.3)*(x[1]-0.3))-0.2)/{0}))".format(eps_init))
phi0.assign(interpolate(phi_init, V))


# Implicit Euler Variacna formulacia 
a = dt*inner(dot(r, grad(phi)), phi_t)*dx + inner(phi, phi_t)*dx
L = inner(phi0, phi_t)*dx



# Explicit Euler variacna formulacia
#a = inner(u,v)*dx
#L = inner(u0,v)*dx+dt*inner(u0,dot(grad(v),r))*dx

# Crack-Nicholson variacna formulacia
#a = inner(u,v)*dx-dt/2.0*inner(u,dot(grad(v),r))*dx
#L = inner(u0,v)*dx+dt/2.0*inner(u0,dot(grad(v),r))*dx

# bilinearna forma, nezavisi na casovom kroku
A = assemble(a)

t = dt
### ADVEKCIA
while t < T + DOLFIN_EPS:
    print "Pocitanie transportu"
    print "t = {}".format(t)
    b = assemble(L)
    [bc.apply(A, b) for bc in [bc_bottom, bc_left, bc_top, bc_right]]
    solve(A, phi1.vector(), b)

    phi0.assign(phi1)

    ### REINICIALIZACIA
    tau = dtau

    # vypocet normaloveho pola
    plot(phi0)
    gu = grad(phi0)

    # cond = conditional(gt(phi0, 0.05), gu/sqrt(dot(gu, gu)), Constant((0, 0)))
    # cond = conditional(lt(phi0, 0.05), 1, 0)
    # condn = conditional(gt(sqrt(dot(gu, gu)), 0.1), gu/sqrt(dot(gu, gu)), Constant((0, 0)))
    print "Projektujem normalu"
    # n.assign(project(cond*condn, W))
    n.assign(project(gu/sqrt(pow(norm_eps, 2)+dot(gu, gu)), W))
    plot(n)

    while tau < Tau + DOLFIN_EPS:
        print "Pocitam reinicializaciu, tau = {0}".format(tau)

        # Crack Nicholson reinicializacia
        # a_r = inner(phi, phi_t)*dx-dtau/2.0*inner(phi, dot(grad(phi_t), n))*dx+ \
        #       eps*dtau/2.0*inner(grad(phi), grad(phi_t))*dx+eps*dtau*inner(phi*phi0, dot(grad(phi_t), n))*dx
        # L_r = inner(phi0, phi_t)*dx+dtau/2.0*inner(phi0, dot(grad(phi_t), n))*dx- \
        #       eps*dtau/2.0*inner(grad(phi0), grad(phi_t))*dx
        # A_r = assemble(a_r)
        # b_r = assemble(L_r)
        #
        # solve(A_r, phi1.vector(), b_r)

        # Implicit Euler reinicializacia
        F = 1/dtau*inner(phi_reinit-phi0, phi_t)*dx-inner(phi_reinit*(1-phi_reinit), dot(grad(phi_t), n))*dx+eps*inner(grad(phi_reinit), grad(phi_t))*dx
        solve(F == 0, phi_reinit, [])

        phi0.assign(phi_reinit)

        print "Norma {0}".format(norm(assemble(F), "L2"))

        tau += dtau
        plot(phi_reinit)

        if norm(assemble(F), "L2") < 0.001:
            break

    t += dt
    plot(phi0)
    lsfile << phi0




