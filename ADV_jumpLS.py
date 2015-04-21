
#
# Advekcna rovnica
#
# Author michal.habera@gmail.com

from dolfin import *
import numpy as np

# Subory na ulozenie vysledkov
lsfile = File("ls.pvd")

# Mriezka a priestory funkcii
m = 20
mesh = UnitSquareMesh(m, m)
V = FunctionSpace(mesh, "Lagrange", 1)
W = VectorFunctionSpace(mesh, 'DG', 0)


# Konstanty advekcie
dt = 0.005
T = 0.1
r = Constant((0.0, 0.5))

# Konstanty reinicializacie
d = 0
dtau = pow(1/float(m), 1+d)/2  # Olsson Kreiss --> dtau = ((dx)^(1+d))/2
eps = pow(1/float(m), 1-d)/2  # Olsson Kreiss --> eps = ((dx)^(1-d))/2
Tau = 0.5


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
bc_left = DirichletBC(V, Constant(0.0),left_boundary)
bc_right = DirichletBC(V, Constant(0.0), right_boundary)


u = TrialFunction(V)
v = TestFunction(V)

# normalove pole na priestore vektorov W
n = Function(W)
# skalarne pole velkosti normal
n_norm = Function(V)

u0 = Function(V)
u1 = Function(V)

# Pociatocna podmienka
begin("Projekcia pociatocneho level setu")
u_0 = Expression("1/( 1+exp( ( sqrt( (x[0]-0.3)*(x[0]-0.3) + (x[1]-0.3)*(x[1]-0.3) )-0.2 )/{0}) )".format(eps))
u0.assign(interpolate(u_0, V))
end()

# Implicit Euler Variacna formulacia 
a = dt*inner(dot(r,grad(u)),v)*dx + inner(u,v)*dx
L = inner(u0,v)*dx

# Explicit Euler variacna formulacia
#a = inner(u,v)*dx
#L = inner(u0,v)*dx+dt*inner(u0,dot(grad(v),r))*dx

# Crack-Nicholson variacna formulacia
#a = inner(u,v)*dx-dt/2.0*inner(u,dot(grad(v),r))*dx
#L = inner(u0,v)*dx+dt/2.0*inner(u0,dot(grad(v),r))*dx

# bilinearna forma, nezavisi na casovom kroku
A = assemble(a)

t = dt
plot(u0, interactive=True)
# Hlavny casovy krok
while t < T + DOLFIN_EPS:
    begin("Pocitanie transportu")
    print("t = {}".format(t))
    b = assemble(L)
    [bc.apply(A, b) for bc in [bc_bottom, bc_left, bc_top, bc_right]]
    solve(A, u1.vector(), b)
    end()

    u0.assign(u1)

    # Reinicializacny casovy krok
    tau = dtau

    # vypocet normaloveho pola
    plot(u0, interactive=True)
    gu = grad(u0)

    cond = conditional(gt(sqrt(dot(gu,gu)), 0.01),-gu/sqrt(dot(gu,gu)), Constant((0,0)) )
    n.assign( project(cond, W) )
    plot(conditional(gt(sqrt(dot(gu,gu)), 0.01), sqrt(dot(gu,gu)), 0 ), interactive=True)

    while tau < Tau + DOLFIN_EPS:
        print("Pocitam reinicializaciu, tau = {0}".format(tau))

        # Crack Nicholson reinicializacia
        a_r = inner(u,v)*dx-dtau/2.0*inner(u,dot(grad(v),n))*dx+eps*dtau/2.0*inner(dot(grad(u),n),dot(grad(v),n))*dx+eps*dtau*inner(u*u0,dot(grad(v),n))*dx
        L_r = inner(u0,v)*dx+dtau/2.0*inner(u0,dot(grad(v),n))*dx-eps*dtau/2.0*inner(dot(grad(u0),n),dot(grad(v),n))*dx
        A_r = assemble(a_r)
        b_r = assemble(L_r)

        solve(A_r, u1.vector(), b_r)
        u0.assign(u1)
        tau += dtau
        plot(u0)

    #plot(n)
    t += dt

    #plot(u0)

    #try:
    #	response = raw_input('Prompt: ')
    #except:
    #	print ''

    lsfile << u1



