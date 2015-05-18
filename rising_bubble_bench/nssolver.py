# 
# Two-phase Navier-Stokes solver
#
# Author: Michal Habera 2015 <michal.habera@gmail.com>

from dolfin import *
import time


def nssolve(_mesh, _P, _U, _ls0, _ls1, _method, _froude, _reynolds, 
            _weber, _rhoin, _rhoout, _nuin, _nuout, _sigma, _u0, _p0, _n, _dt,
            _bcu, _bcp, _log_level=30):

    # start global time measurement
    nsstart = time.time()

    set_log_level(_log_level)

    print("################################ \n"
          " Navier-Stokes solver started... \n"
          "################################ \n")

    # defining trial and test functions
    u = TrialFunction(_U)
    u_t = TestFunction(_U)

    p = TrialFunction(_P)
    p_t = TestFunction(_P)

    # definig result functions
    u1 = Function(_U)
    p1 = Function(_P)

    # continuous density jump
    def rho(_ls):
        return _rhoout + (_rhoin - _rhoout) * _ls

    
    # continuos viscosity jump
    def nu(_ls):
        return _nuout + (_nuin - _nuout) * _ls
    
    # Olsson 2007, Conservative level set 2
    if _method == "olsson":
        
        print "--Olsson-- method chosen for Navier-Stokes solver"

        # tentative velocity
        f = pow(1.0 / _froude, 2) * rho(_ls1) * Constant((0, -1.0))
        F1 = (1.0 / _dt) * inner(rho(_ls1) * u - rho(_ls0) * _u0, u_t) * dx \
             - inner(dot(grad(u_t), _u0), rho(_ls1) * u) * dx \
             - div(u_t)*_p0 * dx \
             + 1.0 / _reynolds * inner(nu(_ls1) * sym(grad(u)), grad(u_t)) * dx \
             + 1.0 / _weber * _sigma * inner((Identity(2) - outer(_n, _n)) * 4.0 * _ls1*(1.0-_ls1), grad(u_t)) * dx \
             - inner(f, u_t) * dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # pressure corr
        a2 = 1.0 / rho(_ls1) * inner(grad(p), grad(p_t)) * dx
        L2 = -(1.0 / _dt) * inner(div(u1), p_t) * dx + 1.0 / rho(_ls1) * inner(grad(_p0), grad(p_t)) * dx

        #  velocity corr
        a3 = inner(u, u_t) * dx
        L3 = inner(u1, u_t) * dx - _dt * inner(grad(p1 - _p0), u_t) / (rho(_ls1)) * dx

        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)

        print "Computing tentative velocity..."
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in _bcu]
        solve(A1, u1.vector(), b1, 'mumps')

        print "Computing pressure correction..."
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in _bcp]
        solve(A2, p1.vector(), b2, "mumps")

        print "Computing velocity correction..."
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in _bcu]
        solve(A3, u1.vector(), b3, 'mumps')

        print("##############################\n"
              " Navier-Stokes solver end     \n"
              " Total execution time: {:.2f}s\n"
              "##############################\n".format(time.time() - nsstart))


        return u1, p1

