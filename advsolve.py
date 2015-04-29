#
# Advection equation solver for jump Level-set
#
# Author: Michal Habera 2015 <michal.habera@gmail.com>

from dolfin import *
import time
from termcolor import colored


def solve(_mesh, _ls_space, _normal_space, _size_ref, _v, _init_ls,
          _dt=0.1, _t_end=0.1, _dtau=0.1, _tau_end=10, _norm_eps=0.0001,
          _bcs=None):
    """

    Solve advection equation for jump Level-set

    :param _mesh: computational mesh
    :param _ls_space: level-set space living on _mesh
    :param _normal_space: space for normal vector field
    :param _size_ref: reference size/dimension of problem
    :param _v: velocity under which level-set is advected
    :param _init_ls: initial level-set
    :param _bcs:
    :return:
    """

    if not isinstance(_mesh, Mesh):
        raise ValueError("Mesh must be instance of Dolfin Mesh().")
    if not isinstance(_ls_space, FunctionSpace):
        raise ValueError("Level-set space must be instance of FunctionSpace().")
    if not isinstance(_normal_space, VectorFunctionSpace):
        raise ValueError("Normal space must be instance of VectorFunctionSpace().")
    if not isinstance(_size_ref, int):
        raise ValueError("Reference size must be integer.")
    if not isinstance(_v, Function):
        raise ValueError("Velocity vector field must be instance of Function().")
    if not isinstance(_init_ls, Function):
        raise ValueError("Initial level-set must be instance of Function().")

    print "\n Advection solver started... \n"

    resfile = File("results/adv_jump_ls.pvd")

    phi = TrialFunction(_ls_space)
    phi_t = TestFunction(_ls_space)

    phires = Function(_ls_space)
    phireinit = Function(_ls_space)
    n = Function(_normal_space)


    # bilinear form, advection step independent
    (bilinform, linform) = get_adv_forms(phi, phi_t, _init_ls, "implicit_euler", _dt, _v)
    asbilinform = assemble(bilinform)

    # advection time step
    t = _dt
    while t <= _t_end:
        print colored("Computing advection at t = {}s...".format(t), "red")

        aslinform = assemble(linform)
        [bc.apply(asbilinform, aslinform) for bc in _bcs]
        solve(asbilinform, phires.vector(), aslinform)

        print colored("Computing unit normal...", "red")
        gradphi = grad(phires)
        start = time.time()
        n.assign(project(gradphi/sqrt(pow(_norm_eps, 2)+dot(gradphi, gradphi)), _normal_space))
        print colored("Normal computation time: {:.3f}s.".format(time.time()-start), "blue")

        # reinitialization sub-time step
        tau = _dtau
        while tau <= _tau_end:
            startreinit = time.time()
            print colored("Computing reinitialization at tau = {}s...".format(tau), "red")

            F = get_reinit_forms(phireinit, phi_t, phires, "implicit_euler", _dtau, eps, n)
            solve(F == 0, phi_reinit, [])

            phi0.assign(phi_reinit)

            print "Norm {0}".format(norm(assemble(F), "L2"))

            tau += dtau
            plot(phi_reinit, title="Total:{:.2f}s/{:.2f}s, Reinit:{:.2f}s/{:.2f}s".format(t, T, tau, Tau))

            if norm(assemble(F), "L2") < 0.001:
                print colored("Reinitialization computation time: {:.3f}s".format(time.time()-startreinit), "blue")
                break

        t += dt
        lsfile << phi0

    print colored("***\n Total computational time: {:.2f}s \n***".format(time.time()-start_global), "red", "on_white")


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