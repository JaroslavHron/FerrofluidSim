#
# Advection equation solver for jump Level-set
#
# Author: Michal Habera 2015 <michal.habera@gmail.com>
# TODO plot condition

from dolfin import *
import time


def advsolve(_mesh, _ls_space, _normal_space, _size_ref, _v, _init_ls,
             _dt=0.1, _t_end=0.1, _dtau=0.1, _tau_end=10, 
             _adv_scheme="implicit_euler", _reinit_scheme="implicit_euler",
             _eps=0.01, _bcs=None, _plot=True, _log_level=30, _bcs_reinit=None,
             _break_norm=0.0001):
    """

    Solve advection equation for jump Level-set
    Takes mashes, function spaces, initial level-set, bcs and some constants.

    :param _mesh: computational mesh
    :param _ls_space: level-set space living on _mesh
    :param _normal_space: space for normal vector field
    :param _size_ref: reference size/dimension of problem
    :param _v: velocity under which level-set is advected
    :param _init_ls: initial level-set
    :param _dt: advection time step size
    :param _t_end: advection time end, usually the same as _dt
    :param _dtau: reinitialization time step size
    :param _tau_end: reinitialization time end, usually some big number,
                     ends on stationary state condition
    :param _adv_scheme: advection time discretization scheme,
                        choices: implicit_euler, explicit_euler, crank_nicholson
    :param _reinit_scheme: reinitialization time discretization scheme,
                        choices: implicit_euler
    :param _eps: determines reinitialization diffusion term size
    :param _bcs: boundary conditions
    :param _plot: show plot
    :param _log_level: DOLFIN logging level, 10...50 = many...few details
    :return:

    """

    # start time measurement
    advtime = time.time()

    # dolfin logging level
    set_log_level(_log_level)

    if not isinstance(_mesh, Mesh):
        raise ValueError("Mesh must be instance of Dolfin Mesh().")
    if not isinstance(_ls_space, FunctionSpace):
        raise ValueError("Level-set space must be instance of FunctionSpace().")
    if not isinstance(_normal_space, VectorFunctionSpace):
        raise ValueError("Normal space must be instance of VectorFunctionSpace().")
    if not isinstance(_size_ref, int):
        raise ValueError("Reference size must be integer.")
    if not isinstance(_v, Function) and not isinstance(_v, Expression):
        raise ValueError("Velocity vector field must be instance of Function() or Expression().")
    if not isinstance(_init_ls, Function):
        raise ValueError("Initial level-set must be instance of Function().")

    print ("########################### \n"
           " Advection solver started...\n"
           "########################### \n")

    phi = TrialFunction(_ls_space)
    phi_t = TestFunction(_ls_space)

    phires = Function(_ls_space)
    phireinit = Function(_ls_space)
    n = Function(_normal_space)

    # bilinear form, advection step independent
    (bilinform, linform) = get_adv_forms(phi, phi_t, _init_ls, _adv_scheme, _dt, _v)
    asbilinform = assemble(bilinform)

    # advection time step
    t = 0.0 
    while t < _t_end:
        print "Computing advection at t = {}s...".format(t)

        aslinform = assemble(linform)
        [bc.apply(asbilinform, aslinform) for bc in _bcs]
        solve(asbilinform, phires.vector(), aslinform, "gmres", "default")

        print "Computing unit normal before reinitialization..."
        gradphi = grad(phires)
        start = time.time()
        n.assign(project(gradphi / sqrt(dot(gradphi, gradphi)), _normal_space))
        print "Normal computation time: {:.3f}s.".format(time.time() - start)
        # plot(n, key="adv normal", title="Normal before reinit")
        # reinitialization sub-time step
        tau = _dtau
        while tau <= _tau_end:
            startreinit = time.time()
            print "Computing reinitialization at tau = {}s...".format(tau)

            f = get_reinit_forms(phireinit, phi_t, phires, _reinit_scheme, _dtau, _eps, n)

            solve(f == 0, phireinit, bcs=[],
                  solver_parameters={"newton_solver": {'linear_solver': 'gmres', "preconditioner": "default"}},
                  form_compiler_parameters={"optimize": True})

            phires.assign(phireinit)
            tau += _dtau

            if norm(assemble(f), "L2") < _break_norm:
                print "Reinitialization computation time: {:.3f}s".format(time.time() - startreinit)
                break

        t += _dt

    print "Computing unit normal after reinitialization..."
    gradphi = grad(phires)
    start = time.time()
    n.assign(project(gradphi / sqrt(dot(gradphi, gradphi)), _normal_space))
    print "Normal computation time: {:.3f}s.".format(time.time() - start)

    print ("\n###########################  \n"
           " Advection solver end         \n"
           " Total execution time: {:.2f}s\n"
           "###########################  \n".format(time.time() - advtime))
    return phires, n


def get_adv_forms(_phi, _phi_t, _phi0, _schema="implicit_euler", _dt=0.1, _r=Constant((0, 0))):
    """
    Returns tuple of multilinear forms for advection equation.
    Bilinear and linear form respectively.
    """
    if _schema == "implicit_euler":
        return (
            _dt * inner(dot(_r, grad(_phi)), _phi_t) * dx + inner(_phi, _phi_t) * dx,
            inner(_phi0, _phi_t) * dx
        )
    elif _schema == "explicit_euler":
        return (
            inner(_phi, _phi_t) * dx,
            inner(_phi0, _phi_t) * dx + _dt * inner(_phi0, dot(grad(_phi_t), _r)) * dx
        )
    elif _schema == "crank_nicholson":
        return (
            inner(_phi, _phi_t) * dx - _dt / 2.0 * inner(_phi, dot(grad(_phi_t), _r)) * dx,
            inner(_phi0, _phi_t) * dx + _dt / 2.0 * inner(_phi0, dot(grad(_phi_t), _r)) * dx
        )
    else:
        raise ValueError("Unknown numerical scheme {scheme_name}!".format(scheme_name=_schema))


def get_reinit_forms(_phi, _phi_t, _phi0, _schema="implicit_euler", _dt=0.1, _eps=0.1, _n=Constant((0, 0))):
    """
    Returns multilinear form for reinitialization equation.
    Nonlinear form F, solved by F == 0.
    """
    if _schema == "implicit_euler":
        return (
            1 / _dt * inner(_phi - _phi0, _phi_t) * dx -
            inner(_phi * (1 - _phi), dot(grad(_phi_t), _n)) * dx +
            _eps * inner(grad(_phi), grad(_phi_t)) * dx
        )
    else:
        raise ValueError("Unknown numerical scheme {scheme_name}!".format(scheme_name=_schema))
