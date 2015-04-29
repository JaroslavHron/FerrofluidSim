#
# Navier-Stokes solution
# implementation of Advection solver for jump Level-set
#
# Author: Michal Habera 2015 <michal.habera@gmail.com>

from dolfin import *
import advsolve

d_ref = 50
mesh = UnitSquareMesh(d_ref, d_ref)
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

