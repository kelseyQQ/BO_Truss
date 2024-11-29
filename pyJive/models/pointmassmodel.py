import numpy as np

from names import Actions as act
from names import ParamNames as pn
from names import GlobNames as gn
from names import PropNames as prn
from models.model import Model
from utils.node import Node
from utils import proputils as pu

NODEGROUP = 'nodeGroup'
MASS = 'mass'
DOFTYPES = 'dofTypes'

class PointMassModel(Model):
    def take_action(self, action, params, globdat):
        if action == act.GETMATRIX2:
            self._get_mass_matrix(params, globdat)

    def configure(self, props, globdat):
        self._mass = float(props[MASS])
        self._nodes = globdat[gn.NGROUPS][props[NODEGROUP]]
        dofs = globdat[gn.DOFSPACE]
        if DOFTYPES in props:
            self._doftypes = pu.parse_list(props[DOFTYPES]) 
        else:
            self._doftypes = dofs.get_types()
            print(self._name, ' using all doftypes by default: ', self._doftypes)

        for doftype in self._doftypes:
            dofs.add_type( doftype )
            for node in self._nodes:
                dofs.add_dof(node, doftype)

    def _get_mass_matrix(self, params, globdat):
        for node in self._nodes:
            for doftype in self._doftypes:
                idof = globdat[gn.DOFSPACE].get_dof( node, doftype )
                params[pn.MATRIX2][idof, idof] += self._mass

def declare(factory):
    factory.declare_model('PointMass', PointMassModel)
