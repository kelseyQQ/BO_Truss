import numpy as np

from names import Actions as act
from names import ParamNames as pn
from names import GlobNames as gn
from models.model import Model

from utils import proputils as pu

GROUPS = 'groups'
DOFS = 'dofs'
VALS = 'values'
INCR = 'dispIncr'
STRAINPATH = 'strainPath'


class LinearBCModel(Model):
    def take_action(self, action, params, globdat):
        if action == act.GETCONSTRAINTS:
            self._get_constraints(params, globdat)
        if action == act.ADVANCE:
            self._advance_step_constraints(params, globdat)


    def configure(self, props, globdat):
        self._groups = { 'left': {'xtype':'min'},
                         'right': {'xtype': 'max'},
                         'bottom': {'ytype': 'min'},
                         'top': {'ytype': 'max'} }

        self._dofs = ["dx", "dy"]
        self._ctol = 1.e-5
        self._strainPath = np.genfromtxt(props[STRAINPATH])
        if self._strainPath.ndim == 1:
            self._strainPath = self._strainPath.reshape(1, -1)
        self._strain = self._strainPath[0]

        coords = np.stack([node.get_coords() for node in globdat[gn.NSET]], axis=1)

        # find nodes in groups
        for g in self._groups.keys():
            group = np.array(globdat[gn.NGROUPS]['all'])
            gprops = self._groups[g]
            if isinstance(gprops,dict):
                for i, axis in enumerate(['xtype', 'ytype', 'ztype']):
                    if axis in gprops:
                        if gprops[axis] == 'min':
                            ubnd = np.min(coords[i, :]) + self._ctol
                            group = group[coords[i, group] < ubnd]
                        elif gprops[axis] == 'max':
                            lbnd = np.max(coords[i, :]) - self._ctol
                            group = group[coords[i, group] > lbnd]
                        elif gprops[axis] == 'mid':
                            mid = 0.5 * (np.max(coords[i, :]) - np.min(coords[i, :]))
                            lbnd = mid - self._ctol
                            ubnd = mid + self._ctol
                            group = group[coords[i, group] > lbnd]
                            group = group[coords[i, group] < ubnd]
                        else:
                            pass
            else:
                group = pu.parse_list(gprops,int)

            # store group
            globdat[gn.NGROUPS][g] = group
            print('InitModule: Created group', g, 'with nodes', group)

    def _get_constraints(self, params, globdat):
        ds = globdat[gn.DOFSPACE]

        doneNodes = []

        # loop over groups
        for group in self._groups:
            for node in globdat[gn.NGROUPS][group]:

                if node in doneNodes:
                    continue

                doneNodes.append(node)

                # get coords
                coords = globdat[gn.NSET][node].get_coords()

                # compute displacements
                dx = coords[0] * self._strain[0] + 0.5 * coords[1] * self._strain[2]
                dy = coords[1] * self._strain[1] + 0.5 * coords[0] * self._strain[2]

                # get dof indices and add constraints
                idofx, idofy = ds.get_dof(node, 'dx'), ds.get_dof(node, 'dy')
                params[pn.CONSTRAINTS].add_constraint(idofx, dx)
                params[pn.CONSTRAINTS].add_constraint(idofy, dy)


    def _advance_step_constraints(self, params, globdat):
        timestep = globdat[gn.TIMESTEP]
        if timestep > 1:
            self._dStrain = self._strainPath[timestep] - self._strainPath[timestep - 1]
        else:
            self._dStrain = self._strainPath[0]

        self._strain = self._strainPath[timestep]


def declare(factory):
    factory.declare_model('LinearBC', LinearBCModel)
