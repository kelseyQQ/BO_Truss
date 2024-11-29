import numpy as np

from names import Actions    as act
from names import ParamNames as pn
from names import GlobNames  as gn
from names import PropNames  as prn
from models.model import Model

ELEMENTS = 'elements'
KAPPA = 'kappa'
RHO = 'rho'
SHAPE = 'shape'
TYPE = 'type'
INTSCHEME = 'intScheme'
DOFTYPES = ['u']


class PoissonModel(Model):
    def take_action(self, action, params, globdat):
        # print('PoissonModel taking action', action)
        if action == act.GETMATRIX0:
            self._get_matrix(params, globdat)
        elif action == act.GETMATRIX2:
            self._get_mass_matrix(params, globdat)

    def configure(self, props, globdat):
        # Get basic parameter values
        self._kappa = float(props[KAPPA])
        self._rho = float(props.get(RHO,0))

        # Get shape and element info
        self._shape = globdat[gn.SHAPEFACTORY].get_shape(props[SHAPE][TYPE], props[SHAPE][INTSCHEME])
        egroup = globdat[gn.EGROUPS][props[ELEMENTS]]
        self._elems = [globdat[gn.ESET][e] for e in egroup]

        # Make sure the shape rank and mesh rank are identitcal
        if self._shape.global_rank() != globdat[gn.MESHRANK]:
            raise RuntimeError('PoissonModel: Shape rank must agree with mesh rank')

        # Get basic dimensionality info
        self._rank = self._shape.global_rank()
        self._ipcount = self._shape.ipoint_count()
        self._dofcount = len(DOFTYPES) * self._shape.node_count()

        # Create a new dof for every node and dof type
        nodes = np.unique([node for elem in self._elems for node in elem.get_nodes()])
        for doftype in DOFTYPES:
            globdat[gn.DOFSPACE].add_type(doftype)
            for node in nodes:
                globdat[gn.DOFSPACE].add_dof(node, doftype)

    def _get_matrix(self, params, globdat):
        kappa = self._kappa * np.identity(self._rank)

        for elem in self._elems:
            # Get the nodal coordinates of each element
            inodes = elem.get_nodes()
            idofs = globdat[gn.DOFSPACE].get_dofs(inodes, DOFTYPES)
            coords = np.stack([globdat[gn.NSET][i].get_coords() for i in inodes], axis=1)[0:self._rank, :]

            # Get the gradients and weights of each integration point
            grads, weights = self._shape.get_shape_gradients(coords)

            # Reset the element stiffness matrix
            elmat = np.zeros((self._dofcount, self._dofcount))

            for ip in range(self._ipcount):
                # Get the B matrix for each integration point
                B = np.zeros((self._rank, self._dofcount))
                B[:, :] = grads[:, :, ip].transpose()

                # Compute the element stiffness matrix
                elmat += weights[ip] * np.matmul(np.transpose(B), np.matmul(kappa, B))

            # Add the element stiffness matrix to the global stiffness matrix
            params[pn.MATRIX0][np.ix_(idofs, idofs)] += elmat

    def _get_mass_matrix(self, params, globdat):
        M = np.array([[self._rho]])

        for elem in self._elems:
            # Get the nodal coordinates of each element
            inodes = elem.get_nodes()
            idofs = globdat[gn.DOFSPACE].get_dofs(inodes, DOFTYPES[0:self._rank])
            coords = np.stack([globdat[gn.NSET][i].get_coords() for i in inodes], axis=1)[0:self._rank, :]

            # Get the shape functions and weights of each integration point
            sfuncs = self._shape.get_shape_functions()
            weights = self._shape.get_integration_weights(coords)

            # Reset the element mass matrix
            elmat = np.zeros((self._dofcount, self._dofcount))

            for ip in range(self._ipcount):
                # Get the N matrix for each integration point
                N = np.zeros((1, self._dofcount))
                N[0, :] = sfuncs[:, ip].transpose()

                # Compute the element mass matrix
                elmat += weights[ip] * np.matmul(np.transpose(N), np.matmul(M, N))

            # Add the element mass matrix to the global mass matrix
            params[pn.MATRIX2][np.ix_(idofs, idofs)] += elmat



def declare(factory):
    factory.declare_model('Poisson', PoissonModel)
