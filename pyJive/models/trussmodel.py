import numpy as np

from names import Actions as act
from names import ParamNames as pn
from names import GlobNames as gn
from names import PropNames as prn
from models.model import Model
from utils.node import Node
from utils.xtable import XTable
from utils import proputils as pu

ELEMENTS = 'elements'
SUBTYPE = 'subtype'
NSECTIONS = 'nsections'
LINEAR = 'linear'
NONLIN = 'nonlin'
YOUNG = 'young'
DENSITY = 'density'
AREA = 'area'
SHAPE = 'shape'
INTSCHEME = 'intScheme'
DOFTYPES = ['dx', 'dy']


class TrussModel(Model):
    def take_action(self, action, params, globdat):

        if action == act.GETMATRIX0:
            self._get_matrix(params, globdat)
        elif action == act.GETMATRIX2:
            self._get_mass_matrix(params, globdat)
        elif action == act.GETMATRIXLB:
            self._get_matrix_lb(params, globdat)
        elif action == act.GETTABLE:
            if 'stress' in params[pn.TABLENAME]:
                self._get_stress_table(params, globdat)

    def configure(self, props, globdat):
        self._nsections = int(props[NSECTIONS])
        self._subtype = str(props[SUBTYPE])
        self._area = pu.parse_list(props[AREA], typ=float, length=self._nsections)
        self._young = pu.parse_list(props[YOUNG], typ=float, length=self._nsections)
        self._density = pu.parse_list(props[DENSITY], typ=float, length=self._nsections)

        self._shape = globdat[gn.SHAPEFACTORY].get_shape(props[SHAPE][prn.TYPE], props[SHAPE][INTSCHEME])
        egroup = globdat[gn.EGROUPS][props[ELEMENTS]]
        self._elems = [globdat[gn.ESET][e] for e in egroup]

        self._ipcount = self._shape.ipoint_count()
        self._dofcount = 2 * self._shape.node_count()
        self._strcount = 1

        nodes = np.unique([node for elem in self._elems for node in elem.get_nodes()])
        for doftype in DOFTYPES:
            globdat[gn.DOFSPACE].add_type(doftype)
            for node in nodes:
                globdat[gn.DOFSPACE].add_dof(node, doftype)

    def _get_matrix(self, params, globdat):
        for elem in self._elems:
            inodes = elem.get_nodes()
            isection = elem.get_family()
            idofs = globdat[gn.DOFSPACE].get_dofs(inodes, DOFTYPES)
            coords = np.stack([globdat[gn.NSET][i].get_coords() for i in inodes], axis=0)[:, :]
            EA = self._young[isection] * self._area[isection];

            d0 = coords[1, :] - coords[0, :]
            phi0 = np.arctan2(d0[1], d0[0])
            l_0 = np.linalg.norm(d0)
            coords1d = np.array([0, l_0])

            sfuncs = self._shape.get_shape_functions()
            grads, weights = self._shape.get_shape_gradients([coords1d])
            elmat = np.zeros((self._dofcount, self._dofcount))
            elfor = np.zeros(self._dofcount)

            ue = [globdat[gn.STATE0][i] for i in idofs]

            if self._subtype == LINEAR:
                for ip in range(self._ipcount):
                    dN = grads[:, 0, ip]

                    B = self._get_B_matrix(dN=dN, omega=phi0)
                    elmat += weights[ip] * EA * np.matmul(B.T, B)
                    elfor += np.matmul(elmat, ue)

            elif self._subtype == NONLIN:
                if self._shape.node_count() > 2:
                    raise NotImplementedError('nonlinear strain only implemented for 2-node element')

                d = d0 + ue[2:4] - ue[0:2]
                l = np.linalg.norm(d)
                phi = np.arctan2(d[1], d[0])

                for ip in range(self._ipcount):
                    N = sfuncs[:, ip]
                    dN = grads[:, 0, ip]

                    gamma = (np.cos(theta) * lsps - np.sin(theta) * lcps) / l_0
                    eps = l / l_0 - 1
                    nforce = EA * eps

                    B = self._get_B_matrix(dN=dN, omega=phi)
                    Kmat = weights[ip] * EA * np.matmul(B.T, B)

                    # TODO: add geometric part of stiffness matrix

                    elmat += Kmat
                    elfor += weights[ip] * nforce * B.ravel()

            params[pn.MATRIX0][np.ix_(idofs, idofs)] += elmat
            params[pn.INTFORCE][np.ix_(idofs)] += elfor

    def _get_matrix_lb(self, params, globdat):
        for elem in self._elems:
            inodes = elem.get_nodes()
            isection = elem.get_family()
            idofs = globdat[gn.DOFSPACE].get_dofs(inodes, DOFTYPES)
            coords = np.stack([globdat[gn.NSET][i].get_coords() for i in inodes], axis=0)[:, :]
            EA = self._young[isection] * self._area[isection]

            d0 = coords[1, :] - coords[0, :]
            phi = np.arctan2(d0[1], d0[0])
            l_0 = np.linalg.norm(d0)
            coords1d = np.array([0, l_0])

            sfuncs = self._shape.get_shape_functions()
            grads, weights = self._shape.get_shape_gradients([coords1d])
            elmatM = np.zeros((4, 4))
            elmatG = np.zeros((4, 4))

            for ip in range(self._ipcount):
                dN = grads[:, 0, ip]

                B = self._get_B_matrix(dN=dN, omega=phi)

                ue = [globdat[gn.STATE0][i] for i in idofs]
                evec = np.matmul(B, ue)
                svec = EA * evec

                # TODO: add geometric part of stiffness matrix

                elmatM += weights[ip] * EA * np.matmul(B.T, B)
                elmatG += 0

            params[pn.MATRIX0][np.ix_(idofs, idofs)] += elmatM
            params[pn.MATRIX1][np.ix_(idofs, idofs)] += elmatG

    def _get_mass_matrix(self, params, globdat):
        for elem in self._elems:
            inodes = elem.get_nodes()
            isection = elem.get_family()
            idofs_dx = globdat[gn.DOFSPACE].get_dofs(inodes, [DOFTYPES[0]])
            idofs_dy = globdat[gn.DOFSPACE].get_dofs(inodes, [DOFTYPES[1]])
            coords = np.stack([globdat[gn.NSET][i].get_coords() for i in inodes], axis=0)[:, :]
            rhoA = self._density[isection] * self._area[isection]
            
            d0 = coords[1, :] - coords[0, :]
            phi = np.arctan2(d0[1], d0[0])
            l_0 = np.linalg.norm(d0)
            coords1d = np.array([0, l_0])
            
            sfuncs = self._shape.get_shape_functions()
            grads, weights = self._shape.get_shape_gradients(coords1d)
            elmat_dx = np.zeros((2, 2))
            
            for ip in range(self._ipcount):
                N = np.array([sfuncs[:, ip]])
                elmat_dx += weights[ip] * rhoA * np.matmul(N.T,  N)
            
            params[pn.MATRIX2][np.ix_(idofs_dx, idofs_dx)] += elmat_dx
            params[pn.MATRIX2][np.ix_(idofs_dy, idofs_dy)] += elmat_dx
            
    def _get_stress(self, globdat, disp):
        stressmat = np.zeros((len(self._elems), self._strcount, 2))
        for i, elem in enumerate(self._elems):
            inodes = elem.get_nodes()
            isection = elem.get_family()
            idofs = globdat[gn.DOFSPACE].get_dofs(inodes, DOFTYPES)
            coords = np.stack([globdat[gn.NSET][j].get_coords() for j in inodes], axis=0)[:, :]
            EA = self._young[isection] * self._area[isection];

            d0 = coords[1, :] - coords[0, :]
            phi = np.arctan2(d0[1], d0[0])
            l_0 = np.linalg.norm(d0)
            coords1d = np.array([0, l_0])

            sfuncs = self._shape.get_shape_functions()
            grads, weights = self._shape.get_shape_gradients([coords1d])

            ue = [disp[j] for j in idofs]

            d = d0 + ue[2:4] - ue[0:2]

            # TODO: make robust implementation for N for more than 1 ip
            assert(self._ipcount == 1)
            ip = 0

            if self._subtype == LINEAR:
                dN = grads[:, 0, ip]

                B = self._get_B_matrix(dN=dN, omega=phi)
                eps = np.matmul(B, ue)
                nforce = EA*eps

            elif self._subtype == NONLIN:
                dN = grads[:, 0, ip]
                d = d0 + ue[2:4] - ue[0:2]
                l = np.linalg.norm(d)
                eps = l / l_0 - 1
                nforce = EA * eps

            stressmat[i,:,0] = [nforce];
            stressmat[i,:,1] = [nforce];

        return stressmat

    def _get_stress_table(self, params, globdat):
        table = params[pn.TABLE]

        # Convert the table to an XTable and store the original class
        cls_ = table.__class__
        table.__class__ = XTable

        # Add the columns of all stress components to the table
        jcols = table.add_columns(['N'])

        if gn.STATE0 in params:
            disp = params[gn.STATE0]
        else:
            disp = globdat[gn.STATE0]

        smat = self._get_stress(globdat, disp)

        for jcol in jcols:
            table.add_col_values(None, jcol, smat[:,jcol,:].flatten())

        # Convert the table back to the original class
        table.__class__ = cls_

        params[pn.TABLEWEIGHTS] = np.ones(table['N'].shape)


    def _get_B_matrix(self, dN, omega):
        B = np.zeros((self._strcount, self._dofcount))
        for inode in range(self._shape.node_count()):
            i = 2 * inode
            c = np.cos(omega) * dN[inode]
            s = np.sin(omega) * dN[inode]
            B[:, i:(i + 2)] = np.array([[c, s]])
        return B

def declare(factory):
    factory.declare_model('Truss', TrussModel)
