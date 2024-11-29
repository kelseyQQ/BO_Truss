import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

from modules.module import Module
from modules.controlmodule import ControlModule
from utils.constrainer import Constrainer
from names import GlobNames as gn
from names import ParamNames as pn
from names import Actions as act

STOREMATRIX = 'storeMatrix'
STOREMASSMATRIX = 'storeMassMatrix'
DELTATIME = 'deltaTime'
THETA = 'theta'

class TrapezoidalModule(ControlModule):
    def init(self, props, globdat):
        super().init(props, globdat)
        myprops = props[self._name]
        self._dtime = float(myprops[DELTATIME])
        self._store_matrix = bool(eval(myprops.get(STOREMATRIX, 'False')))
        self._store_mass_matrix = bool(eval(myprops.get(STOREMASSMATRIX,'False')))
        self._theta = float(myprops.get('theta', 0.5)) 
    
    def run(self, globdat):
        dc = globdat[gn.DOFSPACE].dof_count()
        model = globdat[gn.MODEL]

        K = np.zeros((dc, dc))
        C = np.zeros((dc, dc))
        M = np.zeros((dc, dc))
        f_ext = np.zeros(dc)
        c = Constrainer()

        params = {pn.MATRIX0: K, pn.MATRIX1: C, pn.MATRIX2: M, pn.EXTFORCE: f_ext, pn.CONSTRAINTS: c}

        if self._step == 0:
            a_n = np.zeros(dc)
            a_ndot = np.zeros(dc)
        else:
            a_n = globdat[gn.STATE0]
            a_ndot = globdat[gn.STATE1]

        # Advance time step
        super().advance(globdat)
        model.take_action(act.ADVANCE, params, globdat)
        
        if self._step == 0:
            # Assemble K
            model.take_action(act.GETMATRIX0, params, globdat)

            # Assemble M
            model.take_action(act.GETMATRIX2, params, globdat)

            self._K = K
            self._M = M

        # Get constraints
        model.take_action(act.GETCONSTRAINTS, params, globdat)

        # Assemble external force
        model.take_action(act.GETEXTFORCE, params, globdat)

        if self._theta > 0:
            c0 = 1 / (self._theta * self._dtime)
            c1 = (1 - self._theta) / self._theta

            Khat = c0 * self._M + self._K
            fhat = f_ext + c0 * self._M @ a_n + c1 * self._M @ a_ndot

            Kc, fc = c.constrain(Khat, fhat)

            smat = sparse.csr_matrix(Kc)
            a_plus1 = linalg.spsolve(smat, fc)

            a_plus1dot = c0 * (a_plus1 - a_n) - c1 * a_ndot
        else:
            if self._step == 0:
                self._Ml = np.sum(self._M, axis=1)

            a_plus1 = a_n + self._dtime * a_ndot

            fhat = f_ext - self._K @ a_plus1

            Mc, fc = c.constraindiag(self._Ml,fhat)

            a_plus1dot = fc/Mc

        # Store solution in Globdat for next step
        
        globdat[gn.STATE0] = a_plus1
        globdat[gn.STATE1] = a_plus1dot

        # Optionally store stiffness matrix in Globdat
        if self._store_matrix:
            globdat[gn.MATRIX0] = K
        
        # Optionally get the mass matrix
        if self._store_mass_matrix:
            globdat[gn.MATRIX2] = M
        
        return super().run(globdat)
    
    def shutdown(self, globdat):
        pass

def declare(factory):
    factory.declare_module('Trapezoidal', TrapezoidalModule)
