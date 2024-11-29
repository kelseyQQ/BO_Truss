# import warnings
from elasticmaterial import ElasticMaterial
from elasticmaterial import ANMODEL_PROP, SOLID, PLANE_STRESS, PLANE_STRAIN
import proputils as pu

from names import GlobNames as gn

import numpy as np
from math import exp
from math import isnan
import copy


STIFFS = 'prony_stiffs'
TIMES  = 'prony_times'

class MaxwellMaterial(ElasticMaterial):

    def __init__(self, rank):
        super().__init__(rank)

    def configure(self, props, globdat):
        self._globdat = globdat

        super().configure(props,globdat)

        self._stiffs = pu.parse_list(props[STIFFS],float)
        self._times  = pu.parse_list(props[TIMES],float)

        if len(self._stiffs) is not len(self._times):
            raise RuntimeError('MaxwellMaterial: stiffs and relaxation times must have the same size')

        self._oldtime = 0.0
        self._oldeps  = np.zeros(6)

    def update(self, strain, ipoint=None):
        dt = self._globdat[gn.TIME] - self._oldtime

        self._neweps = self._expand_strain(strain)
        deps = self._neweps - self._oldeps

        stiff_visco, sig_visco = self._prony[ipoint].update(deps,dt)

        stiff_inf, sig_inf = super().update(strain,ipoint)

        stiff  = stiff_inf + self._reduce_stiff(stiff_visco)
        stress = sig_inf + self._reduce_stress(sig_visco)

        return stiff, stress

    def commit(self, ipoint=None):
        self._oldtime = self._globdat[gn.TIME]
        self._oldeps = self._neweps

        if ipoint:
            self._prony[ipoint].commit()
        else:
            for prony in self._prony:
                prony.commit()

    def _reduce_stress(self, vec):
        if self._rank == 1:
            ret = np.zeros(1)
            ret[0] = vec[0]
        elif self._rank == 2:
            ret = np.zeros(3)
            ret[:2] = vec[:2]
            ret[2]  = vec[3]
        elif self._rank == 3:
            ret = np.zeros(6)
            ret = vec
        else:
            raise RuntimeError('MaxwellMaterial: unexpected rank')

        return ret

    def _expand_strain(self,vec):
        ret = np.zeros(6)
        
        if len(vec) == 1:
            ret[0] = vec[0]
        elif len(vec) == 3:
            eps_zz = -self._nu/(1.-self._nu)*sum(vec[:2]) if self._anmodel == PLANE_STRESS else 0.0

            ret[:2] = vec[:2]
            ret[2]  = eps_zz
            ret[3]  = vec[2]
        elif len(vec) == 6:
            ret = vec
        else:
            raise RuntimeError('MaxwellMaterial: unexpected strain vector size')

        return ret

    def _reduce_stiff(self,mat):
        if self._rank == 1:
            ret = np.zeros((1,1))
            ret[0,0] = mat[0,0]
        elif self._rank == 2:
            if self._anmodel == PLANE_STRESS:
                inv = np.linalg.inv(mat)
                tmp = self._select_2D_matrix(inv)
                ret = np.linalg.inv(tmp)
            elif self._anmodel == PLANE_STRAIN:
                ret = self._select_2D_matrix(mat)
            else:
                raise RuntimeError('MaxwellMaterial: unexpected analysis model')
        elif self._rank == 3:
            ret = np.zeros((6,6))
            ret = mat

        return ret

    def _select_2D_matrix(self,mat):
        ret = np.zeros((3,3))
        ret[:2,:2] = mat[:2,:2]
        ret[:2,2]  = mat[:2,3]
        ret[2,:2]  = mat[3,:2]
        ret[2,2]   = mat[3,3]

        return ret

    def create_material_points(self, npoints):
        self._prony = []

        for i in range(npoints):
            self._prony.append(self._PronySeries(self._stiffs,self._times,self._nu))
        print('Created ', npoints, ' integration point(s).\n')

    class _PronySeries:
        def __init__(self,stiffs,times,poisson):
            self._size = len(times)
            self._gstiffs = np.zeros(self._size)
            self._gtimes = np.zeros(self._size)
            self._kstiffs = np.zeros(self._size)
            self._ktimes = np.zeros(self._size)
            self._oldsig = np.zeros((6,self._size))
            self._newsig = np.zeros((6,self._size))

            for i in range(self._size):
                g = stiffs[i]/2./(1.+poisson)
                k = stiffs[i]/3./(1.-2.*poisson)

                self._gstiffs[i] = g
                self._gtimes[i] = times[i]*stiffs[i]/g

                self._kstiffs[i] = k
                self._ktimes[i] = times[i]*stiffs[i]/k
            
        def update(self, deps, dt):
            deps_vol = np.sum(deps[:3])/3.
            deps_dev = deps - deps_vol*np.array([1,1,1,0,0,0])

            stress = np.zeros(6)
            stiff  = np.zeros((6,6))

            for i in range(self._size):
                hist_vol = np.sum(self._oldsig[:3,i])/3.
                hist_dev = self._oldsig[:,i] - hist_vol*np.array([1,1,1,0,0,0])

                k = self._kstiffs[i] * (1.-np.exp(-dt/self._ktimes[i])) * self._ktimes[i] / dt
                g = self._gstiffs[i] * (1.-np.exp(-dt/self._gtimes[i])) * self._gtimes[i] / dt

                new_vol = np.exp(-dt/self._ktimes[i]) * hist_vol + 3.*k*deps_vol
                new_dev = np.exp(-dt/self._gtimes[i]) * hist_dev + 2.*g*deps_dev

                sig = new_dev + new_vol*np.array([1,1,1,0,0,0])

                stress += sig
                stiff[[0, 1, 2], [0, 1, 2]] += k + 4./3.*g
                stiff[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]] += k - 2./3.*g
                stiff[[3, 4, 5], [3, 4, 5]] += g

                self._newsig[:,i] = sig

            return stiff, stress 

        def commit(self):
            self._oldsig = np.copy(self._newsig)

