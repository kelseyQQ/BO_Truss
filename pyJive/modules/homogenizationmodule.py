import numpy as np

from modules.module import Module
from names import GlobNames as gn

SMALL = 1e-6
LARGE = 1e+6


class HomogenizationModule(Module):

    def init(self, props, globdat):

        # define boundary groups
        boundaryGroups = ['left', 'right', 'bottom', 'top']

        # get dimensions of RVE
        x_min, y_min = LARGE, LARGE
        x_max, y_max = SMALL, SMALL
        nodeSet = globdat[gn.NSET]

        for group in boundaryGroups:
            for node in globdat[gn.NGROUPS][group]:
                coords = nodeSet[node].get_coords()
                x_min = np.minimum(x_min, coords[0])
                x_max = np.maximum(x_max, coords[0])
                y_min = np.minimum(y_min, coords[1])
                y_max = np.maximum(y_max, coords[1])

        self._Drve = np.array([x_max - x_min, y_max - y_min])

        mydata = {'stresses': []}

        globdat[self._name] = mydata

    def run(self, globdat):

        mydata = globdat[self._name]

        if globdat[gn.ACCEPTED]:

            # init stresses
            sigma = np.zeros((3))

            # get boundary groups
            groups = globdat['lodi']

            # sigma_xx
            sigma[0] = groups['right']['load']['dx'][-1] / self._Drve[1]

            # sigma_yy
            sigma[1] = groups['top']['load']['dy'][-1] / self._Drve[0]

            # sigma_xy
            sigma[2] = groups['right']['load']['dy'][-1] / self._Drve[0]

            # store in globdat
            mydata['stresses'].append(sigma)

            return 'ok'

    def shutdown(self, globdat):
        pass


def declare(factory):
    factory.declare_module('Homogenization', HomogenizationModule)
