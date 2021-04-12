import numpy as np
from lmfit import minimize
from lmfit import Parameters
import warnings
from lmfit import fit_report
import scipy


class Helix():
    """A class used to fit a helix to given data points.

    Attributes
    ----------
    handednes : int
        Returns 1 or -1 whether the helix is right- or left-handed
    fitting_method : str
        which optimization method to use when fitting the helix
    RMSD : float
        Root mean square deviation
    get_statistics : float
        Returns an array of delta z-values

    Methods
    -------
    fit_helix()
        Tries to fit a helix by minimizing the residuals with the chosen method
    """

    def __init__(
        self,
        data,
        fitting_method: str = 'ampgo',
        radius: float = 0.8,
        handedness: int = None,
        truncation=None,
    ):
        """
        Parameters
        ----------
        data : list or ndarray
            List or numpy.ndarray of coordinates to fit
        a : list
            Unit vector giving the direction of the helix
        fitting_method : str, optional
            Which method to use for the minimization. Default is 'ampgo'
        radius : float
            Radius of the helix. Default is 0.8
        truncation : list
            How to truncate the data. Useful for systems that have
            two helical parts, e.g. metallacumulenes

        Notes
        -----
        ampgo with L-BFGS-B is default as it finds the global minimum.
        It's slow though.
        """
        if type(data) is not np.ndarray:
            if type(data) is list:
                data = np.array(data)
            else:
                raise TypeError('Data has to be a list or numpy.ndarray')

        if data.ndim < 2 or data[1, :].shape != (3,):
            raise ValueError('Data has to be an array of x,y,z-coordinates')

        self.data = data
        self.truncation = truncation

        self.a = np.array([0, 0, 1])
        self.v = np.array([0, 1, 0])
        tmp_w = np.array([1, 0, 0])
        if handedness:
            self.w = handedness * tmp_w
        else:
            self.w = self.handedness * tmp_w

        self.radius = radius

        self._fitting_method = fitting_method

    @property
    def handedness(self) -> int:
        if len(self.data) < 5:
            # Probably not a helix
            # so return default handedness 1
            return 1

        else:
            first_p = [self.data[3, 0], self.data[3, 1]]
            second_p = [self.data[4, 0], self.data[4, 1]]

            return 1 if np.cross(first_p, second_p) > 0 else -1

    @property
    def fitting_method(self) -> str:
        return self._fitting_method

    def _cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        return rho, phi

    def fit_helix(self):
        """Tries to fit the helix to the given data points"""

        params = Parameters()
        params.add(
            'radius',
            value=self.radius,
            vary=False
        )
        params.add('pitch', min=0)
        params.add('phi', min=-np.pi, max=np.pi)
        params.add('gamma', min=0)

        t_range = np.max(self.data)/2
        t = np.linspace(-t_range, t_range, len(self.data))

        # truncate the points so it only fits to one helical part
        # t = t[self.truncation[0]:self.truncation[1]]
        # self.data = self.data[self.truncation[0]:self.truncation[1]]

        self.minimizer_result = minimize(
            self._residual,
            params,
            args=(t, self.data),
            method=self.fitting_method
        )

        fitted_params = self.minimizer_result.params.valuesdict()
        self.fitted_values = self._helix_func(fitted_params, t)

        return fit_report(self.minimizer_result)

    @property
    def RMSD(self):
        return np.sqrt(
            np.sum((self.data - self.fitted_values)**2)/len(self.data)
        )

    @property
    def p_value(self):
        return scipy.stats.chi2.sf(
            self.minimizer_result.chisqr, self.minimizer_result.nfree
        )

    def get_statistics(self):
        delta_z = []
        for i, coord in enumerate(self.data):
            if i > 0:
                delta_z.append(
                    np.linalg.norm(self.data[i]) - np.linalg.norm(self.data[i-1])
                )
        return delta_z

    def _perpendicular_vector(self, v):
        if v[1] == 0 and v[2] == 0:
            if v[0] == 0:
                raise ValueError('zero vector')
            else:
                return np.cross(v, [0, 1, 0])
        return np.cross(v, [1, 0, 0])

    def _func(self, a, v, w, t, params):
        """
        Parameters
        ----------
        R : int
            The radius of the helix
        g : float
            Gamma is the angular frequency. Modifies the input, t, so
            it can "reach the other side".
            Otherwise the range of t might not cover a whole unit circle
        p : float
            Pitch. How far to go up the z-axis to go around the helix once.
        phi : float
            Phase shift. Equivalent to rotation about the z-axis
        """
        R = params['radius']
        P = params['pitch']
        phi = params['phi']
        g = params['gamma']

        return a*t*P*g/(2*np.pi) + R*(w*np.cos(t*g) + v*np.sin(t*g))

    def _helix_func(self, params, t):
        coords = np.column_stack(
            [
                self._func(self.a[0], self.v[0], self.w[0], t, params),
                self._func(self.a[1], self.v[1], self.w[1], t, params),
                self._func(self.a[2], self.v[2], self.w[2], t, params),
            ]
        )

        return coords

    def _residual(self, params, t, data):
        residual = np.array(
            [
                (self._func(self.a[0], self.v[0], self.w[0], t, params) - self.data[:, 0])**2,
                (self._func(self.a[1], self.v[1], self.w[1], t, params) - self.data[:, 1])**2,
                (self._func(self.a[2], self.v[2], self.w[2], t, params) - self.data[:, 2])**2,
            ]
        ).ravel()

        return residual
