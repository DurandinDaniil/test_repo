
import numpy as np

from scipy.optimize import curve_fit
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class KalmanTracker(object):
    """ This class represents the internal state of individual tracked objects observed as point.
    """

    count = 0

    def __init__(self, point):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # initial value for the state (position and velocity)
        self.kf.x[:2] = point
        # define the state transition matrix
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # define the measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        # define the covariance matrix
        self.kf.P *= 1000.
        # assign the measurement noise
        self.kf.R *= 5.
        # assign the process noise
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.13)

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.history = []

    def update(self, point):
        """ Updates the state vector with observed point.
        """
        self.history = []
        self.kf.update(point)

    def predict(self):
        """ Advances the state vector and returns the predicted point (+velocity) estimate.
        """
        self.kf.predict()
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """ Returns the current point estimate.
        """
        return self.kf.x


def quadratic(x, a, b, c):
    """Objective function (putative function)
    :param x:
    :param a:
    :param b:
    :param c:
    :return: value at point `x`, `y(x)`
    """

    return a * x ** 2 + b * x + c


def linear(x, a, b):
    """Objective function (putative function)
    :param x: np.array
    :param a:
    :param b:
    :return: value at point `x`, `y(x)`
    """

    return a * x + b


def fit(points: list, yx=False, func=quadratic):
    """Fit points to func.

    :param points:
    :param yx: bool, optional
            don't forget about `implicit function theorem` when use this
            methods to fit the curve data
    :param func: callable
            The model function, f(x, ...). It must take the independent
            variable as the first argument and the parameters to fit as
            separate remaining arguments.
    :return:
    """
    if len(points) > 2:  # polynomial order is equal to three
        if yx:
            y, x = zip(*points)
        else:
            x, y = zip(*points)
        p_opt, p_cov = curve_fit(func, x, y)

        # summarize the parameter values
        # a, b, c = p_opt
        e = np.sqrt(np.diag(p_cov))
        # print('y = %.5f * x ** 2 + %.5f * x + %.5f' % (a, b, c))
        # print('%.5f, %.5f, %.5f' % tuple(e))

        # return True, (a, b, c), e
        return True, tuple(p_opt), e
    else:
        return False, (), ()
