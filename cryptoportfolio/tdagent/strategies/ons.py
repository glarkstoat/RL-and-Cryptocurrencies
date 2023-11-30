from cryptoportfolio.tdagent.tdagent import TDAgent
import numpy as np
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False


class ONS(TDAgent):
    """
    Online newton step algorithm.

    Reference:
        A.Agarwal, E.Hazan, S.Kale, R.E.Schapire.
        Algorithms for Portfolio Management based on the Newton Method, 2006.
        http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_AgarwalHKS06.pdf
    """
    def __init__(self, delta=0.125, beta=1., eta=0., A = None):
        """
        :param delta, beta, eta: Model parameters. See paper.
        """
        super(ONS, self).__init__()
        self.delta = delta
        self.beta = beta
        self.eta = eta
        self.A = A
        self._name = "ONS"

    def init_portfolio(self, X):
        m = X.size
        self.A = np.mat(np.eye(m))
        self.b = np.mat(np.zeros(m)).T

    def decide_weights(self, x, prev_weights):
        '''
        :param x: input matrix
        :param prev_weights: last portfolio
        '''
        x = self.DataManager.relative_price_vector(self._timestep)
        if self.A is None:
            self.init_portfolio(x)

        # calculate gradient
        grad = np.mat(x / np.dot(prev_weights, x)).T
        # update A
        self.A += grad * grad.T
        # update b
        self.b += (1 + 1./self.beta) * grad

        # projection of p induced by norm A
        pp = self.projection_in_norm(self.delta * self.A.I * self.b, self.A)
        return pp * (1 - self.eta) + np.ones(len(x)) / float(len(x)) * self.eta

    def projection_in_norm(self, x, M):
        """ Projection of x to simplex indiced by matrix M. Uses quadratic programming.
        """
        m = M.shape[0]

        P = matrix(2*M)
        q = matrix(-2 * M * x)
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m,1)))
        A = matrix(np.ones((1,m)))
        b = matrix(1.)

        sol = solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol['x'])