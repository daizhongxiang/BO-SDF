import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import pickle

def acq_max(ac, M, random_features, w_sample, bounds, gp, nu_t):
    '''
    The function to optimize the acquisition function. In particular, we firstly draw some random inputs to evaluate their acquisition values, after that, we use L-BFGS-B with random re-starts to refine the optimization
    '''
    para_dict={"M":M, "random_features":random_features, "w_sample":w_sample, "gp":gp, "nu_t":nu_t}
    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                 size=(10000, bounds.shape[0]))

    ys = []
    for x in x_tries:
        ys.append(ac(x.reshape(1, -1), para_dict))
    ys = np.array(ys)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(100, bounds.shape[0]))
    for x_try in x_seeds:
        res = minimize(lambda x: -ac(x.reshape(1, -1), para_dict), x_try.reshape(1, -1), bounds=bounds, method="L-BFGS-B")
        if max_acq is None or -res.fun >= max_acq:
            x_max = res.x
            max_acq = -res.fun
    
    return x_max


class UtilityFunction(object):
    def __init__(self, kind):
        if kind not in ['ucb', 'ts']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose ucb or ts.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, para_dict):
        M, random_features, w_sample, gp, nu_t = para_dict["M"], para_dict["random_features"], \
                para_dict["w_sample"], para_dict["gp"], para_dict["nu_t"]
        if self.kind == 'ucb':
            return self._ucb(x, gp, nu_t)
        elif self.kind == 'ts':
            return self._ts(x, M, random_features, w_sample)

    @staticmethod
    def _ts(x, M, random_features, w_sample):
        d = x.shape[1]
        
        s = random_features["s"]
        b = random_features["b"]
        obs_noise = random_features["obs_noise"]
        v_kernel = random_features["v_kernel"]

        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
        features = features.reshape(-1, 1)

        features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
        features = np.sqrt(v_kernel) * features

        f_value = np.squeeze(np.dot(w_sample, features))

        return f_value

    @staticmethod
    def _ucb(x, gp, nu_t):
        mean, var = gp.predict(x)
        std = np.sqrt(var)

        return np.squeeze(mean + nu_t * std)
