# -*- coding: utf-8 -*-

import numpy as np
import GPy
from aux_funcs_delayed import UtilityFunction, acq_max
import pickle
import itertools
import time

class BO_delayed(object):
    def __init__(self, f, pbounds, gp_opt_schedule, ARD=False, \
                 gp_mcmc=False, log_file=None, M_random_features=100, beta_t=None, \
                 use_init=False, save_init=False, save_init_file=None, \
                 T=50, m=10, method="our_method", batch_bo=False):
        """
        f: objective function to be maximized
        pbounds: dictionary containing the bounds for each input variable
        gp_opt_schedule: we optimize the GP hyperparameters (by maximizing the marginal likelihood) after every gp_opt_schedule iterations
        ARD: whether to use Automatic Relevance Determination, i.e., whether to learn a separate lengthscale for every input dimension
        gp_mcmc: whether to use MCMC instead of maximizing marginal likelihood to optimize the GP hyperaprameters (may not be complete)
        log_file: the file name where we save the results
        M_random_features: the number of random features we use for Thompson sampling
        beta_t: the parameter beta_t
        use_init: whether to use the initializations previously saved in a file named use_init
        save_init: (Boolean) whether to save the initializations
        save_init_file: the file name in which we save the initializations (effective only if save_init=True)
        T: total number of BO iterations
        m: the parameter m in our delayed BO algorithm
        method: the method we use, it takes values in {"our_method", "baseline_no_update", "baseline_only_update_std"}
        """
        self.batch_bo = batch_bo
        
        self.method = method
        self.m = m

        self.T = T
        
        self.use_init = use_init
        self.save_init = save_init
        self.save_init_file = save_init_file

        self.M_random_features = M_random_features
        self.ARD = ARD
        self.log_file = log_file        
        self.pbounds = pbounds
        self.incumbent = None
        self.beta_t = beta_t
        self.nu_t = []
        
        self.keys = list(pbounds.keys())
        self.dim = len(pbounds)

        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)
        
        self.f = f

        self.initialized = False

        self.init_points = []
        self.x_init = []
        self.y_init = []

        self.X = np.array([]).reshape(-1, 1)
        self.Y = np.array([])
        self.Y_censored = np.array([])
        self.X_censored = np.array([]).reshape(-1, 1)
        self.delays = []
        
        self.f_values = []

        self.iteration = 0

        self.gp_mcmc = gp_mcmc

        self.gp = None
        self.gp_opt_schedule = gp_opt_schedule

        self.gp_update_only_std = None
        
        self.util = None
        
        self.res = {}
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values':[], 'params':[], 'init_values':[], 'init_params':[], 'init':[], \
                          'f_values':[], 'init_f_values':[], 'noise_var_values':[], 'init_noise_var_values':[], \
                          'incumbent_x':[], 'delays':[], 'values_censored':[], 'f_values_censored':[]}

    def init(self, init_points, existing_init_x):
        '''
        The function for initialization (via random search)
        '''

        if existing_init_x is None:
            l = [np.random.uniform(x[0], x[1], size=init_points)
                 for x in self.bounds]
            self.init_points += list(map(list, zip(*l)))
        else:
            self.init_points = existing_init_x

        y_init = []
        delays_init = []
        for x in self.init_points:
            y, f_value, delay = self.f(x)

            y_init.append(y)
            self.delays.append(delay)
            self.f_values.append(f_value)
            
            self.res['all']['init_values'].append(y)
            self.res['all']['init_f_values'].append(f_value)
            self.res['all']['delays'].append(delay)

            self.res['all']['init_params'].append(dict(zip(self.keys, x)))

        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)

        self.incumbent = np.max(y_init)
        self.initialized = True

        init = {"X":self.X, "Y":self.Y, "f_values":self.f_values, "delays":self.delays}
        self.res['all']['init'] = init

        if self.save_init:
            pickle.dump(init, open(self.save_init_file, "wb"))


    def sample_function(self, Xs=None, Ys=None, init_points=2):
        '''
        The function used to (approximately) draw a function from a GP, for running Thompson sampling
        '''
        M_random_features = self.M_random_features

        ls_target = self.gp["rbf.lengthscale"][0]
        v_kernel = self.gp["rbf.variance"][0]
        obs_noise = self.gp["Gaussian_noise.variance"][0]

        try:
            s = np.random.multivariate_normal(np.zeros(self.dim), 1 / (ls_target**2) * np.identity(self.dim), M_random_features)
        except:
            s = np.random.rand(M_random_features, self.dim) - 0.5

        b = np.random.uniform(0, 2 * np.pi, M_random_features)

        random_features_target = {"M":M_random_features, "length_scale":ls_target, "s":s, "b":b, "obs_noise":obs_noise, "v_kernel":v_kernel}
        Phi = np.zeros((Xs.shape[0], M_random_features))
        for i, x in enumerate(Xs):
            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M_random_features) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

            features = features / np.sqrt(np.inner(features, features))
            features = np.sqrt(v_kernel) * features

            Phi[i, :] = features

        Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M_random_features)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), Ys.reshape(-1, 1))

        try:
            w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
        except:
            w_sample = np.random.rand(1, M_random_features) - 0.5

        return M_random_features, random_features_target, w_sample
    
    
    def maximize(self, n_iter=25, init_points=5, acq_type="ts"):
        '''
        The main function we use to run BO
        n_iter: the number of BO iterations
        init_points: the number of initialization points
        acq_type: the type of acquisition function, it takes values in {"ts", "ucb"}
        '''
        self.util_ts = UtilityFunction(kind=acq_type)

        if not self.initialized:
            if self.use_init != None:
                init = pickle.load(open(self.use_init, "rb"))
                self.X = init["X"]
                self.init(init_points, self.X)
            else:
                self.init(init_points, None)

        self.iteration = len(self.X) + 1

        #### Define the GPs for different methods
        if self.method == "our_method":
            # censor the output observations based on the delays
            Y_censored = []
            Y_observed = []
            f_observed = []
            for ind, y in enumerate(self.Y):
                s = ind + 1
                if self.delays[ind] <= min(self.m, self.iteration - s):
                    Y_censored.append(y)
                    Y_observed.append(y)
                    f_observed.append(self.f_values[ind])
                else:
                    Y_censored.append(0)
                    Y_observed.append(-1e6)
                    f_observed.append(-1e6)
            self.Y_censored = np.array(Y_censored)
            self.res['all']['values_censored'].append(np.max(Y_observed))
            self.res['all']['f_values_censored'].append(np.max(f_observed))

            self.gp = GPy.models.GPRegression(self.X, self.Y_censored.reshape(-1, 1), \
                    GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=0.1, variance=1.0, ARD=self.ARD))
            if init_points > 1:
                self.gp.optimize_restarts(num_restarts = 10, messages=False)
                print("---Optimized hyper: ", self.gp)

            self.nu_t = self.beta_t[self.X.shape[0] - init_points + 1]
            for j in np.arange(max(len(self.X)-self.m, 0), len(self.X)):
                _, var = self.gp.predict(self.X[j].reshape(1, -1))
                self.nu_t += np.sqrt(var[0][0])

        elif self.method == "baseline_no_update":
            X_censored = []
            Y_censored = []
            Y_observed = []
            f_observed = []
            for ind, y in enumerate(self.Y):
                s = ind + 1
                if self.delays[ind] <= self.iteration - s:
                    Y_censored.append(y)
                    X_censored.append(self.X[ind, :])
                    Y_observed.append(y)
                    f_observed.append(self.f_values[ind])
                else:
                    Y_observed.append(-1e6)
                    f_observed.append(-1e6)
            self.Y_censored = np.array(Y_censored)
            self.X_censored = np.array(X_censored)
            self.res['all']['values_censored'].append(np.max(Y_observed))
            self.res['all']['f_values_censored'].append(np.max(f_observed))

            if self.Y_censored != []:
                self.gp = GPy.models.GPRegression(self.X_censored, self.Y_censored.reshape(-1, 1), \
                        GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=0.1, variance=1.0, ARD=self.ARD))
                if init_points > 1:
                    self.gp.optimize_restarts(num_restarts = 10, messages=False)
                    print("---Optimized hyper: ", self.gp)

        elif self.method == "baseline_only_update_std":
            Y_censored = []
            Y_hallucinate = []
            Y_observed = []
            f_observed = []
            for ind, y in enumerate(self.Y):
                s = ind + 1
                if self.delays[ind] <= self.iteration - s:
                    Y_censored.append(y)
                    Y_hallucinate.append(y)
                    Y_observed.append(y)
                    f_observed.append(self.f_values[ind])
                else:
                    if self.gp is not None:
                        mean, _ = self.gp.predict(self.X[ind].reshape(1, -1))
                        Y_hallucinate.append(mean[0][0])
                    else:
                        Y_hallucinate.append(0)
                    Y_observed.append(-1e6)
                    f_observed.append(-1e6)
            self.Y_hallucinate = np.array(Y_hallucinate)
            self.Y_censored = np.array(Y_censored)
            self.res['all']['values_censored'].append(np.max(Y_observed))
            self.res['all']['f_values_censored'].append(np.max(f_observed))

            if self.Y_censored != []:
                self.gp = GPy.models.GPRegression(self.X, self.Y_hallucinate.reshape(-1, 1), \
                        GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=0.1, variance=1.0, ARD=self.ARD))                
                
                # optimize GP hypers
                if init_points > 1:
                    self.gp.optimize_restarts(num_restarts = 10, messages=False)
                    print("---Optimized hyper: ", self.gp)
            
            
        ### optimize the acquisition function to get the next input query "x_max"
        if acq_type == "ts":
            if self.method == "our_method":
                M_target, random_features_target, w_sample = self.sample_function(Xs=self.X, Ys=self.Y_censored, init_points=init_points)
            elif self.method == "baseline_no_update":
                if self.gp is not None:
                    M_target, random_features_target, w_sample = self.sample_function(Xs=self.X_censored, Ys=self.Y_censored, init_points=init_points)
            elif self.method == "baseline_only_update_std":
                if self.gp is not None:
                    M_target, random_features_target, w_sample = self.sample_function(Xs=self.X, Ys=self.Y_hallucinate, init_points=init_points)
        else:
            M_target, random_features_target, w_sample = None, None, None

        if self.gp is not None:
            if self.method == "our_method":
                x_max = acq_max(ac=self.util_ts.utility, M=M_target, random_features=random_features_target, w_sample=w_sample, \
                                bounds=self.bounds, gp=self.gp, nu_t=self.nu_t)
            elif self.method == "baseline_no_update" or self.method == "baseline_only_update_std":
                x_max = acq_max(ac=self.util_ts.utility, M=M_target, random_features=random_features_target, w_sample=w_sample, \
                                bounds=self.bounds, gp=self.gp, nu_t=self.beta_t[self.X.shape[0] - init_points + 1])
        else:
            x_max = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.bounds.shape[0])

        for i in range(n_iter):
            y, f_value, delay = self.f(x_max)

            self.Y = np.append(self.Y, y)
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.f_values.append(f_value)
            self.delays.append(delay)
            
            self.res['all']['f_values'].append(f_value)
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])

            incumbent_x = self.X[np.argmax(self.Y)]
            self.res['all']['incumbent_x'].append(incumbent_x)
            
            self.iteration += 1
    
            print("iter {0} ------ x_t: {1}, y_t: {2}, delay: {3}".format(self.iteration, x_max, y, delay))
            
            #### Update the GPs for different methods
            if self.method == "our_method":
                # censor the output observations based on the delays
                Y_censored = []
                Y_observed = []
                f_observed = []
                for ind, y in enumerate(self.Y):
                    s = ind + 1
                    if self.delays[ind] <= min(self.m, self.iteration - s):
                        Y_censored.append(y)
                        Y_observed.append(y)
                        f_observed.append(self.f_values[ind])
                    else:
                        Y_censored.append(0)
                        Y_observed.append(-1e6)
                        f_observed.append(-1e6)
                self.Y_censored = np.array(Y_censored)
                self.res['all']['values_censored'].append(np.max(Y_observed))
                self.res['all']['f_values_censored'].append(np.max(f_observed))

                # update the GP posterior
                self.gp.set_XY(X=self.X, Y=self.Y_censored.reshape(-1, 1))

                # update the GP hyperparameters after every "gp_opt_schedule" iteration
                if len(self.X) >= self.gp_opt_schedule and len(self.X) % self.gp_opt_schedule == 0:
                    self.gp.optimize_restarts(num_restarts = 10, messages=False)
                    print("---Optimized hyper: ", self.gp)

                self.nu_t = self.beta_t[self.X.shape[0] - init_points + 1]
                for j in np.arange(max(len(self.X)-self.m, 0), len(self.X)):
                    _, var = self.gp.predict(self.X[j].reshape(1, -1))
                    self.nu_t += np.sqrt(var[0][0])
                
            elif self.method == "baseline_no_update":
                X_censored = []
                Y_censored = []
                Y_observed = []
                f_observed = []
                for ind, y in enumerate(self.Y):
                    s = ind + 1
                    if self.delays[ind] <= self.iteration - s:
                        Y_censored.append(y)
                        X_censored.append(self.X[ind, :])
                        Y_observed.append(y)
                        f_observed.append(self.f_values[ind])
                    else:
                        Y_observed.append(-1e6)
                        f_observed.append(-1e6)
                Y_censored = np.array(Y_censored)
                X_censored = np.array(X_censored)

                self.Y_censored = Y_censored
                self.X_censored = X_censored

                self.res['all']['values_censored'].append(np.max(Y_observed))
                self.res['all']['f_values_censored'].append(np.max(f_observed))

                if self.Y_censored != []:
                    if self.gp is not None:
                        self.gp.set_XY(X=self.X_censored, Y=self.Y_censored.reshape(-1, 1))
                    else:
                        self.gp = GPy.models.GPRegression(self.X_censored, self.Y_censored.reshape(-1, 1), \
                                GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=0.1, variance=1.0, ARD=self.ARD))                
                    # update the GP hyperparameters after every "gp_opt_schedule" iteration
                    if len(self.X) >= self.gp_opt_schedule and len(self.X) % self.gp_opt_schedule == 0:
                        self.gp.optimize_restarts(num_restarts = 10, messages=False)
                        print("---Optimized hyper: ", self.gp)

            elif self.method == "baseline_only_update_std":
                Y_censored = []
                Y_hallucinate = []
                Y_observed = []
                f_observed = []
                for ind, y in enumerate(self.Y):
                    s = ind + 1
                    if self.delays[ind] <= self.iteration - s:
                        Y_censored.append(y)
                        Y_hallucinate.append(y)
                        Y_observed.append(y)
                        f_observed.append(self.f_values[ind])
                    else:
                        if self.gp is not None:
                            mean, _ = self.gp.predict(self.X[ind].reshape(1, -1))
                            Y_hallucinate.append(mean[0][0])
                        else:
                            Y_hallucinate.append(0)
                        Y_observed.append(-1e6)
                        f_observed.append(-1e6)
                self.Y_hallucinate = np.array(Y_hallucinate)
                self.Y_censored = np.array(Y_censored)
                self.res['all']['values_censored'].append(np.max(Y_observed))
                self.res['all']['f_values_censored'].append(np.max(f_observed))

                if self.Y_censored != []:
                    if self.gp is not None:
                        self.gp.set_XY(X=self.X, Y=self.Y_hallucinate.reshape(-1, 1))
                    else:
                        self.gp = GPy.models.GPRegression(self.X, self.Y_hallucinate.reshape(-1, 1), \
                                GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=0.1, variance=1.0, ARD=self.ARD))
                    # update the GP hyperparameters after every "gp_opt_schedule" iteration
                    if len(self.X) >= self.gp_opt_schedule and len(self.X) % self.gp_opt_schedule == 0:
                        self.gp.optimize_restarts(num_restarts = 10, messages=False)
                        print("---Optimized hyper: ", self.gp)
                
                
            ### optimize the acquisition function to get the next input query "x_max"
            if acq_type == "ts":
                if self.method == "our_method":
                    M_target, random_features_target, w_sample = self.sample_function(Xs=self.X, Ys=self.Y_censored, init_points=init_points)
                elif self.method == "baseline_no_update":
                    if self.gp is not None:
                        M_target, random_features_target, w_sample = self.sample_function(Xs=self.X_censored, Ys=self.Y_censored, init_points=init_points)
                elif self.method == "baseline_only_update_std":
                    if self.gp is not None:
                        M_target, random_features_target, w_sample = self.sample_function(Xs=self.X, Ys=self.Y_hallucinate, init_points=init_points)
            else:
                M_target, random_features_target, w_sample = None, None, None

            if self.gp is not None:
                if self.method == "our_method":
                    x_max = acq_max(ac=self.util_ts.utility, M=M_target, random_features=random_features_target, w_sample=w_sample, \
                                    bounds=self.bounds, gp=self.gp, nu_t=self.nu_t)
                elif self.method == "baseline_no_update" or self.method == "baseline_only_update_std":
                    x_max = acq_max(ac=self.util_ts.utility, M=M_target, random_features=random_features_target, w_sample=w_sample, \
                                    bounds=self.bounds, gp=self.gp, nu_t=self.beta_t[self.X.shape[0] - init_points + 1])
            else:
                x_max = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.bounds.shape[0])

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))
