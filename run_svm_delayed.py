import GPy
from bayesian_optimization_delayed import BO_delayed
import pickle
import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split

from scipy.stats import poisson

max_iter = 100

#### Whether to run methods with stochastic delays (batch_bo = False) or deterministic delays (batch_bo = True)
# batch_bo = True
batch_bo = False

#### The batch size for the setting with deterministic delays, only in effect if batch_bo = True
batch_size = 10

### the mean parameter for the Poisson distribution of delays
mu = 10

diabetes_data = pd.read_csv("clinical_data/diabetes.csv")
label = np.array(diabetes_data["Outcome"])
features = np.array(diabetes_data.iloc[:, :-1])
X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size=0.3, stratify=label, random_state=0)
n_ft = X_train.shape[1]
n_classes = 2

def svm_val_acc_func(param):
    parameter_range = [[1e-4, 100.0], [1e-4, 10.0]]
    C_ = param[0]
    C = C_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0]
    gam_ = param[1]
    gam = gam_ * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]

    clf = svm.SVC(kernel="rbf", C=C, gamma=gam, probability=True)
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    acc = np.count_nonzero(pred == Y_test) / len(Y_test)

    if batch_bo:
        sampled_delay = batch_size
    else:
        sampled_delay = poisson.rvs(mu=mu)

    return acc, acc, sampled_delay

dim = 2
pbounds = {}
for i in range(dim):
    pbounds["x" + str(i+1)] = (0, 1)

init_size = 5
beta_t = np.ones(500)

m = 20

if batch_bo:
    m = batch_size

#### The number of random features for TS
M_random_features = 200

#### which method to run:
method, policy = "our_method", "ucb" # to run GP-UCB-SDF
# method, policy = "our_method", "ts" # to run GP-TS-SDF
# method, policy = "baseline_no_update", "ucb" # to run GP-UCB
# method, policy = "baseline_no_update", "ts" # to run asy-TS
# method, policy = "baseline_only_update_std", "ucb" # to run GP-BUCB
# method, policy = "baseline_only_update_std", "ts" # to run GP-BTS


#### whether to save the current iteration
save_initialization = False

run_list = np.arange(10)
for itr in run_list:
    if not batch_bo:
        log_file_name = "results_" + policy + "_delayed/res_iter_" + str(itr) + "_init_" + str(init_size) + \
                "_m_" + str(m) + "_mu_" + str(mu) + ".pkl"
    else:
        log_file_name = "results_" + policy + "_delayed/res_iter_" + str(itr) + "_init_" + str(init_size) + \
                "_batch_size_" + str(batch_size) + ".pkl"

    init_file_name = "inits_delayed/init_itr_" + str(itr) + "_init_" + str(init_size) + ".p"

    if method == "baseline_no_update":
        log_file_name = log_file_name[:-4] + "_baseline_no_update.pkl"
    elif method == "baseline_only_update_std":
        log_file_name = log_file_name[:-4] + "_baseline_only_update_std.pkl"

    if save_initialization:
        use_init, save_init = None, True
        save_init_file=init_file_name
    else:
        use_init, save_init = init_file_name, False
        save_init_file = None

    bo_ts = BO_delayed(f=svm_val_acc_func, pbounds=pbounds, \
               gp_opt_schedule=10, log_file=log_file_name, M_random_features=M_random_features, beta_t=beta_t, \
               use_init=use_init, save_init=save_init, save_init_file=save_init_file, \
               T=max_iter, m=m, method=method)
    bo_ts.maximize(n_iter=max_iter, init_points=init_size, acq_type=policy)
