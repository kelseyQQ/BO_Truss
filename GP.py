import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from cycler import cycler
import seaborn as sns

# Set the color scheme
sns.set_theme()
colors = [
    "#0076C2",
    "#EC6842",
    "#A50034",
    "#009B77",
    "#FFB81C",
    "#E03C31",
    "#6CC24A",
    "#EF60A3",
    "#0C2340",
    "#00B8C8",
    "#6F1D77",
]
plt.rcParams["axes.prop_cycle"] = cycler(color=colors)

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

# By default torch uses float32, this can lead to numerical problems for the (semi-)positive definiteness of tensors.
torch.set_default_dtype(torch.float64)

# Implement the covariance function

def cov_gaussian(X1, X2, **hyperparams):
    sig_f = hyperparams["sig_f"]
    length = hyperparams["length"]

    out = sig_f**2 * torch.exp(-torch.cdist(X1, X2) ** 2 / (2 * length**2))
    
    return out

def cov_quad_exp_add(X1, X2, **hyperparams):
    theta0 = hyperparams["theta0"]
    theta1 = hyperparams["theta1"]
    theta2 = hyperparams["theta2"]
    theta3 = hyperparams["theta3"]

    out = theta0 * torch.exp(-theta1/2 * torch.cdist(X1, X2)**2) + theta2 + theta3 * torch.mm(X1.T, X2)

    return out

# Gaussian process posterior
def GP(X, t, X_hat, kernel, hyperparams):
    """
    :param X: Observation locations [N]
    :param t: Observation values [N]
    :param X_hat: Prediction locations [Np]
    :param kernel: covariance function
    :param hyperparams: The hyperparameters
    :return: posterior mean [Np] and covariance matrix [Np,Np]
    """
    with torch.no_grad():
        noise = hyperparams["noise"] 

        k11 = kernel(X, X, **hyperparams)

        C = k11 + hyperparams["noise"] ** 2 * torch.eye(k11.shape[0])
        
        mu = kernel(X_hat, X, **hyperparams) @ torch.inverse(C) @ t

        cov = kernel(X_hat, X_hat, **hyperparams) - kernel(X_hat, X, **hyperparams) @ torch.inverse(C) @ kernel(X, X_hat, **hyperparams)

    return mu, cov

# Gaussian process log marginal likelihood
def GP_logmarglike(X, t, kernel, hyperparams):
    """
    Calculate the log marginal likelihood based on the observations (X, t) for a given kernel
    """
    # Kernel of the observations
    k11 = kernel(X, X, **hyperparams)

    C = k11 + hyperparams["noise"] ** 2 * torch.eye(k11.shape[0])
    # ---------------------- student exercise --------------------------------- #
    logmarglike = -torch.logdet(C)/2 - t.T/2 @ torch.inverse(C) @ t - t.shape[0] * torch.log(torch.tensor(2 * np.pi))/2
    # ---------------------- student exercise --------------------------------- #

    return logmarglike