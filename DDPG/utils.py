import numpy as np


def Orstein_Uhlenbeck(x,theta=0.15,mu=0,sigma=0.2):
    return theta*(mu-x) + sigma*np.random.randn(1)
