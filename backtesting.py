import numpy as np

def backtest(w : np.ndarray, Sigma : np.ndarray):
    """
    # Parameters
    w: N x 1 array of weights
    Sigma: N x N predicted risk matrix
    TODO

    # Returns
    TODO: realized risk to predicted risk ratio
    """

    predicted_risk = w @ Sigma @ w
    # realized_risk = TODO

    return

