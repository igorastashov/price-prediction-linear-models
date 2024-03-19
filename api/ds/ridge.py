import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def ridge_model():
    model = TransformedTargetRegressor(
        regressor=Pipeline([("estimator", Ridge())]), func=np.log, inverse_func=np.exp
    )
    return model
