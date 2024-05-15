from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from src.predictions.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self, max_depth: int = None):
        super().__init__(DecisionTreeRegressor(max_depth=max_depth))