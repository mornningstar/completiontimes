from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from src.predictions.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self, max_depth: int = None, grid_search: bool = False):
        super().__init__(DecisionTreeRegressor(max_depth=max_depth))
        if grid_search:
            self.param_grid = {
                'max_depth': [1, 3, 5, 10, 15, 20, 30],
                'min_samples_split': [5, 10, 20, 50],
                'min_samples_leaf': [5, 10, 20]
            }

    def grid_search(self, x_train, y_train):
        grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid=self.param_grid, cv=5,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(x_train, y_train)
        self.model = grid_search.best_estimator_

        return grid_search.best_params_
