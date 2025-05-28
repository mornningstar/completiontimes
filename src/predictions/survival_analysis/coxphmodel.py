from sksurv.linear_model import CoxPHSurvivalAnalysis

from src.predictions.base_model import BaseModel


class CoxPHModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = CoxPHSurvivalAnalysis()
    
    def train(self, x_train, y_train):
        x_scaled = self.scaler.fit_transform(x_train)
        self.model.fit(x_scaled, y_train)

    def evaluate(self, x_test, y_test):
        x_scaled = self.scaler.transform(x_test)

        risk = self.model.predict(x_scaled)


