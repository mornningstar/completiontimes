from src.data_handling.file_data_handling import FileDataHandler


class ModelTrainer:
    def __init__(self, models):
        self.models = models

    def train_and_evaluate_model(self, x_train, y_train, x_test, y_test):
        model_info = {}

        for model in self.models:
            # Only call auto_tune() when the model class has this method
            if hasattr(model, 'auto_tune'):
                model.auto_tune(y_train)

            model.train(x_train, y_train)
            predictions, mse = model.evaluate(y_test, x_test)

            model_info[model.__class__.__name__] = {
                'mse': mse,
                'predictions': predictions,
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test
            }
            print(f"Trained {model.__class__.__name__} with MSE: {mse}")
            print(f"Predictions: {predictions}")

        return model_info
