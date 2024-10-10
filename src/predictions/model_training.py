from src.data_handling.file_data_handling import FileDataHandler


class ModelTrainer:
    def __init__(self, models, modeling_tasks=None):
        self.models = models
        self.modeling_tasks = modeling_tasks

    def train_and_evaluate_model(self, x_train, y_train, x_test, y_test):
        model_info = {}

        for model in self.models:
            # Only call auto_tune() when the model class has this method
            if hasattr(model, 'auto_tune'):
                model.auto_tune(y_train)
            if hasattr(model, 'param_grid'):
                params = model.grid_search(x_train, y_train)
                print(f"Best params for {model.__class__.__name__}: {params}")

            model.train(x_train, y_train)
            predictions, mse, mae, rmse = model.evaluate(y_test, x_test)

            model_info[model.__class__.__name__] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'predictions': predictions,
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test
            }
            print(f"Trained {model.__class__.__name__} with MSE: {mse}, MAE: {mae}, RMSE: {rmse}")
            print(f"Predictions: {predictions}")

        return model_info
