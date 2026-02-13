import enum

class ModelTrainingStatus(enum.Enum):
    improved = 1
    no_improvement = 2
    stop = 3

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, minimize: bool):
        self.patience = patience
        self.min_delta = min_delta
        self.minimize = minimize

        self.counter = 0
        self.best_score = None

    def __call__(self, current_score: float) -> ModelTrainingStatus:
        if self.best_score is None:
            self.best_score = current_score
            return ModelTrainingStatus.improved

        if self.minimize:
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return ModelTrainingStatus.improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return ModelTrainingStatus.stop
            return ModelTrainingStatus.no_improvement
