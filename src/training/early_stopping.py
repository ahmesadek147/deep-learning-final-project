
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="min"):
        if mode not in {"min", "max"}:
            raise ValueError("early-stopping mode must be 'min' or 'max'.")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, value):
        improved = (value < self.best_score - self.min_delta if self.mode == "min"
                    else value > self.best_score + self.min_delta)
        if improved:
            self.best_score = value
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
