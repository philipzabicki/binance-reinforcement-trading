from pymoo.core.callback import Callback


class EarlyStoppingCallback(Callback):
    def __init__(self, patience=7, tol=1e-9):
        super().__init__()
        self.patience = patience
        self.tol = tol
        self.best = None
        self.wait = 0

    def notify(self, algorithm):
        curr_best = np.min(algorithm.pop.get("F"))
        if self.best is None or curr_best < self.best - self.tol:
            self.best = curr_best
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            algorithm.termination.force_termination = True
