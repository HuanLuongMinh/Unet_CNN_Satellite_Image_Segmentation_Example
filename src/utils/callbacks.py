class EarlyStopping:
    """Stops training when mIoU decreases for `patience` consecutive evaluations."""

    def __init__(self, patience: int = 4, delta: float = 1e-4):
        self.patience  = patience
        self.delta     = delta
        self.prev      = None
        self.counter   = 0
        self.triggered = False

    def step(self, miou: float) -> bool:
        """Call after each validation. Returns True when training should stop."""
        if self.prev is not None and miou < self.prev - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        else:
            self.counter = 0
        self.prev = miou
        return self.triggered
