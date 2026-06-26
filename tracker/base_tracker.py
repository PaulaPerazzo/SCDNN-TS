class BaseTracker:
    def start_run(self, model, fold_id: int = None):
        pass

    def log_metrics(self, metrics: dict, step: int):
        pass

    def log_artifact(self, filepath: str):
        pass  # Optional

    def finish(self):
        pass
