import wandb
from tracker.base_tracker import BaseTracker


class WandBTracker(BaseTracker):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def start_run(self, model, fold_id: int = None):
        project = self.config.get("tracker_project_name")
        group = self.config.get("run_id")
        run_name = f"{group}-fold_{fold_id}"

        wandb.init(
            project=project,
            group=group,
            name=run_name,
            config=self.config,
            settings=wandb.Settings(_service_wait=300),
        )

        try:
            wandb.watch(model, log_freq=100)
        except:
            pass

    def log_metrics(self, metrics: dict, step: int = None):
        wandb.log(metrics, step=step)

    def log_artifact(self, filepath: str):
        wandb.save(filepath)

    def finish(self):
        wandb.finish()
