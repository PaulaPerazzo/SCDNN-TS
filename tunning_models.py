import logging

from logger.base import Logger
from tracker.wandb_tracker import WandBTracker
from utils.experiment_io import get_run_id, get_run_dir, save_run_artifacts
from utils.config_handle import load_config, flatten_dict
from utils.get_device import get_device
from utils.seed_all import seed_all

from data_loader.factory import DataLoaderFactory
from data_pre_processing.factory import DataPreProcessingFactory

from modeling.structure.factory import ModelingStructureFactory
from modeling.training.factory import ModelingTrainingFactory

from metrics.factory import MetricsFactory


def main(default_file_name, config=None, X=None, y_true=None):
    """
    Perform train phase

    args:
        config: optional if what to use specific config
        X: optional if what to use specific data
        y_true: optional if what to use specific data

    returns:
        metrics
    """

    if config is None:
        config = load_config(default_file_name=default_file_name)

    if "run_id" not in config:
        run_id = get_run_id(
            config,
            [config["modeling"]["structure"]["name"], config["data_loader"]["name"]],
        )
        config["run_id"] = run_id

    tunning_automated_id = config.get("tunning_automated_id", None)
    config["phase"] = "tunning"  # persist phase
    run_id = config["run_id"]

    if tunning_automated_id is None:
        run_dir = get_run_dir(run_id)
        save_run_artifacts(run_dir, config)

    # Setup logger
    logger = Logger(
        name="train_validation",
        log_file=(
            f"{run_dir}/{config["phase"]}_output.log"
            if tunning_automated_id is None
            else None
        ),
        level=logging.DEBUG if "debug" in config and config["debug"] else logging.INFO,
    )

    # log run id
    logger.info("Initiating training and validation...")
    logger.info(f"[ RUN ID: {run_id} ]")

    seed = 0
    seed_all(seed)
    logger.debug(f"[ Using Seed : {seed} ]")

    device = get_device()
    logger.info(f"Using device: {device}")

    # 1. Load the dataset
    if X is None or y_true is None:
        logger.debug("Loading data...")
        dataset_loader = DataLoaderFactory().get(config, logger, device)
        pre_processing = DataPreProcessingFactory().get(config, logger)
        X, y_true = pre_processing.initialize(dataset_loader)
        logger.info("Data loaded successfully.")
    else:
        logger.info("Using provided data for training and validation.")

    # 3. Initializations
    logger.debug("Initializing components...")

    # 3.1 Tracker
    logger.debug("Initializing tracker...")
    flat_config = flatten_dict(config)
    tracker = WandBTracker(config=flat_config)

    # 3.2 Model
    logger.debug("Initializing model...")
    model = ModelingStructureFactory().get(config, logger, device, tracker)
    model.compile()

    # 3.3 Trainer
    logger.debug("Initializing trainer...")
    trainer = ModelingTrainingFactory().get(config, logger, device, tracker)

    logger.debug("Initializing metrics...")
    metrics_handler = MetricsFactory().get(config, logger, device)

    # 4. Execute training
    logger.debug("Starting training...")
    fold_train_losses, fold_val_losses, metrics = trainer.train(
        model, X, y_true, metrics_handler
    )

    # 5. Get metrics
    logger.debug("Getting metrics...")
    # train_metrics = metrics_handler.get_fold_metrics(
    #     [i[0] for i in fold_train_losses], [i[1] for i in fold_train_losses]
    # )
    # val_metrics = metrics_handler.get_fold_metrics(
    #     [i[0] for i in fold_val_losses], [i[1] for i in fold_val_losses]
    # )

    logger.info("Execution completed.")

    # 7. Save run artifacts
    # if tunning_automated_id is None:
    #     logger.debug("Saving run artifacts...")
    #     save_run_artifacts(
    #         run_dir, {**config, "phase": "tunning_train"}, metrics=train_metrics
    #     )
    #     save_run_artifacts(
    #         run_dir, {**config, "phase": "tunning_val"}, metrics=val_metrics
    #     )
    #     logger.info("Run artifacts saved.")
    save_run_artifacts(run_dir, {**config, "phase": "tunning_train"}, metrics=metrics)

    logger.info("Train and validation completed.")

    return metrics


if __name__ == "__main__":
    main("resnet_ptb")
    main("tcn")
    main("tcn_multi")
    main("tcn_cross")
    main("tcn_cross_multi")
