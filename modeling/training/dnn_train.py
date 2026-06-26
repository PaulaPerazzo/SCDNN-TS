import json
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
import wandb
from torch.nn.utils import clip_grad_norm_

from modeling.structure.factory import ModelingStructureFactory
from modeling.training.early_stopping import EarlyStopping
from modeling.training.pytorch_base import PytorchTrainingAlgorithm
from modeling.structure.pytorch_base import PytorchModelStructure
from utils.create_loader import create_loader
from utils.criterion import get_criterion
from utils.seed_all import DEFAULT_SEED
from torch.utils.data import DataLoader, Dataset


class DNNTrain(PytorchTrainingAlgorithm):
    def fit(
        self, model: PytorchModelStructure, train_loader: DataLoader, epoch: int
    ) -> float:
        """ "
        Update model weights

        args:
            model: pytorch model
            train_loader: data
            epoch: current epoch

        returns:
            train loss
        """

        model.train()
        train_loss = 0

        num_logs = 10
        log_every = max(
            1,
            (self.config.get("modeling", {}).get("training", {}).get("num_epochs"))
            // num_logs,
        )

        for i, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device).reshape(-1)

            self.optimizer.zero_grad()
            out = model.forward(data)
            if self.output_size == 1:
                out = out.reshape(-1)
            loss = self.criterion(out, target)
            loss.backward()

            train_loss += loss.item()

            clip = self.config.get("modeling", {}).get("training", {}).get("clip", -1)
            if clip is not None and float(clip) > 0:
                clip_grad_norm_(model.parameters(), float(clip))

            self.optimizer.step()

            # metrics logs
            should_log = (
                epoch == 0
                or epoch
                == (
                    self.config.get("modeling", {})
                    .get("training", {})
                    .get("num_epochs")
                )
                - 1
                or epoch % log_every == 0
            )
            if should_log:
                self.logger.info(
                    "Epoch: {} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        i * len(data),
                        len(train_loader.dataset),
                        100.0 * i / len(train_loader),
                        loss.item(),
                    )
                )

        train_loss = train_loss / len(train_loader)

        return train_loss

    def validate(
        self, model: PytorchModelStructure, val_loader: DataLoader, epoch: int
    ) -> float:
        """ "
        Inference model to get validation loss

        args:
            model: pytorch model
            val_loader: data
            epoch: current epoch

        returns:
            validation loss
        """

        model.eval()
        val_loss = 0
        y_pred = []
        y_true = []

        num_logs = 10
        log_every = max(
            1,
            (self.config.get("modeling", {}).get("training", {}).get("num_epochs"))
            // num_logs,
        )
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data = data.to(self.device)
                target = target.to(self.device).reshape(-1)

                out = model.forward(data)
                if self.output_size == 1:
                    out = out.reshape(-1)
                loss = self.criterion(out, target)
                val_loss += loss.item()

                y_pred.append(out)
                y_true.append(target)

                # metrics logs
                should_log = (
                    epoch == 0
                    or epoch
                    == (
                        self.config.get("modeling", {})
                        .get("training", {})
                        .get("num_epochs")
                    )
                    - 1
                    or epoch % log_every == 0
                )
                if should_log:
                    self.logger.info(
                        "Epoch: {} \t[{}/{} ({:.0f}%)]\tValidation loss: {:.6f}".format(
                            epoch,
                            i * len(data),
                            len(val_loader.dataset),
                            100.0 * i / len(val_loader),
                            loss.item(),
                        )
                    )

        val_loss = val_loss / len(val_loader)

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        return val_loss, (y_true, y_pred)

    def __create_loaders(
        self, X: np.ndarray, y: pd.DataFrame
    ) -> tuple[DataLoader, DataLoader]:
        batch_size = (
            self.config.get("modeling", {}).get("training", {}).get("batch_size")
        )

        X_train, X_val, X_test = X
        y_train, y_val, y_test = y

        normalize = bool(
            self.config.get("modeling", {}).get("training", {}).get("normalize", False)
        )
        if normalize:
            X_train, X_val, X_test = self.__normalize_signals(X_train, X_val, X_test)

        def convert(label):
            converted_label = label.astype(np.float32).reshape(-1, 1)
            if self.output_size == 1:
                converted_label = (
                    np.where(label == 0, 0, 1).astype(np.float32).reshape(-1, 1)
                )
            return converted_label

        class dataset(Dataset):

            def __init__(self, signal, label):

                self.data = signal
                self.label = label

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()
                ecg = torch.from_numpy(self.data[idx].astype(np.float32))
                target = torch.from_numpy(self.label[idx])  # .type(torch.long)

                sample = (ecg, target)
                return sample

        ### --------------------------------
        # convert target to tensor

        converted_y_train = convert(y_train)
        converted_y_test = convert(y_test)
        converted_y_val = convert(y_val)

        ### -------------------------------
        # create dataset

        trainset = dataset(signal=X_train, label=converted_y_train)
        testset = dataset(signal=X_test, label=converted_y_test)
        valset = dataset(signal=X_val, label=converted_y_val)

        ### --------------------------------
        # create dataloader

        train_loader = DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            dataset=testset,
            batch_size=batch_size,
            shuffle=False,
        )
        val_loader = DataLoader(
            dataset=valset,
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader, test_loader

    def __normalize_signals(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)

        self.logger.info("Applying train-set per-lead z-score normalization")

        return (
            ((X_train - mean) / std).astype(np.float32, copy=False),
            ((X_val - mean) / std).astype(np.float32, copy=False),
            ((X_test - mean) / std).astype(np.float32, copy=False),
        )

    def __get_class_weights(self, y_train: np.ndarray) -> torch.Tensor | None:
        use_class_weights = bool(
            self.config.get("modeling", {})
            .get("training", {})
            .get("class_weights", False)
        )
        criterion_name = (
            self.config.get("modeling", {}).get("training", {}).get("criterion")
        )
        if not use_class_weights or criterion_name not in {
            "cross_entropy_loss",
            "categorical_cross_entropy",
        }:
            return None

        labels = np.asarray(y_train).reshape(-1).astype(np.int64)
        num_classes = (
            int(self.output_size) if self.output_size else int(labels.max() + 1)
        )
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        safe_counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (num_classes * safe_counts)
        weights = weights / weights.mean()

        self.logger.info(
            f"Using class-weighted CrossEntropyLoss. Class counts: {counts.astype(int).tolist()}"
        )

        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def __create_scheduler(self):
        scheduler_config = (
            self.config.get("modeling", {}).get("training", {}).get("scheduler", {})
        )
        if not scheduler_config or scheduler_config.get("name") != "reduce_on_plateau":
            return None

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 2)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )

    def create_fold_split(self, X, Y, test_fold):
        """
        Parameters
        ----------
        X : np.ndarray
            ECG signals.
        Y : pd.DataFrame
            Must contain 'strat_fold' and the target column.
        test_fold : int
            Fold to use as the test set (1-10).

        Returns
        -------
        X_split : tuple
        y_split : tuple
        """

        # Use the next fold for validation
        val_fold = test_fold - 1 if test_fold > 1 else 10

        train_mask = (Y.strat_fold != test_fold) & (Y.strat_fold != val_fold)

        val_mask = Y.strat_fold == val_fold
        test_mask = Y.strat_fold == test_fold

        classes = [
            "NORM",
            "MI",
            "STTC",
            "CD",
            "HYP",
        ]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        num_classes = 5
        labels = Y["label"].map(class_to_idx).values
        y_onehot = np.eye(num_classes)[labels]

        X_train = X[train_mask.values]
        X_val = X[val_mask.values]
        X_test = X[test_mask.values]

        # Replace "target" with your label column
        y_train = np.argmax(y_onehot[train_mask], axis=1)
        y_val = np.argmax(y_onehot[val_mask], axis=1)
        y_test = np.argmax(y_onehot[test_mask], axis=1)

        return (
            (X_train, X_val, X_test),
            (y_train, y_val, y_test),
        )

    def train(
        self,
        model: PytorchModelStructure,
        X: np.ndarray,
        y: pd.DataFrame,
        metrics_handler,
    ) -> tuple[float, float]:
        """ "
        Execute training. Early stopping is options based on configs

        args:
            model: pytorch model
            train_loader: train data
            val_loader: validation data

        returns:
            train loss, validation loss
        """

        self.output_size = (
            self.config.get("modeling", {}).get("structure", {}).get("output_size", 5)
        )
        criterion_name = (
            self.config.get("modeling", {}).get("training", {}).get("criterion")
        )
        reduction = (
            self.config.get("modeling", {}).get("training", {}).get("reduction", "mean")
        )
        learning_rate = (
            self.config.get("modeling", {}).get("training", {}).get("learning_rate")
        )
        num_epochs = (
            self.config.get("modeling", {}).get("training", {}).get("num_epochs")
        )
        es_patience = (
            self.config.get("modeling", {}).get("training", {}).get("es_patience")
        )

        completed_folds_file = self.run_dir / "completed_folds.json"
        if completed_folds_file.exists():
            with open(completed_folds_file, "r") as f:
                completed_folds = set(json.load(f))
        else:
            completed_folds = set()

        for test_fold in range(1, 11):
            model.reset_weights()

            if test_fold in completed_folds:
                self.logger.info(f"Skipping completed fold {test_fold}/{10}")
                continue

            X_split, y_split = self.create_fold_split(
                X,
                y,
                test_fold,
            )

            train_loader, val_loader, test_loader = self.__create_loaders(
                X_split, y_split
            )
            y_train = y_split[0]

            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=float(learning_rate)
            )
            class_weights = self.__get_class_weights(y_train)
            if class_weights is None:
                self.criterion = get_criterion(criterion_name, reduction=reduction)
            else:
                self.criterion = torch.nn.CrossEntropyLoss(
                    weight=class_weights, reduction=reduction
                )
            scheduler = self.__create_scheduler()

            early_stopping = None
            if es_patience:
                early_stopping = EarlyStopping(self.config, self.logger, model)

            self.logger.info(f"Running for {num_epochs} epochs")
            self.logger.info(
                f"-------------------- Training started -------------------"
            )
            self.tracker.start_run(model, fold_id=test_fold)
            for epoch in range(num_epochs):
                train_loss = self.fit(model, train_loader, epoch)
                val_loss, (y_true, y_pred) = self.validate(model, val_loader, epoch)

                metrics = metrics_handler.get_run_metrics(y_true, y_pred, verbose=True)

                self.tracker.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss, **metrics},
                    step=epoch,
                )

                if scheduler:
                    previous_lr = self.optimizer.param_groups[0]["lr"]
                    scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if current_lr != previous_lr:
                        self.logger.info(
                            f"Learning rate reduced from {previous_lr:.6g} to {current_lr:.6g}"
                        )

                if early_stopping:
                    condition_match = early_stopping(val_loss, epoch)
                    if condition_match:
                        break
                else:
                    model.save_model_state_dict()

            val_loss, (y_true, y_pred) = self.validate(model, test_loader, -1)
            metrics = metrics_handler.get_run_metrics(y_true, y_pred, verbose=True)
            self.tracker.log_metrics(metrics)
            if self.output_size == 1:
                preds = (y_pred.detach().cpu().numpy() >= 0.5).astype(int)
            else:
                preds = y_pred.argmax(axis=1).cpu().numpy()
            self.tracker.log_metrics(
                {
                    "conf_mat": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true.detach()
                        .cpu()
                        .numpy(),  # np.asarray(y_true.cpu()).astype(np.float32),
                        preds=(preds),
                    )
                }
            )
            self.tracker.finish()
            completed_folds.add(test_fold)
            with open(completed_folds_file, "w") as f:
                json.dump(sorted(list(completed_folds)), f)

        return train_loss, val_loss, metrics
