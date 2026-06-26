import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class LogisticClassifierMetrics:
    def __init__(self, config, logger, device):
        self.config = config
        self.device = device
        self.logger = logger

        self.criterion = nn.BCELoss()

    def _prepare_logits(self, logits):

        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32, device=self.device)

        return logits

    def _logits_to_predictions(self, logits):

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        logits = np.array(logits)

        # multiclass
        if logits.ndim > 1:
            return np.argmax(logits, axis=1)

        # binary
        return (logits >= 0.5).astype(int)

    def get_run_metrics(self, y_true, y_prob: np.ndarray, verbose=True) -> dict:
        y_t_np = y_true.detach().cpu().numpy()

        logits_tensor = self._prepare_logits(y_prob)

        # logits -> predictions
        y_pred = self._logits_to_predictions(logits_tensor)

        num_classes = len(np.unique(y_t_np))

        average = "binary" if num_classes == 2 else "macro"

        acc = accuracy_score(y_t_np, y_pred)
        error_rate = 1.0 - acc

        precision = precision_score(y_t_np, y_pred, average=average, zero_division=0)

        recall = recall_score(y_t_np, y_pred, average=average, zero_division=0)

        f_measure = f1_score(y_t_np, y_pred, average=average, zero_division=0)

        if verbose:
            self.logger.info(
                f"Error: {error_rate:.6f}"
                f" | Precision: {precision:.6f}"
                f" | Recall: {recall:.6f}"
                f" | F1: {f_measure:.6f}"
            )

        results = {
            "error_rate": error_rate,
            "precision": precision,
            "recall": recall,
            "f_measure": f_measure,
        }

        return results

    def get_fold_metrics(
        self, y_true: list[pd.DataFrame], y_prob: list[np.ndarray], verbose=True
    ) -> dict:

        errors = []
        precisions = []
        recalls = []
        f_measures = []

        for fold_idx, (y_t, logits) in enumerate(zip(y_true, y_prob)):

            y_t_np = np.array(y_t)

            logits_tensor = self._prepare_logits(logits).squeeze(1)

            # logits -> predictions
            y_pred = self._logits_to_predictions(logits_tensor)

            num_classes = len(np.unique(y_t_np))

            average = "binary" if num_classes == 2 else "macro"

            acc = accuracy_score(y_true, y_pred)
            error_rate = 1.0 - acc

            precision = precision_score(
                y_t_np, y_pred, average=average, zero_division=0
            )

            recall = recall_score(y_t_np, y_pred, average=average, zero_division=0)

            f_measure = f1_score(y_t_np, y_pred, average=average, zero_division=0)

            errors.append(error_rate)
            precisions.append(precision)
            recalls.append(recall)
            f_measures.append(f_measure)

            if verbose:
                self.logger.info(
                    f"Fold {fold_idx + 1}"
                    f" | Error: {error_rate:.6f}"
                    f" | Precision: {precision:.6f}"
                    f" | Recall: {recall:.6f}"
                    f" | F1: {f_measure:.6f}"
                )

        def summarize(metric_values):

            metric_values = np.array(metric_values)

            mean = np.mean(metric_values)

            if len(metric_values) > 1:
                std = np.std(metric_values, ddof=1)
                ci95 = 1.96 * std / np.sqrt(len(metric_values))
            else:
                std = 0.0
                ci95 = 0.0

            return {
                "mean": mean,
                "std": std,
                "ci95_lower": mean - ci95,
                "ci95_upper": mean + ci95,
            }

        results = {
            "error_rate": summarize(errors),
            "precision": summarize(precisions),
            "recall": summarize(recalls),
            "f_measure": summarize(f_measures),
        }

        if verbose:

            self.logger.info("\n===== FINAL RESULTS =====")

            for metric_name, values in results.items():

                self.logger.info(
                    f"{metric_name}"
                    f" | Mean: {values['mean']:.6f}"
                    f" | Std: {values['std']:.6f}"
                    f" | CI95%: "
                    f"[{values['ci95_lower']:.6f}, "
                    f"{values['ci95_upper']:.6f}]"
                )

        return results
