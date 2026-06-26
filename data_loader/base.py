import numpy as np
import pandas as pd
from typing import Tuple
from torch import device as TorchDevice

from logging import Logger
from abc import ABC, abstractmethod


class DataLoader(ABC):
    def __init__(self, config: dict, logger: Logger, device: TorchDevice):
        self.config = config
        self.logger = logger
        self.device = device

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        load raw data

        returns:
            Tuple: (X: data, y: labels)
        """
        pass
