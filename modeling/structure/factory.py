from torch import device as TorchDevice

from logger.base import Logger
from tracker.base_tracker import BaseTracker


class ModelingStructureFactory:
    """
    Base class for model structure factory.
    """

    def get(
        self,
        config: dict,
        logger: Logger,
        device: TorchDevice,
        tracker: BaseTracker = None,
    ):
        name = config.get("modeling", {}).get("structure", {}).get("name")

        if name == "resnet_ptb":
            from modeling.structure.ResNet_PTB import (
                ResNet_PTB,
            )

            return ResNet_PTB(config, logger, device)

        if name == "tcn":
            from modeling.structure.tcn import (
                TCNPred,
            )

            return TCNPred(config, logger, device)

        if name == "tcn_cross":
            from modeling.structure.tcn_cross_domain import (
                TCNPred,
            )

            return TCNPred(config, logger, device)

        else:
            raise ValueError(f"Unsupported ModelingStructureFactory name: {name}")
