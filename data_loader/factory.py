from data_loader.base import DataLoader
from logger.base import Logger


class DataLoaderFactory:
    """
    Base class for data_loader factory.
    """

    def get(self, config: dict, logger: Logger, device) -> DataLoader:
        name = config.get("data_loader", {}).get("name")

        if name == "ptbxl":
            from data_loader.ptbxl import PTBXL

            return PTBXL(config, logger, device)

        else:
            raise ValueError(f"Unsupported DataLoaderFactory name: {name}")
