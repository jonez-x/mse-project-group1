import logging
from pathlib import Path
from datetime import datetime

import pytest

def pytest_configure(config: pytest.Config) -> None:
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"test_run_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    config._metadata = getattr(config, "_metadata", {})
    config._metadata["log_file"] = str(log_path)