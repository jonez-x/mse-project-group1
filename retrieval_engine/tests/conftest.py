import logging
import os
import pytest
from datetime import datetime
from pathlib import Path


def pytest_configure(config: pytest.Config) -> None:
    """Setup logging for pytest runs with timestamped log files."""
    log_dir = Path(__file__).parent / "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"test_run_{timestamp}.log"

    # Configure logging format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    # Setup file and console handlers
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [file_handler, console_handler]  # Replace existing handlers

    # Store log path for other plugins
    config._metadata = getattr(config, "_metadata", {})
    config._metadata["log_file"] = str(log_path)
