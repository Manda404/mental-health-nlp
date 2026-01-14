# src/mh_nlp/utils/logging.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Union, cast
from types import FrameType

from loguru import logger

from mh_nlp.utils.root_finder import get_repository_root


class InterceptHandler(logging.Handler):
    """
    Handler pour intercepter les logs standard (logging) et les rediriger vers Loguru.
    Optimisé pour éviter les boucles infinies et conserver la stack trace.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Récupérer le niveau correspondant dans Loguru
        level: Union[str, int]
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Trouver l'origine du message pour que Loguru affiche le bon fichier/ligne
        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger(
    *,
    log_level: str = "INFO",
    log_dir: str | Path = "logs",  # Modification ici : accepte str ou Path
    app_name: str = "mh_nlp",
    json_logs: bool = False,
    enable_file: bool = True,
) -> None:
    """
    Configure Loguru : Console (couleurs) + Fichier (rotation) + Interception.
    """
    # 1. Nettoyage initial
    logger.remove()

    # 2. Configuration Console
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        level=log_level,
        format=console_format,
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    # 3. Configuration Fichier (si activé)
    if enable_file:
        log_path = Path(log_dir) # Convertit en Path au cas où c'est un str
        log_path.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path / f"{app_name}.log"),
            level=log_level,
            rotation="10 MB",
            retention="14 days",
            compression="zip",
            serialize=json_logs,
            enqueue=True,
        )

    # 4. Interception du logging standard
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Liste des bibliothèques bruyantes
    noisy_loggers = ["matplotlib", "PIL", "urllib3", "numexpr", "fsspec", "asyncio"]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.getLogger("transformers").setLevel(logging.ERROR)

    logger.success(f"Logging initialisé [Level: {log_level} | JSON: {json_logs}]")


def setup_logger_from_env() -> None:
    """Version simplifiée utilisant les variables d'environnement."""
    root = get_repository_root()
    # On s'assure que log_dir est passé comme une string pour setup_logger si on veut être strict,
    # mais avec la modif de signature ci-dessus, Path est désormais accepté.
    setup_logger(
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_dir=root / os.getenv("LOG_DIR", "logs"),
        app_name=os.getenv("APP_NAME", "mh_nlp"),
        json_logs=os.getenv("LOG_JSON", "0") == "1",
        enable_file=os.getenv("LOG_FILE", "1") == "1",
    )