import logging

WHITE = "\033[0m"
COLORS = {
    "DEBUG": "\033[32m",  # green
    "INFO": WHITE,
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[1;31m",  # dark red
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        color = COLORS.get(levelname, "")
        record.levelname = f"{color}{levelname}{WHITE}"
        return super().format(record)


def setup_logger(logger: logging.Logger):
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        ColorFormatter(
            "%(asctime)s: %(filename)s: %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
    )

    logger.setLevel(logging.DEBUG)
    logger.handlers = [handler]
    logger.propagate = False


def get_logger(logger_name="test-name") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        setup_logger(logger)

    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
