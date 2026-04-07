import coloredlogs


def setup_logger(debug: bool = False):

    level_styles = {
        "debug": {"color": "green"},
        "info": {"color": "white"},
        "warning": {"color": "yellow"},
        "error": {"color": "red"},
        "critical": {"color": "red", "bold": True},
    }
    fmt = "%(levelname)s-%(message)s"
    level = "INFO"

    if debug:
        level = "DEBUG"
        fmt = f"%(name)s-{fmt}"
    coloredlogs.install(level=level, fmt=fmt, level_styles=level_styles)


if __name__ == "__main__":
    logger = setup_logger(debug=True)
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
