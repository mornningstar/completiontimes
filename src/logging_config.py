import logging


def setup_logging(level: int = logging.DEBUG):
    """ Configure application-wide logging settings. """
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('pymongo').setLevel(logging.WARNING)

    logging.basicConfig(
        level=level,  # Adjust the level as needed
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )