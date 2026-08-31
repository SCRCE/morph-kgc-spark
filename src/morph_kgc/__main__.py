__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "Julián Arenas-Guerrero"
__email__ = "arenas.guerrero.julian@outlook.com"

import logging

from .args_parser import load_config_from_command_line
from .constants import LOGGING_NAMESPACE
from .engine import get_backend


LOGGER = logging.getLogger(LOGGING_NAMESPACE)


def main():
    config = load_config_from_command_line()
    get_backend(config).materialize_to_files()


if __name__ == "__main__":
    main()
