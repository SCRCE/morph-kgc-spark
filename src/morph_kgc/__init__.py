__author__ = "Julián Arenas-Guerrero and Ahmad Hammad"
__credits__ = ["Julián Arenas-Guerrero", "Ahmad Hammad"]
__copyright__ = "Copyright © 2020 Julián Arenas-Guerrero"

__license__ = "Apache-2.0"
__maintainer__ = "Ahmad Hammad"
__email__ = "Ahmad.Hammad@ieee.org"


import logging

from .args_parser import load_config_from_argument
from .constants import LOGGING_NAMESPACE
from .engine import get_backend
from pathlib import Path


LOGGER = logging.getLogger(LOGGING_NAMESPACE)


def materialize_set(config, python_source=None):
    config = load_config_from_argument(config)
    return get_backend(config).materialize_set(python_source)


def materialize(config, python_source=None):
    config = load_config_from_argument(config)
    return get_backend(config).materialize_graph(python_source)


def materialize_oxigraph(config, python_source=None):
    config = load_config_from_argument(config)
    return get_backend(config).materialize_oxigraph(python_source)


def materialize_kafka(config, python_source=None):
    config = load_config_from_argument(config)
    return get_backend(config).materialize_kafka(python_source)


def translate_to_rml(mapping_path):
    from rdflib import Graph

    from .mapping.mapping_parser import MappingParser
    from .mapping.yarrrml import load_yarrrml

    parser = MappingParser(config=None)
    mapping_graph = Graph()
    mapping_path = Path(mapping_path)

    if mapping_path.suffix in ['.ttl', '.rdf', '.nt']:
        mapping_graph.parse(mapping_path, format='ttl')
    elif mapping_path.suffix in ['.yml', '.yaml', '.yarrrml']:
        mapping_graph = load_yarrrml(mapping_path)

    mapping_graph = parser._normalize_mapping_graph(mapping_graph)
    mapping_graph = parser._complete_and_validate_mapping(mapping_graph)

    return mapping_graph
