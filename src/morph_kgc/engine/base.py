__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "Julián Arenas-Guerrero"
__email__ = "arenas.guerrero.julian@outlook.com"


class ExecutionBackend:

    def __init__(self, config):
        self.config = config

    def materialize_set(self, python_source=None):
        raise NotImplementedError

    def materialize_graph(self, python_source=None):
        raise NotImplementedError

    def materialize_oxigraph(self, python_source=None):
        raise NotImplementedError

    def materialize_kafka(self, python_source=None):
        raise NotImplementedError

    def materialize_to_files(self, python_source=None):
        raise NotImplementedError
