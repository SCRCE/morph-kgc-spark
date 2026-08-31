__author__ = "Ahmad Hammad"
__credits__ = ["Julián Arenas-Guerrero", "Ahmad Hammad"]

__license__ = "Apache-2.0"
__maintainer__ = "Ahmad Hammad"
__email__ = "Ahmad.Hammad@ieee.org"


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
