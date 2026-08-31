__author__ = "Ahmad Hammad"
__credits__ = ["Julián Arenas-Guerrero", "Ahmad Hammad"]

__license__ = "Apache-2.0"
__maintainer__ = "Ahmad Hammad"
__email__ = "Ahmad.Hammad@ieee.org"


def get_backend(config):
    execution_engine = config.get_execution_engine()

    if execution_engine == 'pandas':
        from .pandas import PandasExecutionBackend
        return PandasExecutionBackend(config)
    if execution_engine == 'spark':
        from .spark import SparkExecutionBackend
        return SparkExecutionBackend(config)

    raise ValueError(f'Unknown execution engine `{execution_engine}`.')
