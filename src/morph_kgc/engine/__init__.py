__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "Julián Arenas-Guerrero"
__email__ = "arenas.guerrero.julian@outlook.com"


def get_backend(config):
    execution_engine = config.get_execution_engine()

    if execution_engine == 'pandas':
        from .pandas import PandasExecutionBackend
        return PandasExecutionBackend(config)
    if execution_engine == 'spark':
        from .spark import SparkExecutionBackend
        return SparkExecutionBackend(config)

    raise ValueError(f'Unknown execution engine `{execution_engine}`.')
