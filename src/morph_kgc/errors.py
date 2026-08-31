__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "Julián Arenas-Guerrero"
__email__ = "arenas.guerrero.julian@outlook.com"


class MorphKGCError(Exception):
    """Base Morph-KGC exception."""


class SparkBackendError(MorphKGCError):
    """Base Spark backend exception."""


class SparkDependencyError(SparkBackendError):
    """Raised when Spark execution is requested but PySpark is unavailable."""


class SparkUnsupportedFeature(SparkBackendError):
    """Raised when a Spark code path is selected but the feature is unsupported."""

    def __init__(self, feature, reason, suggestion='Use `execution_engine=pandas` for this workload.'):
        message = f'Spark backend does not support `{feature}`: {reason} {suggestion}'
        super().__init__(message)
