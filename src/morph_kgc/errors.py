__author__ = "Ahmad Hammad"
__credits__ = ["Julián Arenas-Guerrero", "Ahmad Hammad"]

__license__ = "Apache-2.0"
__maintainer__ = "Ahmad Hammad"
__email__ = "Ahmad.Hammad@ieee.org"


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
