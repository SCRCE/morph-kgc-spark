__author__ = "Ahmad Hammad"
__credits__ = ["Julián Arenas-Guerrero", "Ahmad Hammad"]

__license__ = "Apache-2.0"
__maintainer__ = "Ahmad Hammad"
__email__ = "Ahmad.Hammad@ieee.org"


import os


def is_spark_runtime_available():
    try:
        import pyspark  # noqa: F401
    except ModuleNotFoundError:
        return False

    java_home = os.environ.get('JAVA_HOME')
    return bool(java_home) or _java_binary_exists()


def _java_binary_exists():
    for path in os.environ.get('PATH', '').split(os.pathsep):
        if os.path.isfile(os.path.join(path, 'java')):
            return True
    return False
