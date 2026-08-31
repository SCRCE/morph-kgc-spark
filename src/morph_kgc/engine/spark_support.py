__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "arenas.guerrero.julian@outlook.com"
__email__ = "arenas.guerrero.julian@outlook.com"


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
