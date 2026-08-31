__author__ = "Ahmad Hammad"
__credits__ = ["Julián Arenas-Guerrero", "Ahmad Hammad"]

__license__ = "Apache-2.0"
__maintainer__ = "Ahmad Hammad"
__email__ = "Ahmad.Hammad@ieee.org"


import sys
import importlib.util

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import ModuleType

from .built_in_functions import bif_dict


UDF_METADATA_KEYS = {
    'output_type',
    'cardinality',
    'null_policy',
    'deterministic',
    'supported_backends',
    'backend_strategy',
}

UDF_DICT_DECORATOR_CODE = """
udf_dict = {}
def udf(fun_id, **params):
    def wrapper(funct):
        metadata = {key: value for key, value in params.items() if key in {
            'output_type',
            'cardinality',
            'null_policy',
            'deterministic',
            'supported_backends',
            'backend_strategy',
        }}
        parameter_map = {key: value for key, value in params.items() if key not in metadata}
        udf_dict[fun_id] = {}
        udf_dict[fun_id]['function'] = funct
        udf_dict[fun_id]['parameters'] = parameter_map
        udf_dict[fun_id]['metadata'] = metadata
        return funct
    return wrapper
"""

SPARK_NATIVE_FUNCTION_IDS = {
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#toUpperCase',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#toLowerCase',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_replace',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_substring',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_reverse',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_sort',
    'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#concat',
    'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode',
}

NONDETERMINISTIC_FUNCTION_IDS = {
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#date_now',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#math_randomNumber',
    'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#uuid',
}

ARRAY_CARDINALITY_FUNCTION_IDS = {
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_splitByLengths',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_splitByCharType',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_match',
    'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_get',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_slice',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_reverse',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_uniques',
    'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_sort',
}

EXPLODING_FUNCTION_IDS = {
    'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode',
}

_UDF_REGISTRY_CACHE = {}


@dataclass(frozen=True)
class FunctionMetadata:
    function_id: str
    parameter_map: dict
    output_type: str = 'python'
    cardinality: str = 'scalar'
    null_policy: str = 'propagate'
    deterministic: bool = True
    supported_backends: tuple = field(default_factory=lambda: ('pandas',))
    backend_strategy: str = 'pandas-row-wise'
    source: str = 'builtin'


@dataclass(frozen=True)
class RegisteredFunction:
    metadata: FunctionMetadata
    function: object


class FunctionRegistry:

    def __init__(self, config):
        self.config = config
        self._functions = {}
        self._udfs_loaded = False
        self._load_builtins()

    def _load_builtins(self):
        for function_id, function_entry in bif_dict.items():
            self._functions[function_id] = RegisteredFunction(
                metadata=self._build_metadata(
                    function_id,
                    function_entry['parameters'],
                    source='builtin',
                    udf_metadata=None,
                ),
                function=function_entry['function'],
            )

    def _load_udfs(self):
        if self._udfs_loaded:
            return

        udf_path = self.config.get_udfs()
        if not udf_path:
            self._udfs_loaded = True
            return

        cached_functions = _UDF_REGISTRY_CACHE.get(udf_path)
        if cached_functions is None:
            cached_functions = self._load_udfs_from_path(udf_path)
            _UDF_REGISTRY_CACHE[udf_path] = cached_functions

        for function_id, function_entry in cached_functions.items():
            self._functions[function_id] = RegisteredFunction(
                metadata=self._build_metadata(
                    function_id,
                    function_entry['parameters'],
                    source='udf',
                    udf_metadata=function_entry.get('metadata'),
                ),
                function=function_entry['function'],
            )
        self._udfs_loaded = True

    def _load_udfs_from_path(self, udf_path):
        resolved_udf_path = self._resolve_udf_source_path(udf_path)
        with open(resolved_udf_path, 'r', encoding='utf-8') as file_handle:
            udfs_code = file_handle.read()

        module_name = f"morph_kgc_udfs_{Path(resolved_udf_path).resolve().stem}"
        udf_module = ModuleType(module_name)
        sys.modules[module_name] = udf_module
        exec(f'{UDF_DICT_DECORATOR_CODE}{udfs_code}', udf_module.__dict__)

        return udf_module.udf_dict

    def _resolve_udf_source_path(self, udf_path):
        candidate_path = Path(udf_path)
        if candidate_path.exists():
            return str(candidate_path)

        basename = candidate_path.name
        for search_path in sys.path:
            if not search_path:
                continue
            distributed_candidate = Path(search_path) / basename
            if distributed_candidate.exists():
                return str(distributed_candidate)

        module_spec = importlib.util.find_spec(candidate_path.stem)
        if module_spec and module_spec.origin and Path(module_spec.origin).exists():
            return module_spec.origin

        raise FileNotFoundError(
            f'Could not resolve UDF file `{udf_path}` locally or from distributed Spark worker paths.'
        )

    def _build_metadata(self, function_id, parameter_map, source, udf_metadata=None):
        udf_metadata = udf_metadata or {}
        cardinality = udf_metadata.get('cardinality')
        if cardinality is None:
            cardinality = 'array' if function_id in ARRAY_CARDINALITY_FUNCTION_IDS else 'scalar'

        if function_id in SPARK_NATIVE_FUNCTION_IDS:
            default_supported_backends = ('pandas', 'spark-native')
            default_backend_strategy = 'spark-native'
        elif cardinality == 'scalar':
            default_supported_backends = ('pandas', 'spark-pandas-udf', 'spark-python-udf')
            default_backend_strategy = 'spark-pandas-udf'
        else:
            default_supported_backends = ('pandas',)
            default_backend_strategy = 'spark-mapInPandas'

        supported_backends = udf_metadata.get('supported_backends', default_supported_backends)
        if isinstance(supported_backends, list):
            supported_backends = tuple(supported_backends)
        backend_strategy = udf_metadata.get('backend_strategy', default_backend_strategy)

        return FunctionMetadata(
            function_id=function_id,
            parameter_map=parameter_map,
            output_type=udf_metadata.get('output_type', 'python'),
            cardinality=cardinality,
            null_policy=udf_metadata.get('null_policy', 'propagate'),
            deterministic=udf_metadata.get(
                'deterministic',
                function_id not in NONDETERMINISTIC_FUNCTION_IDS,
            ),
            supported_backends=supported_backends,
            backend_strategy=backend_strategy,
            source=source,
        )

    def get(self, function_id):
        if function_id not in self._functions:
            self._load_udfs()
        return self._functions[function_id]

    def has(self, function_id):
        if function_id not in self._functions:
            self._load_udfs()
        return function_id in self._functions


def is_exploding_function(function_id):
    return function_id in EXPLODING_FUNCTION_IDS
