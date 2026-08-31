from configparser import ExtendedInterpolation

import pandas as pd
import pytest

from morph_kgc.config import Config
from morph_kgc.constants import RML_EXECUTION
from morph_kgc.errors import SparkUnsupportedFeature
from morph_kgc.fnml.function_executor import SparkFunctionExecutor
from morph_kgc.fnml.function_registry import FunctionRegistry
from morph_kgc.fnml.function_registry import _UDF_REGISTRY_CACHE


def _config(udf_path=''):
    config = Config(interpolation=ExtendedInterpolation())
    config.read_string(
        '[CONFIGURATION]\n'
        f'udfs={udf_path}\n'
        '\n'
        '[DataSource]\n'
        'mappings=/tmp/mapping.ttl\n'
    )
    config.complete_configuration_with_defaults()
    return config


def test_function_registry_loads_builtin_metadata():
    registry = FunctionRegistry(_config())

    registered_function = registry.get('http://users.ugent.be/~bjdmeest/function/grel.ttl#toUpperCase')

    assert registered_function.metadata.function_id.endswith('#toUpperCase')
    assert registered_function.metadata.source == 'builtin'
    assert 'pandas' in registered_function.metadata.supported_backends


def test_function_registry_marks_scalar_builtin_for_pandas_udf_strategy():
    registry = FunctionRegistry(_config())

    registered_function = registry.get('http://users.ugent.be/~bjdmeest/function/grel.ttl#string_md5')

    assert registered_function.metadata.backend_strategy == 'spark-pandas-udf'
    assert 'spark-pandas-udf' in registered_function.metadata.supported_backends
    assert 'spark-python-udf' in registered_function.metadata.supported_backends


def test_function_registry_marks_array_builtin_for_map_in_pandas_strategy():
    registry = FunctionRegistry(_config())

    registered_function = registry.get('http://users.ugent.be/~bjdmeest/function/grel.ttl#array_uniques')

    assert registered_function.metadata.backend_strategy == 'spark-mapInPandas'
    assert registered_function.metadata.supported_backends == ('pandas',)


def test_function_registry_marks_string_match_as_array_builtin():
    registry = FunctionRegistry(_config())

    registered_function = registry.get('http://users.ugent.be/~bjdmeest/function/grel.ttl#string_match')

    assert registered_function.metadata.cardinality == 'array'
    assert registered_function.metadata.backend_strategy == 'spark-mapInPandas'
    assert registered_function.metadata.supported_backends == ('pandas',)


def test_function_registry_marks_date_now_as_nondeterministic_builtin():
    registry = FunctionRegistry(_config())

    registered_function = registry.get('http://users.ugent.be/~bjdmeest/function/grel.ttl#date_now')

    assert registered_function.metadata.deterministic is False
    assert registered_function.metadata.source == 'builtin'


@pytest.mark.parametrize(
    'function_id',
    [
        'http://users.ugent.be/~bjdmeest/function/grel.ttl#math_randomNumber',
        'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#uuid',
    ],
)
def test_function_registry_marks_other_nondeterministic_builtins(function_id):
    registry = FunctionRegistry(_config())

    registered_function = registry.get(function_id)

    assert registered_function.metadata.deterministic is False
    assert registered_function.metadata.source == 'builtin'


def test_function_registry_loads_udf_metadata():
    registry = FunctionRegistry(_config('test/rml-fnml/udf/udf.py'))

    registered_function = registry.get('http://example.com/toUpperCase')

    assert registered_function.metadata.source == 'udf'
    assert registered_function.metadata.parameter_map == {
        'text': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParam'
    }


def test_function_registry_loads_declared_udf_metadata(tmp_path):
    udf_path = tmp_path / 'udf_metadata.py'
    udf_path.write_text(
        '@udf(\n'
        '    fun_id="http://example.com/sorted-array",\n'
        '    cardinality="array",\n'
        '    output_type="string",\n'
        '    null_policy="preserve",\n'
        '    deterministic=False,\n'
        '    backend_strategy="spark-mapInPandas",\n'
        '    supported_backends=("pandas", "spark-mapInPandas"),\n'
        '    items="http://example.com/items"\n'
        ')\n'
        'def sorted_array(items):\n'
        '    return sorted(eval(items))\n',
        encoding='utf-8',
    )

    registry = FunctionRegistry(_config(str(udf_path)))
    registered_function = registry.get('http://example.com/sorted-array')

    assert registered_function.metadata.cardinality == 'array'
    assert registered_function.metadata.output_type == 'string'
    assert registered_function.metadata.null_policy == 'preserve'
    assert registered_function.metadata.deterministic is False
    assert registered_function.metadata.backend_strategy == 'spark-mapInPandas'
    assert registered_function.metadata.supported_backends == ('pandas', 'spark-mapInPandas')
    assert registered_function.metadata.parameter_map == {'items': 'http://example.com/items'}


def test_spark_function_executor_raises_contextual_error_for_fnml():
    config = _config('test/rml-fnml/udf/udf.py')
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-1',
            'function_map_value': 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'name',
        },
        {
            'function_execution': '#exec-1',
            'function_map_value': 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'separator',
        }
    ])

    with pytest.raises(SparkUnsupportedFeature, match='string_split_explode'):
        executor.assert_supported(fnml_df, '#exec-1', triples_map_id='#TM1', position='#TM1:object')


def test_spark_function_executor_accepts_native_uppercase_fnml():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-native',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#toUpperCase',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'name',
        }
    ])

    assert executor.is_native_supported(fnml_df, '#exec-native') is True
    executor.assert_supported(fnml_df, '#exec-native', triples_map_id='#TM2', position='#TM2:object')


def test_spark_function_executor_accepts_nested_native_fnml():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#toUpperCase',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_replace',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'name',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_replace',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_find',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ' ',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_replace',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_replace',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': '-',
        },
    ])

    assert executor.is_native_supported(fnml_df, '#outer') is True


def test_spark_function_executor_detects_nondeterministic_nested_execution():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#date_diff',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_datetime_d',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#date_diff',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_datetime_d2',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': '2023-10-01T00:00:00',
        },
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#date_diff',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_timeunit',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': 'seconds',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#date_now',
            'parameter_map_value': 'http://example.com/unused',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': '',
        },
    ])

    assert executor.is_execution_deterministic(fnml_df, '#inner') is False
    assert executor.is_execution_deterministic(fnml_df, '#outer') is False


def test_spark_function_executor_detects_deterministic_nested_execution():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#date_diff',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_datetime_d',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#date_diff',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_datetime_d2',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': '2023-10-01T00:00:00',
        },
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#date_diff',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_timeunit',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': 'seconds',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#date_toDate',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': '2023-10-02T00:00:00Z',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#date_toDate',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_pattern',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': 'yyyy-MM-ddTHH:mm:ssZ',
        },
    ])

    assert executor.is_execution_deterministic(fnml_df, '#inner') is True
    assert executor.is_execution_deterministic(fnml_df, '#outer') is True


def test_spark_function_executor_accepts_native_array_join_over_native_string_split():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ' - ',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'model',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ' ',
        },
    ])

    assert executor.is_native_array_supported(fnml_df, '#inner') is True
    assert executor.is_native_supported(fnml_df, '#outer') is True
    executor.assert_supported(fnml_df, '#outer', triples_map_id='#TM2b', position='#TM2b:object')


def test_spark_function_executor_accepts_native_array_reverse_over_native_string_split():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_reverse',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'values',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ',',
        },
    ])

    assert executor.is_native_array_supported(fnml_df, '#inner') is True
    assert executor.is_native_array_supported(fnml_df, '#outer') is True
    executor.assert_supported(fnml_df, '#outer', triples_map_id='#TM2c', position='#TM2c:object')


def test_spark_function_executor_accepts_native_array_sort_over_native_string_split():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_sort',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'values',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ',',
        },
    ])

    assert executor.is_native_array_supported(fnml_df, '#inner') is True
    assert executor.is_native_array_supported(fnml_df, '#outer') is True
    executor.assert_supported(fnml_df, '#outer', triples_map_id='#TM2d', position='#TM2d:object')


def test_spark_function_executor_accepts_scalar_udf_path():
    config = _config('test/rml-fnml/udf/udf.py')
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-udf',
            'function_map_value': 'http://example.com/toUpperCase',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParam',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'name',
        }
    ])

    assert executor.is_scalar_pandas_udf_supported(fnml_df, '#exec-udf') is True
    assert executor.is_scalar_python_udf_supported(fnml_df, '#exec-udf') is True
    executor.assert_supported(fnml_df, '#exec-udf', triples_map_id='#TM3', position='#TM3:object')


def test_spark_function_executor_accepts_declared_batch_udf_path(tmp_path):
    udf_path = tmp_path / 'batch_udf.py'
    udf_path.write_text(
        '@udf(\n'
        '    fun_id="http://example.com/split-array",\n'
        '    cardinality="array",\n'
        '    backend_strategy="spark-mapInPandas",\n'
        '    values="http://example.com/value"\n'
        ')\n'
        'def split_array(values):\n'
        '    return values.split("|")\n',
        encoding='utf-8',
    )

    config = _config(str(udf_path))
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-batch-udf',
            'function_map_value': 'http://example.com/split-array',
            'parameter_map_value': 'http://example.com/value',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'values',
        }
    ])

    assert executor.is_batch_map_in_pandas_supported(fnml_df, '#exec-batch-udf') is True
    executor.assert_supported(fnml_df, '#exec-batch-udf', triples_map_id='#TM3c', position='#TM3c:object')


def test_spark_function_executor_routes_declared_batch_udf_to_map_in_pandas(monkeypatch, tmp_path):
    udf_path = tmp_path / 'batch_udf.py'
    udf_path.write_text(
        '@udf(\n'
        '    fun_id="http://example.com/split-array",\n'
        '    cardinality="array",\n'
        '    backend_strategy="spark-mapInPandas",\n'
        '    values="http://example.com/value"\n'
        ')\n'
        'def split_array(values):\n'
        '    return values.split("|")\n',
        encoding='utf-8',
    )

    config = _config(str(udf_path))
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-batch-udf',
            'function_map_value': 'http://example.com/split-array',
            'parameter_map_value': 'http://example.com/value',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'values',
        }
    ])

    class FakeData:
        columns = ['values']

    calls = {'value': 0}

    def fake_apply_map_in_pandas(data, fnml_df_arg, fnml_execution_arg, explode_output=True):
        calls['value'] += 1
        assert fnml_execution_arg == '#exec-batch-udf'
        assert explode_output is True
        return 'mapped-data'

    monkeypatch.setattr(executor, 'apply_map_in_pandas_execution', fake_apply_map_in_pandas)

    result_data, output_column = executor.apply_execution(FakeData(), fnml_df, '#exec-batch-udf', materializer=None)

    assert result_data == 'mapped-data'
    assert output_column == '__fnml_exec_batch_udf'
    assert calls['value'] == 1


def test_spark_function_executor_explodes_top_level_native_array_execution(monkeypatch):
    import sys
    from types import ModuleType

    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-native-array',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'model',
        },
        {
            'function_execution': '#exec-native-array',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ' ',
        },
    ])

    class FakeData:
        columns = ['model']

        def __init__(self):
            self.calls = []

        def withColumn(self, name, value):
            self.calls.append((name, value))
            return self

    fake_data = FakeData()

    monkeypatch.setattr(executor, 'is_native_exploding_supported', lambda *_args, **_kwargs: False)
    monkeypatch.setattr(executor, 'is_native_array_supported', lambda *_args, **_kwargs: True)
    monkeypatch.setattr(executor, 'build_native_array_column', lambda *_args, **_kwargs: 'native-array-column')

    class FakeFunctions:
        @staticmethod
        def col(name):
            return f'col({name})'

        @staticmethod
        def explode(value):
            return f'explode({value})'

    fake_sql_module = ModuleType('pyspark.sql')
    fake_sql_module.functions = FakeFunctions
    monkeypatch.setitem(sys.modules, 'pyspark.sql', fake_sql_module)

    result_data, output_column = executor.apply_execution(
        fake_data,
        fnml_df,
        '#exec-native-array',
        materializer=None,
    )

    assert result_data is fake_data
    assert output_column == '__fnml_exec_native_array'
    assert fake_data.calls == [
        ('__fnml_exec_native_array', 'native-array-column'),
        ('__fnml_exec_native_array', 'explode(col(__fnml_exec_native_array))'),
    ]


def test_spark_function_executor_accepts_nested_scalar_inside_batch_udf(tmp_path):
    udf_path = tmp_path / 'batch_udf.py'
    udf_path.write_text(
        '@udf(\n'
        '    fun_id="http://example.com/split-array",\n'
        '    cardinality="array",\n'
        '    backend_strategy="spark-mapInPandas",\n'
        '    values="http://example.com/value"\n'
        ')\n'
        'def split_array(values):\n'
        '    return values.split("|")\n',
        encoding='utf-8',
    )

    config = _config(str(udf_path))
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://example.com/split-array',
            'parameter_map_value': 'http://example.com/value',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#toUpperCase',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'values',
        },
    ])

    assert executor.is_batch_map_in_pandas_supported(fnml_df, '#outer') is True
    executor.assert_supported(fnml_df, '#outer', triples_map_id='#TM3d', position='#TM3d:object')


def test_spark_function_executor_accepts_scalar_udf_with_nested_batch_dependency():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ', ',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_uniques',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'authors',
        },
    ])

    assert executor.is_scalar_pandas_udf_supported(fnml_df, '#outer') is True
    executor.assert_supported(fnml_df, '#outer', triples_map_id='#TM3e', position='#TM3e:object')


def test_spark_function_executor_accepts_scalar_udf_with_nested_declared_batch_udf(tmp_path):
    udf_path = tmp_path / 'batch_udf.py'
    udf_path.write_text(
        '@udf(\n'
        '    fun_id="http://example.com/split-array",\n'
        '    cardinality="array",\n'
        '    backend_strategy="spark-mapInPandas",\n'
        '    values="http://example.com/value"\n'
        ')\n'
        'def split_array(values):\n'
        '    return values.split("|")\n',
        encoding='utf-8',
    )

    config = _config(str(udf_path))
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ', ',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://example.com/split-array',
            'parameter_map_value': 'http://example.com/value',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'values',
        },
    ])

    assert executor.is_scalar_pandas_udf_supported(fnml_df, '#outer') is True
    executor.assert_supported(fnml_df, '#outer', triples_map_id='#TM3f', position='#TM3f:object')


def test_spark_function_executor_prepares_nested_batch_dependency_before_scalar_build(monkeypatch):
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ', ',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_uniques',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'authors',
        },
    ])

    class FakeData:
        def __init__(self, columns):
            self.columns = columns

        def withColumn(self, column_name, value):
            return FakeData(self.columns + [column_name])

    calls = []

    def fake_apply_map_in_pandas(data, fnml_df_arg, fnml_execution_arg, explode_output=True):
        calls.append(('batch', fnml_execution_arg, explode_output))
        return FakeData(data.columns + ['__fnml_inner'])

    def fake_build_pandas(data, fnml_df_arg, fnml_execution_arg, materializer):
        calls.append(('scalar', fnml_execution_arg, tuple(data.columns)))
        assert '__fnml_inner' in data.columns
        return 'scalar-column'

    monkeypatch.setattr(executor, 'apply_map_in_pandas_execution', fake_apply_map_in_pandas)
    monkeypatch.setattr(executor, 'build_pandas_udf_column', fake_build_pandas)

    result_data, output_column = executor.apply_execution(FakeData(['authors']), fnml_df, '#outer', materializer=None)

    assert output_column == '__fnml_outer'
    assert result_data.columns[-1] == '__fnml_outer'
    assert calls[0] == ('batch', '#inner', False)
    assert calls[1][0] == 'scalar'


def test_spark_function_executor_falls_back_to_python_udf_with_nested_batch_dependency(monkeypatch):
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': RML_EXECUTION,
            'value_map_value': '#inner',
        },
        {
            'function_execution': '#outer',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ', ',
        },
        {
            'function_execution': '#inner',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_uniques',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'authors',
        },
    ])

    class FakeData:
        def __init__(self, columns):
            self.columns = columns

        def withColumn(self, column_name, value):
            return FakeData(self.columns + [column_name])

    calls = []

    def fake_apply_map_in_pandas(data, fnml_df_arg, fnml_execution_arg, explode_output=True):
        calls.append(('batch', fnml_execution_arg, explode_output))
        return FakeData(data.columns + ['__fnml_inner'])

    def fake_build_pandas(*args, **kwargs):
        raise ModuleNotFoundError('simulate pandas_udf unavailable')

    def fake_build_python(data, fnml_df_arg, fnml_execution_arg, materializer):
        calls.append(('python', fnml_execution_arg, tuple(data.columns)))
        assert '__fnml_inner' in data.columns
        return 'python-column'

    monkeypatch.setattr(executor, 'apply_map_in_pandas_execution', fake_apply_map_in_pandas)
    monkeypatch.setattr(executor, 'build_pandas_udf_column', fake_build_pandas)
    monkeypatch.setattr(executor, 'build_python_udf_column', fake_build_python)

    result_data, output_column = executor.apply_execution(FakeData(['authors']), fnml_df, '#outer', materializer=None)

    assert output_column == '__fnml_outer'
    assert result_data.columns[-1] == '__fnml_outer'
    assert calls[0] == ('batch', '#inner', False)
    assert calls[1][0] == 'python'


def test_spark_function_executor_accepts_scalar_builtin_pandas_udf_path():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-hash',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_md5',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'name',
        }
    ])

    assert executor.is_scalar_pandas_udf_supported(fnml_df, '#exec-hash') is True
    assert executor.is_scalar_python_udf_supported(fnml_df, '#exec-hash') is True
    executor.assert_supported(fnml_df, '#exec-hash', triples_map_id='#TM3b', position='#TM3b:object')


def test_spark_function_executor_rejects_exploding_builtin_for_scalar_udf_path():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-explode',
            'function_map_value': 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'name',
        },
        {
            'function_execution': '#exec-explode',
            'function_map_value': 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/constant',
            'value_map_value': ',',
        },
    ])

    assert executor.is_scalar_python_udf_supported(fnml_df, '#exec-explode') is False
    assert executor.is_native_exploding_supported(fnml_df, '#exec-explode') is True
    executor.assert_supported(fnml_df, '#exec-explode', triples_map_id='#TM4', position='#TM4:object')


def test_spark_function_executor_rejects_non_constant_separator_for_native_explode():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-explode',
            'function_map_value': 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'name',
        },
        {
            'function_execution': '#exec-explode',
            'function_map_value': 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'sep_col',
        },
    ])

    assert executor.is_native_exploding_supported(fnml_df, '#exec-explode') is False
    with pytest.raises(SparkUnsupportedFeature, match='string_split_explode'):
        executor.assert_supported(fnml_df, '#exec-explode', triples_map_id='#TM4b', position='#TM4b:object')


def test_spark_function_executor_accepts_batch_map_in_pandas_array_path():
    config = _config()
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-array',
            'function_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_uniques',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'authors',
        }
    ])

    assert executor.is_batch_map_in_pandas_supported(fnml_df, '#exec-array') is True
    executor.assert_supported(fnml_df, '#exec-array', triples_map_id='#TM4c', position='#TM4c:object')


def test_udf_registry_is_cached_per_process(monkeypatch, tmp_path):
    udf_path = tmp_path / 'udf.py'
    udf_path.write_text(
        '@udf(fun_id="http://example.com/cached", value="http://example.com/value")\n'
        'def cached(value):\n'
        '    return value\n',
        encoding='utf-8',
    )

    call_count = {'value': 0}
    original_loader = FunctionRegistry._load_udfs_from_path

    def counting_loader(self, path):
        call_count['value'] += 1
        return original_loader(self, path)

    monkeypatch.setattr(FunctionRegistry, '_load_udfs_from_path', counting_loader)
    _UDF_REGISTRY_CACHE.pop(str(udf_path), None)

    FunctionRegistry(_config(str(udf_path))).get('http://example.com/cached')
    FunctionRegistry(_config(str(udf_path))).get('http://example.com/cached')

    assert call_count['value'] == 1


def test_udf_registry_resolves_distributed_worker_path(monkeypatch, tmp_path):
    local_udf_path = tmp_path / 'missing' / 'udf.py'
    distributed_dir = tmp_path / 'distributed'
    distributed_dir.mkdir()
    distributed_udf_path = distributed_dir / 'udf.py'
    distributed_udf_path.write_text(
        '@udf(fun_id="http://example.com/distributed", value="http://example.com/value")\n'
        'def distributed(value):\n'
        '    return value\n',
        encoding='utf-8',
    )

    monkeypatch.setattr('morph_kgc.fnml.function_registry.sys.path', [str(distributed_dir)])

    registry = FunctionRegistry(_config(str(local_udf_path)))
    registered_function = registry.get('http://example.com/distributed')

    assert registered_function.metadata.source == 'udf'
    assert registered_function.metadata.parameter_map == {'value': 'http://example.com/value'}


def test_spark_executor_falls_back_to_python_udf_when_pandas_udf_is_unavailable(monkeypatch):
    config = _config('test/rml-fnml/udf/udf.py')
    executor = SparkFunctionExecutor(config)
    fnml_df = pd.DataFrame([
        {
            'function_execution': '#exec-udf',
            'function_map_value': 'http://example.com/toUpperCase',
            'parameter_map_value': 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParam',
            'value_map_type': 'http://w3id.org/rml/reference',
            'value_map_value': 'name',
        }
    ])

    pandas_calls = {'value': 0}
    python_calls = {'value': 0}

    def fake_build_pandas(*args, **kwargs):
        pandas_calls['value'] += 1
        raise ModuleNotFoundError('pyspark pandas_udf unavailable')

    def fake_build_python(*args, **kwargs):
        python_calls['value'] += 1
        return 'python-udf-column'

    monkeypatch.setattr(executor, 'build_pandas_udf_column', fake_build_pandas)
    monkeypatch.setattr(executor, 'build_python_udf_column', fake_build_python)

    result = executor.build_column(data=None, fnml_df=fnml_df, fnml_execution='#exec-udf', materializer=None)

    assert result == 'python-udf-column'
    assert pandas_calls['value'] == 1
    assert python_calls['value'] == 1


def test_spark_executor_uses_dataframe_transform_output_column_name():
    executor = SparkFunctionExecutor(_config())

    assert executor.get_execution_output_column('#Execution') == '__fnml_Execution'
