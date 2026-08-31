from configparser import ExtendedInterpolation
from datetime import time
from pathlib import Path
import sys
import types

import pandas as pd
import pytest

from morph_kgc import translate_to_rml
from morph_kgc.args_parser import load_config_from_argument
from morph_kgc.config import Config
from morph_kgc.engine import get_backend
from morph_kgc.engine.spark import SparkExecutionBackend
from morph_kgc.errors import SparkDependencyError
from morph_kgc.errors import SparkUnsupportedFeature


def test_translate_to_rml_keeps_public_graph_api_available():
    mapping_graph = translate_to_rml('test/rml-core/csv/RMLTC0000/mapping.ttl')

    assert len(mapping_graph) > 0


def test_execution_engine_defaults_to_pandas():
    config = Config(interpolation=ExtendedInterpolation())
    config.read_string('[DataSource1]\nmappings=/tmp/mapping.ttl\n')
    config.complete_configuration_with_defaults()

    assert config.get_execution_engine() == 'pandas'


def test_invalid_execution_engine_is_rejected():
    with pytest.raises(ValueError, match='execution_engine'):
        load_config_from_argument(
            '[CONFIGURATION]\n'
            'execution_engine=duckdb\n'
            '\n'
            '[DataSource1]\n'
            'mappings=/tmp/mapping.ttl\n'
        )


def test_pandas_backend_is_selected_by_default():
    config = load_config_from_argument('[DataSource1]\nmappings=/tmp/mapping.ttl\n')

    backend = get_backend(config)

    assert backend.__class__.__name__ == 'PandasExecutionBackend'


def test_spark_dependency_error_is_lazy_and_clear(monkeypatch):
    config = load_config_from_argument(
        '[CONFIGURATION]\n'
        'execution_engine=spark\n'
        '\n'
        '[DataSource1]\n'
        'mappings=/tmp/mapping.ttl\n'
    )

    def fake_import_module(name):
        if name == 'pyspark.sql':
            raise ModuleNotFoundError("No module named 'pyspark'")
        raise AssertionError(f'unexpected import: {name}')

    monkeypatch.setattr('importlib.import_module', fake_import_module)

    with pytest.raises(SparkDependencyError, match='PySpark'):
        get_backend(config)


def test_spark_library_api_fails_explicitly_without_fallback(monkeypatch):
    monkeypatch.setattr('importlib.import_module', lambda name: object())

    config = load_config_from_argument(
        '[CONFIGURATION]\n'
        'execution_engine=spark\n'
        '\n'
        '[DataSource1]\n'
        'mappings=/tmp/mapping.ttl\n'
    )

    backend = SparkExecutionBackend(config)

    with pytest.raises(SparkUnsupportedFeature, match='materialize_set'):
        backend.materialize_set()

    with pytest.raises(SparkUnsupportedFeature, match='materialize'):
        backend.materialize_graph()

    with pytest.raises(SparkUnsupportedFeature, match='materialize_oxigraph'):
        backend.materialize_oxigraph()

    with pytest.raises(SparkUnsupportedFeature, match='materialize_kafka'):
        backend.materialize_kafka()


def test_spark_backend_registers_udf_file_with_spark_context(monkeypatch, tmp_path):
    monkeypatch.setattr('importlib.import_module', lambda name: object())

    udf_path = tmp_path / 'udf.py'
    udf_path.write_text('def placeholder():\n    return None\n', encoding='utf-8')

    config = load_config_from_argument(
        '[CONFIGURATION]\n'
        f'udfs={udf_path}\n'
        'execution_engine=spark\n'
        '\n'
        '[DataSource1]\n'
        'mappings=/tmp/mapping.ttl\n'
    )

    backend = SparkExecutionBackend(config)

    calls = []

    class FakeSparkContext:
        def addPyFile(self, path):
            calls.append(path)

    class FakeSparkSession:
        sparkContext = FakeSparkContext()

    backend._distribute_udf_dependencies(FakeSparkSession())

    assert len(calls) == 1
    registered_path = Path(calls[0])
    assert registered_path.name == 'morph_kgc_udfs_udf.py'
    assert registered_path.read_text(encoding='utf-8') == udf_path.read_text(encoding='utf-8')


def test_spark_backend_disables_local_multiprocessing(monkeypatch):
    monkeypatch.setattr('importlib.import_module', lambda name: object())

    config = load_config_from_argument(
        '[CONFIGURATION]\n'
        'execution_engine=spark\n'
        'number_of_processes=8\n'
        '\n'
        '[DataSource1]\n'
        'mappings=/tmp/mapping.ttl\n'
    )

    backend = SparkExecutionBackend(config)

    assert backend.config.get_number_of_processes() == 1


def test_spark_backend_drops_redundant_rdb_source_rows():
    rml_df = pd.DataFrame(
        [
            {
                'triples_map_id': '#TM0',
                'source_name': 'DB1',
                'source_type': 'RDB',
                'logical_source_type': 'http://w3id.org/rml/query',
                'logical_source_value': 'SELECT id FROM demo',
                'mapping_partition': '0-0-0-0',
                'triples_map_type': 'http://w3id.org/rml/TriplesMap',
                'subject_map_type': 'http://w3id.org/rml/template',
                'subject_map_value': 'http://example.com/{id}',
                'subject_termtype': 'http://w3id.org/rml/IRI',
                'predicate_map_type': 'http://w3id.org/rml/constant',
                'predicate_map_value': 'http://example.com/p',
                'object_map_type': 'http://w3id.org/rml/reference',
                'object_map_value': 'id',
                'object_termtype': 'http://w3id.org/rml/Literal',
                'lang_datatype': None,
                'lang_datatype_map_type': None,
                'lang_datatype_map_value': None,
                'graph_map_type': None,
                'graph_map_value': None,
                'subject_join_conditions': None,
                'object_join_conditions': None,
            },
            {
                'triples_map_id': '#TM1',
                'source_name': 'DB1',
                'source_type': 'RDB',
                'logical_source_type': 'http://w3id.org/rml/source',
                'logical_source_value': 'DB1',
                'mapping_partition': '9-9-9-9',
                'triples_map_type': 'http://w3id.org/rml/TriplesMap',
                'subject_map_type': 'http://w3id.org/rml/template',
                'subject_map_value': 'http://example.com/{id}',
                'subject_termtype': 'http://w3id.org/rml/IRI',
                'predicate_map_type': 'http://w3id.org/rml/constant',
                'predicate_map_value': 'http://example.com/p',
                'object_map_type': 'http://w3id.org/rml/reference',
                'object_map_value': 'id',
                'object_termtype': 'http://w3id.org/rml/Literal',
                'lang_datatype': None,
                'lang_datatype_map_type': None,
                'lang_datatype_map_value': None,
                'graph_map_type': None,
                'graph_map_value': None,
                'subject_join_conditions': None,
                'object_join_conditions': None,
            },
        ]
    )

    normalized = SparkExecutionBackend._drop_redundant_rdb_source_rows(rml_df)

    assert normalized['triples_map_id'].tolist() == ['#TM0']


def test_spark_pandas_staging_converts_time_values_to_string_rows():
    pyspark_module = types.ModuleType('pyspark')
    pyspark_sql_module = types.ModuleType('pyspark.sql')
    pyspark_functions_module = types.ModuleType('pyspark.sql.functions')
    pyspark_types_module = types.ModuleType('pyspark.sql.types')
    pyspark_functions_module.udf = lambda fn, return_type=None: fn
    pyspark_functions_module.col = lambda name: name
    pyspark_types_module.StringType = object
    pyspark_types_module.StructField = object
    pyspark_types_module.StructType = object
    sys.modules.setdefault('pyspark', pyspark_module)
    sys.modules.setdefault('pyspark.sql', pyspark_sql_module)
    sys.modules.setdefault('pyspark.sql.functions', pyspark_functions_module)
    sys.modules.setdefault('pyspark.sql.types', pyspark_types_module)

    from morph_kgc.engine.spark_materializer import _pandas_df_to_string_rows

    rows = _pandas_df_to_string_rows(
        pd.DataFrame(
            {
                'id': [1, 2],
                'started_at': [time(1, 2, 3, 456), None],
            }
        )
    )

    assert rows == [('1', '01:02:03.000456'), ('2', None)]


def test_spark_materializer_merges_part_files_into_requested_output(tmp_path):
    pyspark_module = types.ModuleType('pyspark')
    pyspark_sql_module = types.ModuleType('pyspark.sql')
    pyspark_functions_module = types.ModuleType('pyspark.sql.functions')
    pyspark_types_module = types.ModuleType('pyspark.sql.types')
    pyspark_functions_module.udf = lambda fn, return_type=None: fn
    pyspark_functions_module.col = lambda name: name
    pyspark_types_module.StringType = object
    pyspark_types_module.StructField = object
    pyspark_types_module.StructType = object
    sys.modules.setdefault('pyspark', pyspark_module)
    sys.modules.setdefault('pyspark.sql', pyspark_sql_module)
    sys.modules.setdefault('pyspark.sql.functions', pyspark_functions_module)
    sys.modules.setdefault('pyspark.sql.types', pyspark_types_module)

    from morph_kgc.engine.spark_materializer import SparkMaterializer

    materializer = object.__new__(SparkMaterializer)
    output_path = tmp_path / 'kg.nt'
    part_dir = tmp_path / 'parts'
    part_dir.mkdir()
    (part_dir / 'part-00000').write_text('a .\n', encoding='utf-8')
    (part_dir / 'part-00001').write_text('b .\n', encoding='utf-8')
    (part_dir / '_SUCCESS').write_text('', encoding='utf-8')

    class FakeWriter:
        def mode(self, value):
            assert value == 'overwrite'
            return self

        def text(self, path):
            copied_dir = Path(path)
            copied_dir.mkdir(parents=True, exist_ok=True)
            for child in part_dir.iterdir():
                target = copied_dir / child.name
                if child.is_file():
                    target.write_bytes(child.read_bytes())

    class FakeTriplesDF:
        write = FakeWriter()

    materializer._write_text_output(FakeTriplesDF(), output_path)

    assert output_path.read_text(encoding='utf-8') == 'a .\nb .\n'
