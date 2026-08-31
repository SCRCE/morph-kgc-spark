import sys
from pathlib import Path

from morph_kgc.engine.spark_support import is_spark_runtime_available


def test_spark_runtime_probe_is_false_in_missing_runtime_environment(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyspark', None)

    assert is_spark_runtime_available() is False


def test_spark_engine_source_does_not_collect_full_output_to_driver():
    sources = [
        Path('src/morph_kgc/engine/spark.py').read_text(encoding='utf-8'),
        Path('src/morph_kgc/engine/spark_materializer.py').read_text(encoding='utf-8'),
        Path('src/morph_kgc/fnml/function_executor.py').read_text(encoding='utf-8'),
    ]
    combined_source = '\n'.join(sources)

    assert '.collect(' not in combined_source
    assert '.toPandas(' not in combined_source


def test_spark_engine_source_does_not_use_local_multiprocessing():
    sources = [
        Path('src/morph_kgc/engine/spark.py').read_text(encoding='utf-8'),
        Path('src/morph_kgc/engine/spark_materializer.py').read_text(encoding='utf-8'),
        Path('src/morph_kgc/fnml/function_executor.py').read_text(encoding='utf-8'),
    ]
    combined_source = '\n'.join(sources)

    assert 'import multiprocessing' not in combined_source
    assert 'from multiprocessing' not in combined_source
    assert 'mp.Pool(' not in combined_source
