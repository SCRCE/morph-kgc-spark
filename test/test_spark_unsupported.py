from pathlib import Path
import textwrap

import pytest

import morph_kgc
from morph_kgc.args_parser import load_config_from_argument
from morph_kgc.engine import get_backend
from morph_kgc.engine.spark_support import is_spark_runtime_available
from morph_kgc.errors import SparkUnsupportedFeature


SPARK_AVAILABLE = is_spark_runtime_available()


def _materialize_with_spark(mapping_path, extra_source_lines='', python_source=None):
    config = load_config_from_argument(
        textwrap.dedent(
            f'''\
            [CONFIGURATION]
            output_file=/tmp/morph-kgc-spark-unsupported.nq
            output_format=N-QUADS
            execution_engine=spark
            number_of_processes=1

            [DataSource]
            mappings={mapping_path}
            {extra_source_lines}'''
        )
    )
    get_backend(config).materialize_to_files(python_source=python_source)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path', 'extra_source_lines', 'expected_message'),
    [
        (
            'excel-source',
            'test/rml-core/tabular/RMLTC0002a_EXCEL/mapping.ttl',
            '',
            'source type `XLSX` is not implemented in the Spark backend yet.',
        ),
        (
            'feather-source',
            'test/rml-core/tabular/RMLTC0002a_FEATHER/mapping.ttl',
            '',
            'source type `FEATHER` is not implemented in the Spark backend yet.',
        ),
        (
            'ods-source',
            'test/rml-core/tabular/RMLTC0002a_ODS/mapping.ttl',
            '',
            'source type `ODS` is not implemented in the Spark backend yet.',
        ),
        (
            'stata-source',
            'test/rml-core/tabular/RMLTC0002a_STATA/mapping.ttl',
            '',
            'source type `DTA` is not implemented in the Spark backend yet.',
        ),
        (
            'geoparquet-source',
            'test/geoparquet/RMLTC0001a/mapping.ttl',
            '',
            'GeoParquet sources are not implemented in the Spark backend yet.',
        ),
    ],
)
def test_spark_fails_explicitly_for_unsupported_fixtures(
    case_name,
    mapping_rel_path,
    extra_source_lines,
    expected_message,
):
    mapping_path = Path(mapping_rel_path).resolve().as_posix()

    with pytest.raises(SparkUnsupportedFeature) as error:
        _materialize_with_spark(mapping_path, extra_source_lines=extra_source_lines)

    message = str(error.value)
    assert expected_message in message
    assert 'Use `execution_engine=pandas` for this workload.' in message


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_fails_explicitly_for_in_memory_sources_without_python_source():
    mapping_path = Path('test/rml-in-memory/pandas_dataframe/RMLIMTC0000/mapping.ttl').resolve().as_posix()

    with pytest.raises(SparkUnsupportedFeature) as error:
        _materialize_with_spark(mapping_path)

    message = str(error.value)
    assert 'in-memory Python sources require providing `python_source` directly to the Spark file-output backend.' in message
    assert 'Use `execution_engine=pandas` for this workload.' in message


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_fails_explicitly_for_property_graph_sources(tmp_path):
    mapping_path = tmp_path / 'property-graph.ttl'
    mapping_path.write_text(
        textwrap.dedent(
            '''\
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix rml: <http://w3id.org/rml/> .

            <TriplesMap1> a rml:TriplesMap;
              rml:logicalSource [
                rml:referenceFormulation rml:Cypher;
                rml:query "MATCH (n) RETURN n.name AS Name"
              ];
              rml:subjectMap [ rml:template "http://example.com/{Name}" ];
              rml:predicateObjectMap [
                rml:predicate foaf:name ;
                rml:objectMap [ rml:reference "Name" ]
              ].
            '''
        ),
        encoding='utf-8',
    )

    with pytest.raises(SparkUnsupportedFeature) as error:
        _materialize_with_spark(
            mapping_path.as_posix(),
            extra_source_lines='db_url=neo4j://example.invalid\n',
        )

    message = str(error.value)
    assert 'property graph sources are not implemented in the Spark backend yet.' in message
    assert 'Use `execution_engine=pandas` for this workload.' in message


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_fails_explicitly_for_parent_triples_without_join_across_different_sources(tmp_path):
    students_csv = tmp_path / 'students.csv'
    students_csv.write_text(
        'id,name,sport_id\n'
        '1,Alice,10\n',
        encoding='utf-8',
    )
    sports_csv = tmp_path / 'sports.csv'
    sports_csv.write_text(
        'id,label\n'
        '10,Tennis\n',
        encoding='utf-8',
    )
    mapping_path = tmp_path / 'cross-source-no-join.ttl'
    mapping_path.write_text(
        textwrap.dedent(
            f'''\
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix ex: <http://example.com/> .
            @prefix rml: <http://w3id.org/rml/> .

            <ParentTM>
              a rml:TriplesMap;
              rml:logicalSource [
                rml:source "{sports_csv.as_posix()}";
                rml:referenceFormulation rml:CSV
              ];
              rml:subjectMap [ rml:template "http://example.com/sport/{{id}}" ];
              rml:predicateObjectMap [
                rml:predicate foaf:name ;
                rml:objectMap [ rml:reference "label" ]
              ].

            <ChildTM>
              a rml:TriplesMap;
              rml:logicalSource [
                rml:source "{students_csv.as_posix()}";
                rml:referenceFormulation rml:CSV
              ];
              rml:subjectMap [ rml:template "http://example.com/student/{{id}}" ];
              rml:predicateObjectMap [
                rml:predicate ex:sport ;
                rml:objectMap [
                  a rml:RefObjectMap ;
                  rml:parentTriplesMap <ParentTM>
                ]
              ].
            '''
        ),
        encoding='utf-8',
    )

    with pytest.raises(SparkUnsupportedFeature) as error:
        _materialize_with_spark(mapping_path.as_posix())

    message = str(error.value)
    assert 'parent triples maps without explicit join conditions are only supported when child and parent use the same logical source in the Spark backend right now.' in message
    assert 'Use `execution_engine=pandas` for this workload.' in message


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('entrypoint_name', 'callable_entrypoint'),
    [
        ('materialize_set', morph_kgc.materialize_set),
        ('materialize', morph_kgc.materialize),
        ('materialize_oxigraph', morph_kgc.materialize_oxigraph),
        ('materialize_kafka', morph_kgc.materialize_kafka),
    ],
)
def test_spark_library_entrypoints_fail_explicitly_without_fallback(entrypoint_name, callable_entrypoint):
    config = textwrap.dedent(
        '''\
        [CONFIGURATION]
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings=test/rml-core/csv/RMLTC0001a/mapping.ttl
        '''
    )

    with pytest.raises(SparkUnsupportedFeature) as error:
        callable_entrypoint(config)

    message = str(error.value)
    assert entrypoint_name in message
    assert 'Use `execution_engine=pandas` for this workload.' in message
