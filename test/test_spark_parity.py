import json
from datetime import datetime
from pathlib import Path
import csv
import re
import textwrap

import pandas as pd
import pytest

from morph_kgc.args_parser import load_config_from_argument
from morph_kgc.engine import get_backend
from morph_kgc.engine.spark_support import is_spark_runtime_available


SPARK_AVAILABLE = is_spark_runtime_available()
RDF_LINE_PATTERN = re.compile(r'^(<[^>]+>) (<[^>]+>) (.+?)\s+\.$')
UUID_LITERAL_PATTERN = re.compile(r'^"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"$')


def canonicalize_rdf_output_path(output_path):
    path = Path(output_path)
    if path.is_dir():
        lines = []
        for part_file in sorted(path.glob('part-*')):
            lines.extend(part_file.read_text(encoding='utf-8').splitlines())
    else:
        lines = path.read_text(encoding='utf-8').splitlines()

    normalized_lines = [line.replace('\r\n', '\n').replace('\r', '\n') for line in lines if line]
    return sorted(normalized_lines)


def _write_output_via_backend(config_text, python_source=None):
    config = load_config_from_argument(config_text)
    backend = get_backend(config)
    backend.materialize_to_files(python_source=python_source)


def _read_rdf_lines(output_path):
    return canonicalize_rdf_output_path(output_path)


def _index_objects_by_subject_predicate(output_path):
    indexed_objects = {}
    for line in _read_rdf_lines(output_path):
        match = RDF_LINE_PATTERN.match(line)
        assert match is not None, f'Unexpected RDF line format: {line}'
        indexed_objects[(match.group(1), match.group(2))] = match.group(3)
    return indexed_objects


def _parse_literal_string(literal):
    assert literal.startswith('"') and literal.endswith('"')
    return literal[1:-1]


def _predicate_local_name(predicate):
    return predicate.rsplit('#', 1)[-1].rstrip('>')


def _assert_datetime_literals_close(pandas_literal, spark_literal, max_delta_seconds=180):
    pandas_value = datetime.fromisoformat(_parse_literal_string(pandas_literal))
    spark_value = datetime.fromisoformat(_parse_literal_string(spark_literal))
    assert abs((spark_value - pandas_value).total_seconds()) <= max_delta_seconds


def _assert_integer_literals_close(pandas_literal, spark_literal, max_delta=180):
    pandas_value = int(_parse_literal_string(pandas_literal))
    spark_value = int(_parse_literal_string(spark_literal))
    assert abs(spark_value - pandas_value) <= max_delta


def _subject_local_name(subject):
    iri = subject.strip('<>')
    if '#' in iri:
        return iri.rsplit('#', 1)[-1]
    return iri.rsplit('/', 1)[-1]


def _assert_spark_matches_pandas(
    tmp_path,
    pandas_output_name,
    spark_output_name,
    mapping_path,
    source_path=None,
    python_source=None,
    datasource_extra='',
):
    pandas_output = tmp_path / pandas_output_name
    spark_output = tmp_path / spark_output_name

    source_line = f'file_path={source_path}\n' if source_path else ''
    datasource_extra = datasource_extra.rstrip()
    if datasource_extra:
        datasource_extra = datasource_extra + '\n'
    pandas_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        {source_line}{datasource_extra}'''
    )
    spark_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        {source_line}{datasource_extra}'''
    )

    _write_output_via_backend(pandas_config, python_source=python_source)
    _write_output_via_backend(spark_config, python_source=python_source)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


def _sqlite_db_url(fixture_dir):
    return f"db_url=sqlite:///{Path(fixture_dir, 'resource.db').resolve().as_posix()}"


def _assert_spark_semantics_for_nondeterministic_date_fixture(
    tmp_path,
    pandas_output_name,
    spark_output_name,
    mapping_path,
    source_path,
    expected_predicates,
):
    pandas_output = tmp_path / pandas_output_name
    spark_output = tmp_path / spark_output_name

    pandas_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={source_path}
        '''
    )
    spark_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={source_path}
        '''
    )

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    pandas_index = _index_objects_by_subject_predicate(pandas_output)
    spark_index = _index_objects_by_subject_predicate(spark_output)

    assert set(pandas_index) == set(spark_index)
    assert {predicate for _, predicate in pandas_index} == expected_predicates

    for key in pandas_index:
        predicate = _predicate_local_name(key[1])
        if predicate in {'date_now', 'date_datePart'}:
            _assert_datetime_literals_close(pandas_index[key], spark_index[key])
        elif predicate == 'date_diff':
            _assert_integer_literals_close(pandas_index[key], spark_index[key])
        else:
            raise AssertionError(f'Unexpected nondeterministic predicate `{key[1]}`.')


def _assert_spark_semantics_for_nondeterministic_math_random_fixture(tmp_path):
    mapping_path = Path('test/rml-fnml/math_functions/math_random/mapping.yarrrml').resolve().as_posix()
    source_path = Path('test/rml-fnml/math_functions/cars.csv').resolve().as_posix()
    pandas_output = tmp_path / 'pandas-math-random.nq'
    spark_output = tmp_path / 'spark-math-random.nq'

    pandas_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={source_path}
        '''
    )
    spark_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={source_path}
        '''
    )

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    pandas_index = _index_objects_by_subject_predicate(pandas_output)
    spark_index = _index_objects_by_subject_predicate(spark_output)

    assert set(pandas_index) == set(spark_index)
    assert {predicate for _, predicate in pandas_index} == {'<http://example.com#number_check>'}

    bounds_by_subject = {}
    with open(source_path, encoding='utf-8') as csv_handle:
        reader = csv.DictReader(csv_handle)
        for row in reader:
            subject = f'car_{row["ID"]}'
            bounds_by_subject[subject] = (float(row['Seats']), float(row['Year']))

    for (subject, predicate), spark_literal in spark_index.items():
        assert predicate == '<http://example.com#number_check>'
        subject_key = _subject_local_name(subject)
        lower_bound, upper_bound = bounds_by_subject[subject_key]
        spark_value = float(_parse_literal_string(spark_literal))
        pandas_value = float(_parse_literal_string(pandas_index[(subject, predicate)]))
        assert lower_bound <= spark_value <= upper_bound
        assert lower_bound <= pandas_value <= upper_bound


def _assert_spark_semantics_for_nondeterministic_uuid_fixture(tmp_path):
    mapping_path = Path('test/rml-fnml/uuid/mapping.ttl').resolve().as_posix()
    source_path = Path('test/rml-fnml/uuid/student.csv').resolve().as_posix()
    pandas_output = tmp_path / 'pandas-uuid.nq'
    spark_output = tmp_path / 'spark-uuid.nq'

    pandas_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={source_path}
        '''
    )
    spark_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={source_path}
        '''
    )

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    pandas_index = _index_objects_by_subject_predicate(pandas_output)
    spark_index = _index_objects_by_subject_predicate(spark_output)

    assert set(pandas_index) == set(spark_index)
    assert len(pandas_index) == 1
    assert {predicate for _, predicate in pandas_index} == {'<http://xmlns.com/foaf/0.1/name>'}

    pandas_literal = next(iter(pandas_index.values()))
    spark_literal = next(iter(spark_index.values()))
    assert UUID_LITERAL_PATTERN.match(pandas_literal)
    assert UUID_LITERAL_PATTERN.match(spark_literal)


def _write_tabular_text_edge_source(tmp_path, source_type):
    source_path = tmp_path / f'text-edges.{source_type}'
    frame = pd.DataFrame(
        {
            'ID': ['1', '2', '3'],
            'Name': ['Alice "Quoted"', r'Back\\slash', 'Unicode Ω'],
            'Note': ['Line 1\nLine 2', '', 'Tab\tSeparated'],
        }
    )

    if source_type == 'parquet':
        frame.to_parquet(source_path, engine='pyarrow', index=False)
    elif source_type == 'orc':
        pyarrow = pytest.importorskip('pyarrow')
        pyarrow_orc = pytest.importorskip('pyarrow.orc')
        table = pyarrow.Table.from_pandas(frame, preserve_index=False)
        with pyarrow.OSFile(str(source_path), 'wb') as sink:
            pyarrow_orc.write_table(table, sink)
    else:
        raise ValueError(f'Unexpected tabular source type `{source_type}`.')

    return source_path


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_parity_helper_reads_part_files(tmp_path):
    output_dir = tmp_path / 'knowledge-graph.nt'
    output_dir.mkdir()
    (output_dir / 'part-00000').write_text('b .\na .\n', encoding='utf-8')
    (output_dir / 'part-00001').write_text('c .\n', encoding='utf-8')

    assert canonicalize_rdf_output_path(output_dir) == ['a .', 'b .', 'c .']


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_issue_67(tmp_path):
    pandas_output = tmp_path / 'pandas-output.nq'
    spark_output = tmp_path / 'spark-output.nq'
    mapping_path = Path('test/issues/issue_67/mapping.ttl').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_parent_triples_join_fixture(tmp_path):
    pandas_output = tmp_path / 'pandas-output.nq'
    spark_output = tmp_path / 'spark-output.nq'
    mapping_path = Path('test/rml-core/csv/RMLTC0009a/mapping.ttl').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('fixture_dir', 'pandas_name', 'spark_name'),
    [
        ('test/r2rml/R2RMLTC0001a', 'pandas-r2rml-0001a.nq', 'spark-r2rml-0001a.nq'),
        ('test/r2rml/R2RMLTC0003b', 'pandas-r2rml-0003b.nq', 'spark-r2rml-0003b.nq'),
        ('test/r2rml/R2RMLTC0009a', 'pandas-r2rml-0009a.nq', 'spark-r2rml-0009a.nq'),
    ],
    ids=[
        'r2rml-table-0001a',
        'r2rml-sqlquery-0003b',
        'r2rml-parent-join-0009a',
    ],
)
def test_spark_matches_pandas_for_supported_r2rml_sqlite_fixtures(tmp_path, fixture_dir, pandas_name, spark_name):
    fixture_path = Path(fixture_dir)
    _assert_spark_matches_pandas(
        tmp_path,
        pandas_name,
        spark_name,
        fixture_path.joinpath('mapping.ttl').resolve().as_posix(),
        datasource_extra=_sqlite_db_url(fixture_path),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_native_fnml_uppercase(tmp_path):
    pandas_output = tmp_path / 'pandas-uppercase.nq'
    spark_output = tmp_path / 'spark-uppercase.nq'
    mapping_path = Path('test/rml-fnml/string_functions/RMLFNOTC0001-CSV_upperCase/mapping.ttl').resolve().as_posix()
    csv_path = Path('test/rml-fnml/string_functions/RMLFNOTC0001-CSV_upperCase/student.csv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_fnml_literal_escaping(tmp_path):
    csv_path = tmp_path / 'quoted-students.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(['ID', 'Name'])
        writer.writerow(['1', 'Alice "Quoted"'])

    mapping_path = tmp_path / 'quoted-uppercase.ttl'
    mapping_path.write_text(
        textwrap.dedent(
            f'''\
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix rml: <http://w3id.org/rml/> .
            @prefix grel: <http://users.ugent.be/~bjdmeest/function/grel.ttl#> .

            <TriplesMap1>
                rml:logicalSource [
                    rml:source "{csv_path.as_posix()}";
                    rml:referenceFormulation rml:CSV
                ];
                rml:subjectMap [
                    rml:template "http://example.com/{{ID}}"
                ];
                rml:predicateObjectMap [
                    rml:predicate foaf:name;
                    rml:objectMap [
                        rml:functionExecution <#Execution> ;
                    ]
                ] .

            <#Execution>
                rml:function grel:toUpperCase ;
                rml:input
                    [
                        rml:parameter grel:valueParameter ;
                        rml:inputValueMap [
                            rml:reference "Name" ;
                        ]
                    ] .
            '''
        ),
        encoding='utf-8',
    )

    _assert_spark_matches_pandas(
        tmp_path,
        'pandas-fnml-literal-escaping.nq',
        'spark-fnml-literal-escaping.nq',
        mapping_path.as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_nested_native_fnml_replace_uppercase(tmp_path):
    pandas_output = tmp_path / 'pandas-replace-uppercase.nq'
    spark_output = tmp_path / 'spark-replace-uppercase.nq'
    mapping_path = Path('test/rml-fnml/string_functions/RMLFNOTC0009-CSV_replace/mapping.ttl').resolve().as_posix()
    csv_path = Path('test/rml-fnml/string_functions/RMLFNOTC0009-CSV_replace/student.csv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('null-filter', 'test/rml-core/csv/null_filter/mapping.ttl'),
        ('language-tags-short', 'test/rml-core/csv/RMLTC0015a/mapping.ttl'),
        ('language-tags-word', 'test/rml-core/csv/RMLTC0015b/mapping.ttl'),
        ('graph-constant', 'test/rml-core/csv/RMLTC0007e/mapping.ttl'),
        ('graph-default', 'test/rml-core/csv/RMLTC0007g/mapping.ttl'),
        ('graph-template', 'test/rml-core/csv/RMLTC0008a/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_csv_core_semantics_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_encodes_spaces_in_iri_templates(tmp_path):
    csv_path = tmp_path / 'phone-numbers.csv'
    csv_path.write_text(
        'ID,Number\n'
        '1,0020 100222801\n',
        encoding='utf-8',
    )

    mapping_path = tmp_path / 'phone-number-mapping.ttl'
    mapping_path.write_text(
        textwrap.dedent(
            f'''\
            @prefix rr: <http://www.w3.org/ns/r2rml#> .
            @prefix rml: <http://w3id.org/rml/> .
            @prefix ql: <http://semweb.mmlab.be/ns/ql#> .
            @prefix cdr: <http://www.ig.io/cdr/> .

            <TriplesMap1>
                rml:logicalSource [
                    rml:source "{csv_path.as_posix()}";
                    rml:referenceFormulation ql:CSV
                ];
                rml:subjectMap [
                    rml:template "http://www.ig.io/cdr/PhoneCall_{{ID}}"
                ];
                rr:predicateObjectMap [
                    rr:predicate cdr:hasOriginalCalledNumber;
                    rr:objectMap [
                        rml:template "http://www.ig.io/cdr/PhoneNumber_{{Number}}";
                        rr:termType rr:IRI
                    ]
                ] .
            '''
        ),
        encoding='utf-8',
    )

    _assert_spark_matches_pandas(
        tmp_path,
        'pandas-phone-number-iri.nq',
        'spark-phone-number-iri.nq',
        mapping_path.as_posix(),
    )

    assert _read_rdf_lines(tmp_path / 'spark-phone-number-iri.nq') == [
        '<http://www.ig.io/cdr/PhoneCall_1> <http://www.ig.io/cdr/hasOriginalCalledNumber> <http://www.ig.io/cdr/PhoneNumber_0020%20100222801>  .'
    ]


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('iri-template-encoding', 'test/rml-core/csv/RMLTC0020a/mapping.ttl'),
        ('iri-reference-encoding', 'test/rml-core/csv/RMLTC0020b/mapping.ttl'),
        ('graph-multi-parent-join', 'test/rml-core/csv/RMLTC0009b/mapping.ttl'),
        ('ref-object-no-explicit-join', 'test/rml-core/csv/RMLTC0008b/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_csv_iri_and_join_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('blank-node-simple', 'test/rml-core/csv/RMLTC0001b/mapping.ttl'),
        ('blank-node-join', 'test/rml-core/csv/RMLTC0012b/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_blank_node_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('csv-empty-output', 'test/rml-core/csv/RMLTC0000/mapping.ttl'),
        ('csv-iri-template-subject', 'test/rml-core/csv/RMLTC0002a/mapping.ttl'),
        ('csv-blanknode-template-subject', 'test/rml-core/csv/RMLTC0002b/mapping.ttl'),
        ('csv-reference-object', 'test/rml-core/csv/RMLTC0003c/mapping.ttl'),
        ('csv-subject-join', 'test/rml-core/csv/RMLTC0004a/mapping.ttl'),
        ('csv-constant-object', 'test/rml-core/csv/RMLTC0005a/mapping.ttl'),
        ('csv-constant-graph', 'test/rml-core/csv/RMLTC0006a/mapping.ttl'),
        ('csv-class-triples', 'test/rml-core/csv/RMLTC0007a/mapping.ttl'),
        ('csv-class-and-predicate', 'test/rml-core/csv/RMLTC0007b/mapping.ttl'),
        ('csv-template-predicate', 'test/rml-core/csv/RMLTC0007c/mapping.ttl'),
        ('csv-reference-predicate', 'test/rml-core/csv/RMLTC0007d/mapping.ttl'),
        ('csv-template-graph', 'test/rml-core/csv/RMLTC0007f/mapping.ttl'),
        ('csv-parent-ref-object-template', 'test/rml-core/csv/RMLTC0008c/mapping.ttl'),
        ('csv-constant-subject-iris', 'test/rml-core/csv/RMLTC0010a/mapping.ttl'),
        ('csv-datatype-integer', 'test/rml-core/csv/RMLTC0010b/mapping.ttl'),
        ('csv-language-map', 'test/rml-core/csv/RMLTC0010c/mapping.ttl'),
        ('csv-multiple-triples-maps', 'test/rml-core/csv/RMLTC0011b/mapping.ttl'),
        ('csv-blanknode-datatype', 'test/rml-core/csv/RMLTC0012a/mapping.ttl'),
        ('csv-blanknode-reordered-template', 'test/rml-core/csv/RMLTC0012d/mapping.ttl'),
        ('csv-language-reference-values', 'test/rml-core/csv/RMLTC0019a/mapping.ttl'),
        ('csv-language-constant-values', 'test/rml-core/csv/RMLTC0019b/mapping.ttl'),
        ('csv-triples-map-without-pom', 'test/rml-core/csv/triples_map_without_pom/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_additional_csv_audit_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('tabular-tsv-core', 'test/rml-core/tabular/RMLTC0002a_TSV/mapping.ttl'),
        ('tabular-parquet-core', 'test/rml-core/tabular/RMLTC0002a_PARQUET/mapping.ttl'),
        ('json-core-0000', 'test/rml-core/json/RMLTC0000/mapping.ttl'),
        ('json-core-0001a', 'test/rml-core/json/RMLTC0001a/mapping.ttl'),
        ('json-blank-node-simple-0001b', 'test/rml-core/json/RMLTC0001b/mapping.ttl'),
        ('json-iri-template-subject-0002a', 'test/rml-core/json/RMLTC0002a/mapping.ttl'),
        ('json-blanknode-template-subject-0002b', 'test/rml-core/json/RMLTC0002b/mapping.ttl'),
        ('json-reference-object-0003c', 'test/rml-core/json/RMLTC0003c/mapping.ttl'),
        ('json-subject-join-0004a', 'test/rml-core/json/RMLTC0004a/mapping.ttl'),
        ('json-constant-object-0005a', 'test/rml-core/json/RMLTC0005a/mapping.ttl'),
        ('json-constant-graph-0006a', 'test/rml-core/json/RMLTC0006a/mapping.ttl'),
        ('json-ref-object-same-source-0008b', 'test/rml-core/json/RMLTC0008b/mapping.ttl'),
        ('json-parent-join-0009a', 'test/rml-core/json/RMLTC0009a/mapping.ttl'),
        ('json-blanknode-datatype-0012a', 'test/rml-core/json/RMLTC0012a/mapping.ttl'),
        ('json-blanknode-reordered-template-0012d', 'test/rml-core/json/RMLTC0012d/mapping.ttl'),
        ('json-language-reference-values-0019a', 'test/rml-core/json/RMLTC0019a/mapping.ttl'),
        ('json-language-constant-values-0019b', 'test/rml-core/json/RMLTC0019b/mapping.ttl'),
        ('json-iri-template-0020a', 'test/rml-core/json/RMLTC0020a/mapping.ttl'),
        ('json-iri-reference-0020b', 'test/rml-core/json/RMLTC0020b/mapping.ttl'),
        ('xml-core-0000', 'test/rml-core/xml/RMLTC0000/mapping.ttl'),
        ('xml-core-0001a', 'test/rml-core/xml/RMLTC0001a/mapping.ttl'),
        ('xml-blank-node-simple-0001b', 'test/rml-core/xml/RMLTC0001b/mapping.ttl'),
        ('xml-iri-template-subject-0002a', 'test/rml-core/xml/RMLTC0002a/mapping.ttl'),
        ('xml-blanknode-template-subject-0002b', 'test/rml-core/xml/RMLTC0002b/mapping.ttl'),
        ('xml-reference-object-0003c', 'test/rml-core/xml/RMLTC0003c/mapping.ttl'),
        ('xml-subject-join-0004a', 'test/rml-core/xml/RMLTC0004a/mapping.ttl'),
        ('xml-constant-object-0005a', 'test/rml-core/xml/RMLTC0005a/mapping.ttl'),
        ('xml-constant-graph-0006a', 'test/rml-core/xml/RMLTC0006a/mapping.ttl'),
        ('xml-ref-object-same-source-0008b', 'test/rml-core/xml/RMLTC0008b/mapping.ttl'),
        ('xml-parent-join-0009a', 'test/rml-core/xml/RMLTC0009a/mapping.ttl'),
        ('xml-blanknode-datatype-0012a', 'test/rml-core/xml/RMLTC0012a/mapping.ttl'),
        ('xml-blanknode-reordered-template-0012d', 'test/rml-core/xml/RMLTC0012d/mapping.ttl'),
        ('xml-language-reference-values-0019a', 'test/rml-core/xml/RMLTC0019a/mapping.ttl'),
        ('xml-language-constant-values-0019b', 'test/rml-core/xml/RMLTC0019b/mapping.ttl'),
        ('xml-iri-template-0020a', 'test/rml-core/xml/RMLTC0020a/mapping.ttl'),
        ('xml-iri-reference-0020b', 'test/rml-core/xml/RMLTC0020b/mapping.ttl'),
        ('csv-ref-object-same-source-0008b', 'test/rml-core/csv/RMLTC0008b/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_core_supported_tabular_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('json-class-triples-0007a', 'test/rml-core/json/RMLTC0007a/mapping.ttl'),
        ('json-class-and-predicate-0007b', 'test/rml-core/json/RMLTC0007b/mapping.ttl'),
        ('json-template-predicate-0007c', 'test/rml-core/json/RMLTC0007c/mapping.ttl'),
        ('json-reference-predicate-0007d', 'test/rml-core/json/RMLTC0007d/mapping.ttl'),
        ('json-constant-graph-0007e', 'test/rml-core/json/RMLTC0007e/mapping.ttl'),
        ('json-template-graph-0007f', 'test/rml-core/json/RMLTC0007f/mapping.ttl'),
        ('json-default-graph-0007g', 'test/rml-core/json/RMLTC0007g/mapping.ttl'),
        ('json-default-graph-no-pom-0008a', 'test/rml-core/json/RMLTC0008a/mapping.ttl'),
        ('json-parent-ref-object-template-0008c', 'test/rml-core/json/RMLTC0008c/mapping.ttl'),
        ('json-parent-ref-object-join-0009b', 'test/rml-core/json/RMLTC0009b/mapping.ttl'),
        ('json-constant-subject-iris-0010a', 'test/rml-core/json/RMLTC0010a/mapping.ttl'),
        ('json-datatype-integer-0010b', 'test/rml-core/json/RMLTC0010b/mapping.ttl'),
        ('json-language-map-0010c', 'test/rml-core/json/RMLTC0010c/mapping.ttl'),
        ('json-multiple-triples-maps-0011b', 'test/rml-core/json/RMLTC0011b/mapping.ttl'),
        ('json-blank-node-simple-0012b', 'test/rml-core/json/RMLTC0012b/mapping.ttl'),
        ('json-language-map-reference-0015a', 'test/rml-core/json/RMLTC0015a/mapping.ttl'),
        ('json-language-map-constant-0015b', 'test/rml-core/json/RMLTC0015b/mapping.ttl'),
        ('xml-class-triples-0007a', 'test/rml-core/xml/RMLTC0007a/mapping.ttl'),
        ('xml-class-and-predicate-0007b', 'test/rml-core/xml/RMLTC0007b/mapping.ttl'),
        ('xml-template-predicate-0007c', 'test/rml-core/xml/RMLTC0007c/mapping.ttl'),
        ('xml-reference-predicate-0007d', 'test/rml-core/xml/RMLTC0007d/mapping.ttl'),
        ('xml-constant-graph-0007e', 'test/rml-core/xml/RMLTC0007e/mapping.ttl'),
        ('xml-template-graph-0007f', 'test/rml-core/xml/RMLTC0007f/mapping.ttl'),
        ('xml-default-graph-0007g', 'test/rml-core/xml/RMLTC0007g/mapping.ttl'),
        ('xml-default-graph-no-pom-0008a', 'test/rml-core/xml/RMLTC0008a/mapping.ttl'),
        ('xml-parent-ref-object-template-0008c', 'test/rml-core/xml/RMLTC0008c/mapping.ttl'),
        ('xml-parent-ref-object-join-0009b', 'test/rml-core/xml/RMLTC0009b/mapping.ttl'),
        ('xml-datatype-integer-0010b', 'test/rml-core/xml/RMLTC0010b/mapping.ttl'),
        ('xml-language-map-0010c', 'test/rml-core/xml/RMLTC0010c/mapping.ttl'),
        ('xml-multiple-triples-maps-0011b', 'test/rml-core/xml/RMLTC0011b/mapping.ttl'),
        ('xml-blank-node-simple-0012b', 'test/rml-core/xml/RMLTC0012b/mapping.ttl'),
        ('xml-language-map-reference-0015a', 'test/rml-core/xml/RMLTC0015a/mapping.ttl'),
        ('xml-language-map-constant-0015b', 'test/rml-core/xml/RMLTC0015b/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_additional_json_xml_audit_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('json-language-datatype-map-0013a', 'test/rml-core/json/RMLTC0013a/mapping.ttl'),
        ('json-complex-wildcard-mapping', 'test/rml-core/json/complex/mapping.ttl'),
        ('xml-attributes-mapping', 'test/rml-core/xml/attributes/mapping.ttl'),
        ('xml-rml-spec-example-section-3', 'test/rml-core/xml/rml_spec_example_section_3/mapping.ttl'),
        ('xml-rml-spec-example-section-5', 'test/rml-core/xml/rml_spec_example_section_5/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_additional_json_xml_parser_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_in_memory_dataframe_source(tmp_path):
    mapping_path = Path('test/rml-in-memory/pandas_dataframe/RMLIMTC0000/mapping.ttl').resolve().as_posix()
    python_source = {'variable1': pd.DataFrame({'Name': ['Alice', 'Bob']})}

    _assert_spark_matches_pandas(
        tmp_path,
        'in-memory-dataframe-pandas.nq',
        'in-memory-dataframe-spark.nq',
        mapping_path,
        python_source=python_source,
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('in-memory-dataframe-parent-join-0009a', 'test/rml-in-memory/pandas_dataframe/RMLIMTC0009a/mapping.ttl'),
        ('in-memory-dataframe-parent-join-0009b', 'test/rml-in-memory/pandas_dataframe/RMLIMTC0009b/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_in_memory_dataframe_parent_join_fixtures(tmp_path, case_name, mapping_rel_path):
    python_source = {
        'variable1': pd.DataFrame(
            {'ID': [10, 20], 'Sport': ['100', ''], 'Name': ['Venus Williams', 'Demi Moore']}
        ),
        'variable2': pd.DataFrame({'ID': [100], 'Name': ['Tennis']}),
    }

    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
        python_source=python_source,
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('in-memory-dictionary-parent-join-0009a', 'test/rml-in-memory/json_dictionary/RMLIMTC0009a/mapping.ttl'),
        ('in-memory-dictionary-parent-join-0009b', 'test/rml-in-memory/json_dictionary/RMLIMTC0009b/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_in_memory_dictionary_parent_join_fixtures(tmp_path, case_name, mapping_rel_path):
    python_source = {
        'variable1': {'sports': [{'ID': 100, 'Name': 'Tennis'}]},
        'variable2': {
            'students': [
                {'ID': 10, 'Sport': 100, 'Name': 'Venus Williams'},
                {'ID': 20, 'Name': 'Demi Moore'},
            ]
        },
    }

    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
        python_source=python_source,
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'json_file_name'),
    [
        ('issue-316-a', 'a.json'),
        ('issue-316-b', 'b.json'),
    ],
)
def test_spark_matches_pandas_for_in_memory_json_udf_issue_fixtures(tmp_path, case_name, json_file_name):
    issue_dir = Path('test/issues/issue_316').resolve()
    mapping_path = (issue_dir / 'mapping.ttl').as_posix()
    udfs_path = (issue_dir / 'udfs.py').as_posix()
    with (issue_dir / json_file_name).open(encoding='utf-8') as json_handle:
        python_source = {'data': json.load(json_handle)}

    pandas_output = tmp_path / f'{case_name}-pandas.nq'
    spark_output = tmp_path / f'{case_name}-spark.nq'
    pandas_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1
        udfs={udfs_path}

        [DataSource]
        mappings={mapping_path}
        '''
    )
    spark_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1
        udfs={udfs_path}

        [DataSource]
        mappings={mapping_path}
        '''
    )

    _write_output_via_backend(pandas_config, python_source=python_source)
    _write_output_via_backend(spark_config, python_source=python_source)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('tabular-view-0002d', 'test/rml-tv/RMLTVTC0002d/mapping.ttl'),
        ('tabular-view-0002i', 'test/rml-tv/RMLTVTC0002i/mapping.ttl'),
        ('tabular-view-0002j', 'test/rml-tv/RMLTVTC0002j/mapping.ttl'),
        ('tabular-view-0003b', 'test/rml-tv/RMLTVTC0003b/mapping.ttl'),
        ('tabular-view-0009c', 'test/rml-tv/RMLTVTC0009c/mapping.ttl'),
        ('tabular-view-0009d', 'test/rml-tv/RMLTVTC0009d/mapping.ttl'),
        ('tabular-view-0011a', 'test/rml-tv/RMLTVTC0011a/mapping.ttl'),
        ('tabular-view-0014d', 'test/rml-tv/RMLTVTC0014d/mapping.ttl'),
        ('tabular-view-0015a', 'test/rml-tv/RMLTVTC0015a/mapping.ttl'),
        ('tabular-view-0019a', 'test/rml-tv/RMLTVTC0019a/mapping.ttl'),
        ('tabular-view-0026a', 'test/rml-tv/RMLTVTC0026a/mapping.ttl'),
        ('tabular-view-0027a', 'test/rml-tv/RMLTVTC0027a/mapping.ttl'),
        ('tabular-view-0028a', 'test/rml-tv/RMLTVTC0028a/mapping.ttl'),
        ('tabular-view-0029a', 'test/rml-tv/RMLTVTC0029a/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_supported_tabular_view_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('quoted-triples-001a', 'test/rml-star/RMLSTARTC001a/mapping.ttl'),
        ('quoted-triples-001b', 'test/rml-star/RMLSTARTC001b/mapping.ttl'),
        ('quoted-triples-002a', 'test/rml-star/RMLSTARTC002a/mapping.ttl'),
        ('quoted-triples-002b', 'test/rml-star/RMLSTARTC002b/mapping.ttl'),
        ('quoted-triples-003a', 'test/rml-star/RMLSTARTC003a/mapping.ttl'),
        ('quoted-triples-003b', 'test/rml-star/RMLSTARTC003b/mapping.ttl'),
        ('quoted-triples-004a', 'test/rml-star/RMLSTARTC004a/mapping.ttl'),
        ('quoted-triples-004b', 'test/rml-star/RMLSTARTC004b/mapping.ttl'),
        ('quoted-triples-005a', 'test/rml-star/RMLSTARTC005a/mapping.ttl'),
        ('quoted-triples-005b', 'test/rml-star/RMLSTARTC005b/mapping.ttl'),
        ('quoted-triples-006a', 'test/rml-star/RMLSTARTC006a/mapping.ttl'),
        ('quoted-triples-006b', 'test/rml-star/RMLSTARTC006b/mapping.ttl'),
        ('quoted-triples-007a', 'test/rml-star/RMLSTARTC007a/mapping.ttl'),
        ('quoted-triples-007b', 'test/rml-star/RMLSTARTC007b/mapping.ttl'),
        ('quoted-triples-008a', 'test/rml-star/RMLSTARTC008a/mapping.ttl'),
        ('quoted-triples-008b', 'test/rml-star/RMLSTARTC008b/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_supported_quoted_triples_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('parent-join-multicolumn-issue-62', 'test/issues/issue_62/mapping.ttl'),
    ],
)
def test_spark_matches_pandas_for_supported_multicolumn_parent_join_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('nested-quoted-triples-issue-124', 'test/issues/issue_124/mapping.ttl'),
        ('nested-quoted-triples-issue-174a', 'test/issues/issue_174/mapping-a.ttl'),
        ('nested-quoted-triples-issue-174b', 'test/issues/issue_174/mapping-b.ttl'),
        ('nested-quoted-triples-issue-174c', 'test/issues/issue_174/mapping-c.ttl'),
    ],
)
def test_spark_matches_pandas_for_supported_nested_quoted_triples_issue_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'{case_name}-pandas.nq',
        f'{case_name}-spark.nq',
        Path(mapping_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_quoted_triples_over_parent_triples_map_fixture(tmp_path):
    students_csv = tmp_path / 'students.csv'
    students_csv.write_text(
        'id,name,sport_id\n'
        '1,Alice,10\n'
        '2,Bob,20\n',
        encoding='utf-8',
    )
    sports_csv = tmp_path / 'sports.csv'
    sports_csv.write_text(
        'id,label\n'
        '10,Tennis\n'
        '20,Chess\n',
        encoding='utf-8',
    )

    mapping_path = tmp_path / 'quoted-parent-triples.ttl'
    mapping_path.write_text(
        textwrap.dedent(
            f'''\
            @prefix ex: <http://example.com/> .
            @prefix rml: <http://w3id.org/rml/> .

            <OuterTM>
              a rml:TriplesMap;
              rml:logicalSource [
                rml:source "{students_csv.as_posix()}";
                rml:referenceFormulation rml:CSV
              ];
              rml:subjectMap [ rml:template "http://example.com/student/{{id}}" ];
              rml:predicateObjectMap [
                rml:predicate ex:statement ;
                rml:objectMap [ rml:quotedTriplesMap <InnerTM> ]
              ].

            <InnerTM>
              a rml:TriplesMap;
              rml:logicalSource [
                rml:source "{students_csv.as_posix()}";
                rml:referenceFormulation rml:CSV
              ];
              rml:subjectMap [ rml:template "http://example.com/student/{{id}}" ];
              rml:predicateObjectMap [
                rml:predicate ex:plays ;
                rml:objectMap [
                  a rml:RefObjectMap ;
                  rml:parentTriplesMap <SportTM> ;
                  rml:joinCondition [
                    rml:child "sport_id" ;
                    rml:parent "id"
                  ]
                ]
              ].

            <SportTM>
              a rml:TriplesMap;
              rml:logicalSource [
                rml:source "{sports_csv.as_posix()}";
                rml:referenceFormulation rml:CSV
              ];
              rml:subjectMap [ rml:template "http://example.com/sport/{{id}}" ];
              rml:predicateObjectMap [
                rml:predicate ex:label ;
                rml:objectMap [ rml:reference "label" ]
              ].
            '''
        ),
        encoding='utf-8',
    )

    _assert_spark_matches_pandas(
        tmp_path,
        'quoted-parent-triples-pandas.nq',
        'quoted-parent-triples-spark.nq',
        mapping_path.as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_subject_quoted_triples_join_fixture(tmp_path):
    parent_csv = tmp_path / 'parents.csv'
    parent_csv.write_text(
        'id,name\n'
        '1,Alice\n'
        '2,Bob\n',
        encoding='utf-8',
    )
    child_csv = tmp_path / 'children.csv'
    child_csv.write_text(
        'id,label\n'
        '1,first\n'
        '2,second\n',
        encoding='utf-8',
    )

    mapping_path = tmp_path / 'subject-quoted-join.ttl'
    mapping_path.write_text(
        textwrap.dedent(
            f'''\
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix ex: <http://example.com/> .
            @prefix rml: <http://w3id.org/rml/> .

            <ParentTM>
              a rml:TriplesMap;
              rml:logicalSource [
                rml:source "{parent_csv.as_posix()}";
                rml:referenceFormulation rml:CSV
              ];
              rml:subjectMap [ rml:template "http://example.com/person/{{id}}" ];
              rml:predicateObjectMap [
                rml:predicate foaf:name;
                rml:objectMap [ rml:reference "name" ]
              ].

            <ChildTM>
              a rml:TriplesMap;
              rml:logicalSource [
                rml:source "{child_csv.as_posix()}";
                rml:referenceFormulation rml:CSV
              ];
              rml:subjectMap [
                rml:quotedTriplesMap <ParentTM>;
                rml:joinCondition [
                  rml:child "id";
                  rml:parent "id"
                ]
              ];
              rml:predicateObjectMap [
                rml:predicate ex:label;
                rml:objectMap [ rml:reference "label" ]
              ].
            '''
        ),
        encoding='utf-8',
    )

    _assert_spark_matches_pandas(
        tmp_path,
        'subject-quoted-join-pandas.nq',
        'subject-quoted-join-spark.nq',
        mapping_path.as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_duplicate_row_fixture(tmp_path):
    csv_path = tmp_path / 'duplicate-rows.csv'
    csv_path.write_text(
        'ID,Name\n'
        '1,Alice\n'
        '1,Alice\n'
        '2,Bob\n',
        encoding='utf-8',
    )

    mapping_path = tmp_path / 'duplicate-rows.ttl'
    mapping_path.write_text(
        textwrap.dedent(
            f'''\
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix rml: <http://w3id.org/rml/> .

            <TriplesMap1>
              a rml:TriplesMap;
              rml:logicalSource [
                rml:source "{csv_path.as_posix()}";
                rml:referenceFormulation rml:CSV
              ];
              rml:subjectMap [ rml:template "http://example.com/person/{{ID}}" ];
              rml:predicateObjectMap [
                rml:predicate foaf:name;
                rml:objectMap [ rml:reference "Name" ]
              ].
            '''
        ),
        encoding='utf-8',
    )

    _assert_spark_matches_pandas(
        tmp_path,
        'duplicate-rows-pandas.nq',
        'duplicate-rows-spark.nq',
        mapping_path.as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_ntriples_output_fixture(tmp_path):
    mapping_path = Path('test/rml-core/csv/RMLTC0001a/mapping.ttl').resolve().as_posix()
    pandas_output = tmp_path / 'pandas-output.nt'
    spark_output = tmp_path / 'spark-output.nt'

    pandas_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-TRIPLES
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        '''
    )
    spark_config = textwrap.dedent(
        f'''\
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-TRIPLES
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        '''
    )

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_text_edge_case_fixture(tmp_path):
    csv_path = tmp_path / 'text-edges.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(['ID', 'Name', 'Note'])
        writer.writerow(['1', 'Alice "Quoted"', 'Line 1\nLine 2'])
        writer.writerow(['2', r'Back\\slash', ''])
        writer.writerow(['3', 'Unicode Ω', 'Tab\tSeparated'])

    mapping_path = tmp_path / 'text-edges.ttl'
    mapping_path.write_text(
        textwrap.dedent(
            f'''\
            @prefix ex: <http://example.com/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix rml: <http://w3id.org/rml/> .

            <TriplesMap1>
              a rml:TriplesMap;
              rml:logicalSource [
                rml:source "{csv_path.as_posix()}";
                rml:referenceFormulation rml:CSV
              ];
              rml:subjectMap [ rml:template "http://example.com/person/{{ID}}" ];
              rml:predicateObjectMap [
                rml:predicate foaf:name;
                rml:objectMap [ rml:reference "Name" ]
              ];
              rml:predicateObjectMap [
                rml:predicate ex:note;
                rml:objectMap [ rml:reference "Note" ]
              ].
            '''
        ),
        encoding='utf-8',
    )

    _assert_spark_matches_pandas(
        tmp_path,
        'text-edges-pandas.nq',
        'text-edges-spark.nq',
        mapping_path.as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize('source_type', ['parquet', 'orc'])
def test_spark_matches_pandas_for_columnar_text_edge_sources(tmp_path, source_type):
    source_path = _write_tabular_text_edge_source(tmp_path, source_type)

    mapping_path = tmp_path / f'text-edges-{source_type}.ttl'
    mapping_path.write_text(
        textwrap.dedent(
            f'''\
            @prefix ex: <http://example.com/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix rml: <http://w3id.org/rml/> .

            <TriplesMap1>
              a rml:TriplesMap;
              rml:logicalSource [
                rml:source "{source_path.as_posix()}";
                rml:referenceFormulation rml:CSV
              ];
              rml:subjectMap [ rml:template "http://example.com/person/{{ID}}" ];
              rml:predicateObjectMap [
                rml:predicate foaf:name;
                rml:objectMap [ rml:reference "Name" ]
              ];
              rml:predicateObjectMap [
                rml:predicate ex:note;
                rml:objectMap [ rml:reference "Note" ]
              ].
            '''
        ),
        encoding='utf-8',
    )

    _assert_spark_matches_pandas(
        tmp_path,
        f'{source_type}-text-edges-pandas.nq',
        f'{source_type}-text-edges-spark.nq',
        mapping_path.as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('string-length', 'test/rml-fnml/string_functions/string_length/mapping.yarrrml'),
        ('string-indexof', 'test/rml-fnml/string_functions/string_indexof/mapping.yarrrml'),
        ('string-contains', 'test/rml-fnml/string_functions/string_contains/mapping.yarrrml'),
        ('string-substring', 'test/rml-fnml/string_functions/string_substring/mapping.yarrrml'),
        ('string-diff', 'test/rml-fnml/string_functions/string_diff/mapping.yarrrml'),
        ('string-starts-endswith', 'test/rml-fnml/string_functions/string_starts_endswith/mapping.yarrrml'),
        ('string-hash', 'test/rml-fnml/string_functions/string_hash/mapping.yarrrml'),
        ('string-chomp-trim', 'test/rml-fnml/string_functions/string_chomp_trim/mapping.yarrrml'),
    ],
)
def test_spark_matches_pandas_for_supported_string_function_fixtures(tmp_path, case_name, mapping_rel_path):
    mapping_path = Path(mapping_rel_path).resolve().as_posix()
    csv_path = Path('test/rml-fnml/string_functions/cars.csv').resolve().as_posix()

    _assert_spark_matches_pandas(
        tmp_path,
        f'pandas-{case_name}.nq',
        f'spark-{case_name}.nq',
        mapping_path,
        csv_path,
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path', 'source_rel_path'),
    [
        ('string-split', 'test/rml-fnml/string_functions/string_splits/split/mapping.yarrrml', 'test/rml-fnml/string_functions/string_splits/split/cars.csv'),
        ('string-split-by-char-type', 'test/rml-fnml/string_functions/string_splits/by_char_type/mapping.yarrrml', 'test/rml-fnml/string_functions/cars.csv'),
        ('string-casing', 'test/rml-fnml/string_functions/string_casing/mapping.yarrrml', 'test/rml-fnml/string_functions/cars.csv'),
    ],
)
def test_spark_matches_pandas_for_additional_string_array_fixtures(tmp_path, case_name, mapping_rel_path, source_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'pandas-{case_name}.nq',
        f'spark-{case_name}.nq',
        Path(mapping_rel_path).resolve().as_posix(),
        Path(source_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_string_match_fixture(tmp_path):
    mapping_path = Path('test/rml-fnml/string_functions/string_match/mapping.yarrrml').resolve().as_posix()
    csv_path = Path('test/rml-fnml/string_functions/cars.csv').resolve().as_posix()

    _assert_spark_matches_pandas(
        tmp_path,
        'pandas-string-match.nq',
        'spark-string-match.nq',
        mapping_path,
        csv_path,
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_split_explode_fixture(tmp_path):
    pandas_output = tmp_path / 'pandas-split-explode.nq'
    spark_output = tmp_path / 'spark-split-explode.nq'
    mapping_path = Path('test/rml-fnml/string_functions/split_explode/mapping.ttl').resolve().as_posix()
    csv_path = Path('test/rml-fnml/string_functions/split_explode/mixed_content_list.csv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_split_explode_null_fixture(tmp_path):
    pandas_output = tmp_path / 'pandas-split-explode-null.nq'
    spark_output = tmp_path / 'spark-split-explode-null.nq'
    mapping_path = Path('test/rml-fnml/string_functions/split_explode_null/mapping.ttl').resolve().as_posix()
    csv_path = Path('test/rml-fnml/string_functions/split_explode_null/mixed_content_list.csv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_array_get_slice_fixture(tmp_path):
    pandas_output = tmp_path / 'pandas-array-get-slice.nq'
    spark_output = tmp_path / 'spark-array-get-slice.nq'
    mapping_path = Path('test/rml-fnml/array_functions/array_get_slice/mapping.ttl').resolve().as_posix()
    tsv_path = Path('test/rml-fnml/array_functions/array_get_slice/article.tsv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={tsv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={tsv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_array_length_fixture(tmp_path):
    pandas_output = tmp_path / 'pandas-array-length.nq'
    spark_output = tmp_path / 'spark-array-length.nq'
    mapping_path = Path('test/rml-fnml/array_functions/array_length/mapping.yarrrml').resolve().as_posix()
    csv_path = Path('test/rml-fnml/array_functions/cars.csv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_array_sum_fixture(tmp_path):
    pandas_output = tmp_path / 'pandas-array-sum.nq'
    spark_output = tmp_path / 'spark-array-sum.nq'
    mapping_path = Path('test/rml-fnml/array_functions/array_sum/mapping.yarrrml').resolve().as_posix()
    csv_path = Path('test/rml-fnml/array_functions/cars.csv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path'),
    [
        ('array-unique', 'test/rml-fnml/array_functions/array_unique/mapping.yarrrml'),
    ],
)
def test_spark_matches_pandas_for_additional_array_fixtures(tmp_path, case_name, mapping_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'pandas-{case_name}.nq',
        f'spark-{case_name}.nq',
        Path(mapping_rel_path).resolve().as_posix(),
        Path('test/rml-fnml/array_functions/cars.csv').resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path', 'source_rel_path'),
    [
        ('controls-and', 'test/rml-fnml/controls_functions/controls_and/mapping.yarrrml', 'test/rml-fnml/controls_functions/cars.csv'),
        ('controls-not-or', 'test/rml-fnml/controls_functions/controls_not_or/mapping.yarrrml', 'test/rml-fnml/controls_functions/cars.csv'),
        ('controls-true-yarrrml', 'test/rml-fnml/controls_functions/controls_true/mapping.yarrrml', 'test/rml-fnml/controls_functions/cars.csv'),
        ('controls-xor-if', 'test/rml-fnml/controls_functions/controls_xor_if/mapping.yarrrml', 'test/rml-fnml/controls_functions/cars.csv'),
    ],
)
def test_spark_matches_pandas_for_controls_function_fixtures(tmp_path, case_name, mapping_rel_path, source_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'pandas-{case_name}.nq',
        f'spark-{case_name}.nq',
        Path(mapping_rel_path).resolve().as_posix(),
        Path(source_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path', 'source_rel_path'),
    [
        ('controls-if', 'test/rml-fnml/controls_functions/controls_if/mapping.ttl', 'test/rml-fnml/controls_functions/controls_if/calendar.csv'),
        ('controls-if-cast', 'test/rml-fnml/controls_functions/controls_if_cast/mapping.ttl', 'test/rml-fnml/controls_functions/controls_if_cast/calendar.csv'),
    ],
)
def test_spark_matches_pandas_for_controls_if_fixtures(tmp_path, case_name, mapping_rel_path, source_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'pandas-{case_name}.nq',
        f'spark-{case_name}.nq',
        Path(mapping_rel_path).resolve().as_posix(),
        Path(source_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path', 'source_rel_path'),
    [
        ('date-datepart', 'test/rml-fnml/date_functions/date_datepart/mapping.yarrrml', 'test/rml-fnml/date_functions/cars.csv'),
        ('math-abs', 'test/rml-fnml/math_functions/math_abs/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-acos', 'test/rml-fnml/math_functions/math_acos/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-asin', 'test/rml-fnml/math_functions/math_asin/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-atan', 'test/rml-fnml/math_functions/math_atan/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-atan2', 'test/rml-fnml/math_functions/math_atan2/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-cos', 'test/rml-fnml/math_functions/math_cos/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-cosh', 'test/rml-fnml/math_functions/math_cosh/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-degrees', 'test/rml-fnml/math_functions/math_degrees/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-even-odd', 'test/rml-fnml/math_functions/math_even_odd/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-exp', 'test/rml-fnml/math_functions/math_exp/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-fact', 'test/rml-fnml/math_functions/math_fact/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-factn', 'test/rml-fnml/math_functions/math_factn/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-floor', 'test/rml-fnml/math_functions/math_floor/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-gcd', 'test/rml-fnml/math_functions/math_gcd/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-lcm', 'test/rml-fnml/math_functions/math_lcm/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-ln', 'test/rml-fnml/math_functions/math_ln/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-log', 'test/rml-fnml/math_functions/math_log/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-min-max', 'test/rml-fnml/math_functions/math_min_max/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-ceil', 'test/rml-fnml/math_functions/math_ceil/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-combin', 'test/rml-fnml/math_functions/math_combin/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-mod', 'test/rml-fnml/math_functions/math_mod/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-multinominal', 'test/rml-fnml/math_functions/math_multinominal/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-pow', 'test/rml-fnml/math_functions/math_pow/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-quotient', 'test/rml-fnml/math_functions/math_quotient/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-radians', 'test/rml-fnml/math_functions/math_radians/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-sin', 'test/rml-fnml/math_functions/math_sin/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-sinh', 'test/rml-fnml/math_functions/math_sinh/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-tan', 'test/rml-fnml/math_functions/math_tan/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
        ('math-tanh', 'test/rml-fnml/math_functions/math_tanh/mapping.yarrrml', 'test/rml-fnml/math_functions/cars.csv'),
    ],
)
def test_spark_matches_pandas_for_additional_scalar_builtin_fixtures(tmp_path, case_name, mapping_rel_path, source_rel_path):
    _assert_spark_matches_pandas(
        tmp_path,
        f'pandas-{case_name}.nq',
        f'spark-{case_name}.nq',
        Path(mapping_rel_path).resolve().as_posix(),
        Path(source_rel_path).resolve().as_posix(),
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
@pytest.mark.parametrize(
    ('case_name', 'mapping_rel_path', 'source_rel_path'),
    [
        ('abbreviated-syntax', 'test/rml-fnml/abbreviated_syntax/mapping.yarrrml', 'test/rml-fnml/abbreviated_syntax/cars.csv'),
        ('fnml-to-rml-yarrrml', 'test/rml-fnml/fnml_to_rml/mapping.yarrrml', 'test/rml-fnml/fnml_to_rml/cars.csv'),
        ('fnml-to-rml-ttl', 'test/rml-fnml/fnml_to_rml/fnml.ttl', 'test/rml-fnml/fnml_to_rml/cars.csv'),
        ('other-type', 'test/rml-fnml/other_functions/type/mapping.yarrrml', None),
        ('escape', 'test/rml-fnml/other_functions/RMLFNOTC0003-CSV_escape/mapping.ttl', 'test/rml-fnml/other_functions/RMLFNOTC0003-CSV_escape/student.csv'),
        ('touppercase', 'test/rml-fnml/string_functions/RMLFNOTC0002-CSV_touppercase/mapping.ttl', 'test/rml-fnml/string_functions/RMLFNOTC0002-CSV_touppercase/student.csv'),
        ('uppercase-0008', 'test/rml-fnml/string_functions/RMLFNOTC0008-CSV_uppercase/mapping.ttl', 'test/rml-fnml/string_functions/RMLFNOTC0008-CSV_uppercase/student.csv'),
    ],
)
def test_spark_matches_pandas_for_additional_misc_fixtures(tmp_path, case_name, mapping_rel_path, source_rel_path):
    source_path = None if source_rel_path is None else Path(source_rel_path).resolve().as_posix()
    _assert_spark_matches_pandas(
        tmp_path,
        f'pandas-{case_name}.nq',
        f'spark-{case_name}.nq',
        Path(mapping_rel_path).resolve().as_posix(),
        source_path,
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_math_round_fixture(tmp_path):
    pandas_output = tmp_path / 'pandas-math-round.nq'
    spark_output = tmp_path / 'spark-math-round.nq'
    mapping_path = Path('test/rml-fnml/math_functions/math_round/mapping.ttl').resolve().as_posix()
    tsv_path = Path('test/rml-fnml/math_functions/math_round/distances.tsv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={tsv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={tsv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_date_to_date_fixture(tmp_path):
    pandas_output = tmp_path / 'pandas-date-to-date.nq'
    spark_output = tmp_path / 'spark-date-to-date.nq'
    mapping_path = Path('test/rml-fnml/date_functions/date_to_date/mapping.ttl').resolve().as_posix()
    csv_path = Path('test/rml-fnml/date_functions/date_to_date/calendar.csv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_semantics_for_nondeterministic_date_diff_fixture(tmp_path):
    mapping_path = Path('test/rml-fnml/date_functions/date_diff/mapping.yarrrml').resolve().as_posix()
    csv_path = Path('test/rml-fnml/date_functions/cars.csv').resolve().as_posix()

    _assert_spark_semantics_for_nondeterministic_date_fixture(
        tmp_path,
        'pandas-date-diff.nq',
        'spark-date-diff.nq',
        mapping_path,
        csv_path,
        {
            '<http://example.com#date_now>',
            '<http://example.com#date_diff>',
        },
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_semantics_for_nondeterministic_date_inc_fixture(tmp_path):
    mapping_path = Path('test/rml-fnml/date_functions/date_inc/mapping.yarrrml').resolve().as_posix()
    csv_path = Path('test/rml-fnml/date_functions/cars.csv').resolve().as_posix()

    _assert_spark_semantics_for_nondeterministic_date_fixture(
        tmp_path,
        'pandas-date-inc.nq',
        'spark-date-inc.nq',
        mapping_path,
        csv_path,
        {
            '<http://example.com#date_datePart>',
        },
    )


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_semantics_for_nondeterministic_math_random_fixture(tmp_path):
    _assert_spark_semantics_for_nondeterministic_math_random_fixture(tmp_path)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_semantics_for_nondeterministic_uuid_fixture(tmp_path):
    _assert_spark_semantics_for_nondeterministic_uuid_fixture(tmp_path)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_array_join_fixture(tmp_path):
    pandas_output = tmp_path / 'pandas-array-join.nq'
    spark_output = tmp_path / 'spark-array-join.nq'
    mapping_path = Path('test/rml-fnml/array_functions/array_join/mapping.yarrrml').resolve().as_posix()
    csv_path = Path('test/rml-fnml/array_functions/cars.csv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_array_reverse_fixture(tmp_path):
    pandas_output = tmp_path / 'pandas-array-reverse.nq'
    spark_output = tmp_path / 'spark-array-reverse.nq'
    mapping_path = Path('test/rml-fnml/array_functions/array_reverse/mapping.yarrrml').resolve().as_posix()
    csv_path = Path('test/rml-fnml/array_functions/cars.csv').resolve().as_posix()

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
        file_path={csv_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_native_array_sort_fixture(tmp_path):
    csv_path = tmp_path / 'values.csv'
    csv_path.write_text('id,values\n1,c,b,a\n2,z,a,m\n', encoding='utf-8')

    mapping_path = tmp_path / 'mapping.ttl'
    mapping_path.write_text(textwrap.dedent(f'''
        @prefix ex: <http://example.com/> .
        @prefix rml: <http://w3id.org/rml/> .
        @prefix grel: <http://users.ugent.be/~bjdmeest/function/grel.ttl#> .

        <TriplesMap1>
            rml:logicalSource [
                rml:source "{csv_path.as_posix()}";
                rml:referenceFormulation rml:CSV
            ];
            rml:subjectMap [
                rml:template "http://example.com/{{id}}"
            ];
            rml:predicateObjectMap [
                rml:predicate ex:value;
                rml:objectMap [
                    rml:functionExecution <#OuterExecution> ;
                ]
            ] .

        <#OuterExecution>
            rml:function <http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join> ;
            rml:input [
                rml:parameter <http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a> ;
                rml:inputValueMap [
                    rml:functionExecution <#InnerExecution>
                ]
            ] ,
            [
                rml:parameter <http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep> ;
                rml:inputValue "|"
            ] .

        <#InnerExecution>
            rml:function <http://users.ugent.be/~bjdmeest/function/grel.ttl#array_sort> ;
            rml:input [
                rml:parameter <http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a> ;
                rml:inputValueMap [
                    rml:functionExecution <#SplitExecution>
                ]
            ] .

        <#SplitExecution>
            rml:function <http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split> ;
            rml:input [
                rml:parameter <http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter> ;
                rml:inputValueMap [
                    rml:reference "values"
                ]
            ] ,
            [
                rml:parameter <http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep> ;
                rml:inputValue ","
            ] .
    '''), encoding='utf-8')

    pandas_output = tmp_path / 'pandas-array-sort.nq'
    spark_output = tmp_path / 'spark-array-sort.nq'

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1

        [DataSource]
        mappings={mapping_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_declared_array_udf_fixture(tmp_path):
    csv_path = tmp_path / 'values.csv'
    csv_path.write_text('id,values\n1,a|b\n2,c\n', encoding='utf-8')

    udf_path = tmp_path / 'udf.py'
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

    mapping_path = tmp_path / 'mapping.ttl'
    mapping_path.write_text(textwrap.dedent(f'''
        @prefix ex: <http://example.com/> .
        @prefix rml: <http://w3id.org/rml/> .
        @prefix grel: <http://users.ugent.be/~bjdmeest/function/grel.ttl#> .

        <TriplesMap1>
            rml:logicalSource [
                rml:source "{csv_path.as_posix()}";
                rml:referenceFormulation rml:CSV
            ];
            rml:subjectMap [
                rml:template "http://example.com/{{id}}"
            ];
            rml:predicateObjectMap [
                rml:predicate ex:value;
                rml:objectMap [
                    rml:functionExecution <#Execution> ;
                ]
            ] .

        <#Execution>
            rml:function <http://example.com/split-array> ;
            rml:input [
                rml:parameter <http://example.com/value> ;
                rml:inputValueMap [
                    rml:reference "values"
                ]
            ] .
    '''), encoding='utf-8')

    pandas_output = tmp_path / 'pandas-array-udf.nq'
    spark_output = tmp_path / 'spark-array-udf.nq'

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1
        udfs={udf_path}

        [DataSource]
        mappings={mapping_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1
        udfs={udf_path}

        [DataSource]
        mappings={mapping_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)


@pytest.mark.spark
@pytest.mark.skipif(not SPARK_AVAILABLE, reason='PySpark or Java runtime is unavailable.')
def test_spark_matches_pandas_for_composite_declared_array_udf_fixture(tmp_path):
    csv_path = tmp_path / 'values.csv'
    csv_path.write_text('id,values\n1,a|b\n2,c|d|e\n', encoding='utf-8')

    udf_path = tmp_path / 'udf.py'
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

    mapping_path = tmp_path / 'mapping.ttl'
    mapping_path.write_text(textwrap.dedent(f'''
        @prefix ex: <http://example.com/> .
        @prefix rml: <http://w3id.org/rml/> .
        @prefix grel: <http://users.ugent.be/~bjdmeest/function/grel.ttl#> .

        <TriplesMap1>
            rml:logicalSource [
                rml:source "{csv_path.as_posix()}";
                rml:referenceFormulation rml:CSV
            ];
            rml:subjectMap [
                rml:template "http://example.com/{{id}}"
            ];
            rml:predicateObjectMap [
                rml:predicate ex:value;
                rml:objectMap [
                    rml:functionExecution <#OuterExecution> ;
                ]
            ] .

        <#OuterExecution>
            rml:function <http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join> ;
            rml:input [
                rml:parameter <http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a> ;
                rml:inputValueMap [
                    rml:functionExecution <#InnerExecution>
                ]
            ] ,
            [
                rml:parameter <http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep> ;
                rml:inputValue ", "
            ] .

        <#InnerExecution>
            rml:function <http://example.com/split-array> ;
            rml:input [
                rml:parameter <http://example.com/value> ;
                rml:inputValueMap [
                    rml:reference "values"
                ]
            ] .
    '''), encoding='utf-8')

    pandas_output = tmp_path / 'pandas-composite-array-udf.nq'
    spark_output = tmp_path / 'spark-composite-array-udf.nq'

    pandas_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={pandas_output}
        output_format=N-QUADS
        number_of_processes=1
        udfs={udf_path}

        [DataSource]
        mappings={mapping_path}
    ''')
    spark_config = textwrap.dedent(f'''
        [CONFIGURATION]
        output_file={spark_output}
        output_format=N-QUADS
        execution_engine=spark
        number_of_processes=1
        udfs={udf_path}

        [DataSource]
        mappings={mapping_path}
    ''')

    _write_output_via_backend(pandas_config)
    _write_output_via_backend(spark_config)

    assert canonicalize_rdf_output_path(pandas_output) == canonicalize_rdf_output_path(spark_output)
