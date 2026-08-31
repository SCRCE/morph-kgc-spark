__author__ = "Ahmad Hammad"
__credits__ = ["Julián Arenas-Guerrero", "Ahmad Hammad"]

__license__ = "Apache-2.0"
__maintainer__ = "Ahmad Hammad"
__email__ = "Ahmad.Hammad@ieee.org"


import os
import re
import shutil
import tempfile
from functools import reduce
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from pyspark.sql import functions as sf
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType

from ..compat import encode_iri_value
from ..data_source.data_file import get_file_data
from ..data_source.python_data import get_ram_data
from ..data_source.relational_db import _build_sql_query
from ..data_source.relational_db import _replace_query_enclosing_characters
from ..data_source.relational_db import get_sql_data
from ..constants import CSV
from ..constants import JSON
from ..constants import NQUADS
from ..constants import ORC
from ..constants import PARQUET
from ..constants import GEOPARQUET
from ..constants import PGDB
from ..constants import POSTGRESQL
from ..constants import PYTHON_SOURCE
from ..constants import RDB
from ..constants import RML_BLANK_NODE
from ..constants import RML_CONSTANT
from ..constants import RML_DATATYPE_MAP
from ..constants import RML_DEFAULT_GRAPH
from ..constants import RML_EXECUTION
from ..constants import RML_IRI
from ..constants import RML_LANGUAGE_MAP
from ..constants import RML_LITERAL
from ..constants import RML_PARENT_TRIPLES_MAP
from ..constants import RML_QUOTED_TRIPLES_MAP
from ..constants import RML_QUERY
from ..constants import RML_TABLE_NAME
from ..constants import RML_REFERENCE
from ..constants import RML_TEMPLATE
from ..constants import TSV
from ..constants import XML
from ..constants import XSD_BOOLEAN
from ..constants import XSD_DATETIME
from ..constants import XSD_INTEGER
from ..constants import XSD_NONNEGATIVEINTEGER
from ..errors import SparkUnsupportedFeature
from ..fnml.function_executor import SparkFunctionExecutor
from ..materializer import _get_references_in_rml_rule
from ..utils import create_dirs_in_path
from ..utils import get_references_in_join_condition
from ..utils import get_references_in_template
from ..utils import get_rml_rule
from ..utils import normalize_oracle_identifier_casing
from .spark_jdbc import postgresql_jdbc_options


_ENCODE_IRI_UDF = sf.udf(lambda value: None if value is None else encode_iri_value(value), StringType())
_REMOVE_NON_PRINTABLE_UDF = sf.udf(
    lambda value: None if value is None else ''.join(char for char in value if char.isprintable()),
    StringType(),
)


def _pandas_df_to_string_rows(pandas_df):
    return [
        tuple(None if pd.isna(value) else str(value) for value in row)
        for row in pandas_df.itertuples(index=False, name=None)
    ]


class SparkMaterializer:

    def __init__(self, spark, config, python_source=None):
        self.spark = spark
        self.config = config
        self.python_source = python_source
        self.function_executor = SparkFunctionExecutor(config)
        self._temporary_query_dirs = []

    def write_mapping_groups(self, mapping_groups, rml_df, fnml_df):
        try:
            if not self.config.get_output_dir():
                triples_df = None
                for mapping_group in mapping_groups:
                    group_triples_df = self._materialize_mapping_group(mapping_group, rml_df, fnml_df)
                    triples_df = group_triples_df if triples_df is None else triples_df.unionByName(group_triples_df)

                if triples_df is None:
                    triples_df = self.spark.range(0).select(self._column('id').cast('string').alias('value'))
                else:
                    triples_df = triples_df.dropDuplicates(['value'])

                output_path = self.config.get_output_file_path(None)
                self._prepare_output_path(output_path)
                self._write_text_output(triples_df, output_path)
                return

            for mapping_group in mapping_groups:
                triples_df = self._materialize_mapping_group(mapping_group, rml_df, fnml_df)
                output_path = self.config.get_output_file_path(mapping_group.iloc[0]['mapping_partition'])
                self._prepare_output_path(output_path)
                self._write_text_output(triples_df, output_path)
        finally:
            self._cleanup_temporary_query_dirs()

    def _cleanup_temporary_query_dirs(self):
        for temp_dir in self._temporary_query_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._temporary_query_dirs = []

    def _prepare_output_path(self, output_path):
        create_dirs_in_path(output_path)
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        elif os.path.exists(output_path):
            os.remove(output_path)

    def _write_text_output(self, triples_df, output_path):
        output_parent = Path(output_path).parent
        temp_dir = tempfile.mkdtemp(prefix='morph-kgc-spark-output-', dir=str(output_parent) if str(output_parent) else None)
        try:
            triples_df.write.mode('overwrite').text(temp_dir)
            with open(output_path, 'wb') as destination:
                for part_file in sorted(Path(temp_dir).glob('part-*')):
                    with open(part_file, 'rb') as source:
                        shutil.copyfileobj(source, destination)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _column(self, column_name):
        escaped_column_name = str(column_name).replace('`', '``')
        return sf.col(f'`{escaped_column_name}`')

    def _materialize_mapping_group(self, mapping_group_df, rml_df, fnml_df):
        triples_df = None
        for _, rml_rule in mapping_group_df.iterrows():
            rule_df = self._materialize_rule(rml_rule, rml_df, fnml_df)
            triples_df = rule_df if triples_df is None else triples_df.unionByName(rule_df)

        if triples_df is None:
            return self.spark.range(0).select(self._column('id').cast('string').alias('value'))

        return triples_df.dropDuplicates(['value'])

    def _materialize_rule(self, rml_rule, rml_df, fnml_df):
        self._ensure_rule_is_supported(rml_rule, fnml_df)
        if rml_rule['object_map_type'] == RML_PARENT_TRIPLES_MAP:
            return self._materialize_parent_triples_rule(rml_rule, rml_df, fnml_df)

        references = set(_get_references_in_rml_rule(rml_rule, rml_df, fnml_df))
        data = self._load_source_data(rml_rule, references)
        data = self._apply_quoted_triples_joins(data, rml_rule, rml_df, fnml_df)
        data = self._apply_fnml_dataframe_transforms(data, rml_rule, fnml_df)

        subject = self._materialize_term_value(
            data,
            rml_rule,
            'subject',
            rml_df,
            fnml_df,
        )
        predicate = self._materialize_term(
            data,
            rml_rule['predicate_map_value'],
            rml_rule['predicate_map_type'],
            RML_IRI,
            fnml_df=fnml_df,
        )
        obj = self._materialize_term_value(
            data,
            rml_rule,
            'object',
            rml_df,
            fnml_df,
        )

        if rml_rule['lang_datatype'] == RML_LANGUAGE_MAP:
            lang = self._materialize_term(
                data,
                rml_rule['lang_datatype_map_value'],
                rml_rule['lang_datatype_map_type'],
                '',
                fnml_df=fnml_df,
            )
            obj = sf.concat(obj, sf.lit('@'), lang)
        elif rml_rule['lang_datatype'] == RML_DATATYPE_MAP:
            datatype = self._materialize_term(
                data,
                rml_rule['lang_datatype_map_value'],
                rml_rule['lang_datatype_map_type'],
                RML_IRI,
                fnml_df=fnml_df,
            )
            obj = sf.concat(obj, sf.lit('^^'), datatype)

        triple = sf.concat(subject, sf.lit(' '), predicate, sf.lit(' '), obj)
        triple_suffix = sf.lit(' .')
        if self.config.get_output_format() == NQUADS and rml_rule['graph_map_value'] != RML_DEFAULT_GRAPH:
            graph = self._materialize_term(
                data,
                rml_rule['graph_map_value'],
                rml_rule['graph_map_type'],
                RML_IRI,
                fnml_df=fnml_df,
            )
            triple = sf.concat(triple, sf.lit(' '), graph)
        elif self.config.get_output_format() == NQUADS:
            triple_suffix = sf.lit('  .')

        return data.select(sf.concat(triple, triple_suffix).alias('value'))

    def _materialize_parent_triples_rule(self, rml_rule, rml_df, fnml_df):
        parent_rml_rule = get_rml_rule(rml_df, rml_rule['object_map_value'])
        child_join_references, parent_join_references = get_references_in_join_condition(
            rml_rule,
            'object_join_conditions',
        )
        child_references = set(_get_references_in_rml_rule(rml_rule, rml_df, fnml_df))

        if not child_join_references or not parent_join_references:
            if not self._same_logical_source(rml_rule, parent_rml_rule):
                raise SparkUnsupportedFeature(
                    rml_rule['triples_map_id'],
                    'parent triples maps without explicit join conditions are only supported when child and parent use the same logical source in the Spark backend right now.'
                )

            parent_references = set(_get_references_in_rml_rule(parent_rml_rule, rml_df, fnml_df, only_subject_map=True))
            child_references.update(parent_references)
            data = self._load_source_data(rml_rule, child_references)
        else:
            child_data = self._load_source_data(rml_rule, child_references)
            data = self._join_parent_triples_map_data(
                child_data,
                rml_rule,
                parent_rml_rule,
                rml_df,
                fnml_df,
            )
        data = self._apply_fnml_dataframe_transforms(data, rml_rule, fnml_df)

        subject = self._materialize_term(
            data,
            rml_rule['subject_map_value'],
            rml_rule['subject_map_type'],
            rml_rule['subject_termtype'],
            fnml_df=fnml_df,
        )
        predicate = self._materialize_term(
            data,
            rml_rule['predicate_map_value'],
            rml_rule['predicate_map_type'],
            RML_IRI,
            fnml_df=fnml_df,
        )
        obj = self._materialize_term(
            data,
            parent_rml_rule['subject_map_value'],
            parent_rml_rule['subject_map_type'],
            parent_rml_rule['subject_termtype'],
            column_prefix='parent_',
            fnml_df=fnml_df,
        )

        triple = sf.concat(subject, sf.lit(' '), predicate, sf.lit(' '), obj)
        triple_suffix = sf.lit(' .')
        if self.config.get_output_format() == NQUADS and rml_rule['graph_map_value'] != RML_DEFAULT_GRAPH:
            graph = self._materialize_term(
                data,
                rml_rule['graph_map_value'],
                rml_rule['graph_map_type'],
                RML_IRI,
                fnml_df=fnml_df,
            )
            triple = sf.concat(triple, sf.lit(' '), graph)
        elif self.config.get_output_format() == NQUADS:
            triple_suffix = sf.lit('  .')

        return data.select(sf.concat(triple, triple_suffix).alias('value'))

    def _same_logical_source(self, child_rml_rule, parent_rml_rule):
        return (
            child_rml_rule['source_type'] == parent_rml_rule['source_type']
            and child_rml_rule['logical_source_type'] == parent_rml_rule['logical_source_type']
            and child_rml_rule['logical_source_value'] == parent_rml_rule['logical_source_value']
            and child_rml_rule['iterator'] == parent_rml_rule['iterator']
        )

    def _join_parent_triples_map_data(
        self,
        child_data,
        child_rml_rule,
        parent_rml_rule,
        rml_df,
        fnml_df,
        child_column_prefix='',
        parent_column_prefix='parent_',
    ):
        child_join_references, parent_join_references = get_references_in_join_condition(
            child_rml_rule,
            'object_join_conditions',
        )
        if not child_join_references or not parent_join_references:
            raise SparkUnsupportedFeature(
                child_rml_rule['triples_map_id'],
                'parent triples maps without explicit join conditions are not implemented in the Spark backend yet.'
            )

        parent_references = set(_get_references_in_rml_rule(parent_rml_rule, rml_df, fnml_df, only_subject_map=True))
        parent_references.update(parent_join_references)
        parent_data = self._load_source_data(parent_rml_rule, parent_references)
        parent_data = parent_data.select(
            *[self._column(column_name).alias(f'{parent_column_prefix}{column_name}') for column_name in parent_data.columns]
        )

        join_condition = None
        for child_reference, parent_reference in zip(child_join_references, parent_join_references):
            predicate = (
                self._column(f'{child_column_prefix}{child_reference}')
                == self._column(f'{parent_column_prefix}{parent_reference}')
            )
            join_condition = predicate if join_condition is None else (join_condition & predicate)

        return child_data.join(parent_data, on=join_condition, how='inner')

    def _apply_fnml_dataframe_transforms(self, data, rml_rule, fnml_df):
        execution_positions = [
            ('subject_map_type', 'subject_map_value'),
            ('predicate_map_type', 'predicate_map_value'),
            ('object_map_type', 'object_map_value'),
            ('graph_map_type', 'graph_map_value'),
            ('lang_datatype_map_type', 'lang_datatype_map_value'),
        ]

        for map_type_key, map_value_key in execution_positions:
            if rml_rule[map_type_key] == RML_EXECUTION:
                data, _ = self.function_executor.apply_execution(
                    data,
                    fnml_df,
                    rml_rule[map_value_key],
                    self,
                )

        return data

    def _apply_quoted_triples_joins(self, data, rml_rule, rml_df, fnml_df):
        for position in ['subject', 'object']:
            if rml_rule[f'{position}_map_type'] != RML_QUOTED_TRIPLES_MAP:
                continue

            parent_rml_rule = get_rml_rule(rml_df, rml_rule[f'{position}_map_value'])
            join_conditions_key = f'{position}_join_conditions'
            if pd.isna(rml_rule[join_conditions_key]) or not rml_rule[join_conditions_key]:
                parent_prefix = ''
            else:
                parent_references = set(_get_references_in_rml_rule(parent_rml_rule, rml_df, fnml_df))
                child_join_references, parent_join_references = get_references_in_join_condition(
                    rml_rule,
                    join_conditions_key,
                )
                parent_references.update(parent_join_references)
                parent_data = self._load_source_data(parent_rml_rule, parent_references)

                parent_prefix = f'{position}_parent_'
                parent_data = parent_data.select(
                    *[self._column(column_name).alias(f'{parent_prefix}{column_name}') for column_name in parent_data.columns]
                )

                join_condition = None
                for child_reference, parent_reference in zip(child_join_references, parent_join_references):
                    predicate = self._column(child_reference) == self._column(f'{parent_prefix}{parent_reference}')
                    join_condition = predicate if join_condition is None else (join_condition & predicate)

                data = data.join(parent_data, on=join_condition, how='inner')

            if (
                pd.notna(parent_rml_rule['object_join_conditions'])
                and parent_rml_rule['object_join_conditions']
                and parent_rml_rule['object_map_type'] == RML_PARENT_TRIPLES_MAP
            ):
                grandparent_rml_rule = get_rml_rule(rml_df, parent_rml_rule['object_map_value'])
                grandparent_prefix = f'{parent_prefix}parent_'
                data = self._join_parent_triples_map_data(
                    data,
                    parent_rml_rule,
                    grandparent_rml_rule,
                    rml_df,
                    fnml_df,
                    child_column_prefix=parent_prefix,
                    parent_column_prefix=grandparent_prefix,
                )

        return data

    def _ensure_rule_is_supported(self, rml_rule, fnml_df):
        triples_map_id = rml_rule['triples_map_id']

        unsupported_term_map_types = {
            'subject': rml_rule['subject_map_type'],
            'predicate': rml_rule['predicate_map_type'],
            'object': rml_rule['object_map_type'],
            'graph': rml_rule['graph_map_type'],
            'lang/datatype': rml_rule['lang_datatype_map_type'],
        }
        for term_name, term_map_type in unsupported_term_map_types.items():
            if term_map_type == RML_EXECUTION:
                execution_id = rml_rule[f'{term_name}_map_value'] if term_name != 'lang/datatype' else rml_rule['lang_datatype_map_value']
                self.function_executor.assert_supported(
                    fnml_df,
                    execution_id,
                    triples_map_id=triples_map_id,
                    position=f'{triples_map_id}:{term_name}',
                )

        if (
            pd.notna(rml_rule['subject_join_conditions'])
            and rml_rule['subject_join_conditions']
            and rml_rule['subject_map_type'] != RML_QUOTED_TRIPLES_MAP
        ):
            raise SparkUnsupportedFeature(
                triples_map_id,
                'subject join conditions are not implemented in the Spark backend yet.'
            )

        if (
            pd.notna(rml_rule['object_join_conditions'])
            and rml_rule['object_join_conditions']
            and rml_rule['object_map_type'] not in [RML_PARENT_TRIPLES_MAP, RML_QUOTED_TRIPLES_MAP]
        ):
            raise SparkUnsupportedFeature(
                triples_map_id,
                'object join conditions are only supported for parent triples maps in the Spark backend right now.'
            )

        if rml_rule['source_type'] == PGDB:
            raise SparkUnsupportedFeature(
                triples_map_id,
                'property graph sources are not implemented in the Spark backend yet.'
            )

        if rml_rule['source_type'] == GEOPARQUET:
            raise SparkUnsupportedFeature(
                triples_map_id,
                'GeoParquet sources are not implemented in the Spark backend yet.'
            )

        supported_sources = {CSV, TSV, PARQUET, ORC, PYTHON_SOURCE, RDB}
        if rml_rule['source_type'] not in supported_sources and rml_rule['source_type'] not in JSON and rml_rule['source_type'] not in XML:
            raise SparkUnsupportedFeature(
                triples_map_id,
                f"source type `{rml_rule['source_type']}` is not implemented in the Spark backend yet."
            )

        if rml_rule['logical_source_type'] != rml_rule['logical_source_type']:
            raise SparkUnsupportedFeature(
                triples_map_id,
                'unexpected null logical source type.'
            )

        supported_logical_sources = {RML_QUERY}
        if rml_rule['source_type'] == RDB:
            supported_logical_sources.add(RML_TABLE_NAME)
        else:
            supported_logical_sources.add('http://w3id.org/rml/source')

        if rml_rule['logical_source_type'] not in supported_logical_sources:
            raise SparkUnsupportedFeature(
                triples_map_id,
                f"logical source type `{rml_rule['logical_source_type']}` is not implemented in the Spark backend yet."
            )

    def _load_source_data(self, rml_rule, references):
        if not references:
            return self.spark.range(1).drop('id')

        references = list(references)
        if rml_rule['source_type'] == RDB:
            data = self._load_rdb_source_data(rml_rule, references)
        elif rml_rule['logical_source_type'] == RML_QUERY:
            data = self._load_query_source_data(rml_rule)
        elif rml_rule['source_type'] == PYTHON_SOURCE:
            data = self._load_python_source_data(rml_rule, references)
        elif rml_rule['source_type'] in [CSV, TSV]:
            separator = ',' if rml_rule['source_type'] == CSV else '\t'
            data = self.spark.read.options(
                header=True,
                sep=separator,
                inferSchema=False,
                multiLine=True,
                quote='"',
                escape='"',
            ).csv(rml_rule['logical_source_value'])
        elif rml_rule['source_type'] == PARQUET:
            data = self.spark.read.parquet(rml_rule['logical_source_value'])
        elif rml_rule['source_type'] == ORC:
            data = self.spark.read.orc(rml_rule['logical_source_value'])
        elif rml_rule['source_type'] in JSON or rml_rule['source_type'] in XML:
            data = self._load_locally_preprocessed_file_source_data(rml_rule, references)
        else:
            raise SparkUnsupportedFeature(
                rml_rule['triples_map_id'],
                f"source type `{rml_rule['source_type']}` is not implemented in the Spark backend yet."
            )

        missing_columns = sorted(set(references).difference(set(data.columns)))
        if missing_columns:
            raise SparkUnsupportedFeature(
                rml_rule['triples_map_id'],
                f'source is missing referenced columns: {missing_columns}'
            )

        projected = data.select(*[self._column(reference).cast('string').alias(reference) for reference in references])
        if self.config.get_na_values():
            projected = projected.replace(self.config.get_na_values(), None)
            projected = projected.filter(
                reduce(
                    lambda left, right: left & right,
                    [self._column(column_name).isNotNull() for column_name in projected.columns],
                )
            )

        return projected.dropDuplicates()

    def _load_rdb_source_data(self, rml_rule, references):
        if self.config.get_spark_rdb_mode(rml_rule['source_name']) == 'jdbc':
            return self._load_jdbc_source_data(rml_rule, references)
        return self._load_locally_preprocessed_rdb_source_data(rml_rule, references)

    def _load_jdbc_source_data(self, rml_rule, references):
        source_name = rml_rule['source_name']
        partition_column = self.config.get_spark_jdbc_partition_column(source_name)
        query_references = list(dict.fromkeys([*references, partition_column]))
        query = _build_sql_query(rml_rule, query_references)
        if query is None:
            raise SparkUnsupportedFeature(
                rml_rule['triples_map_id'],
                'Spark JDBC could not build a query for this logical source. Use `spark_rdb_mode = local_preprocess`.'
            )
        query = _replace_query_enclosing_characters(query.rstrip().rstrip(';'), POSTGRESQL)

        try:
            options = postgresql_jdbc_options(self.config.get_db_url(source_name))
        except ValueError as exception:
            raise SparkUnsupportedFeature(rml_rule['triples_map_id'], str(exception)) from exception

        options.update({
            'dbtable': f'({query}) AS morph_kgc_source',
            'partitionColumn': partition_column,
            'lowerBound': str(self.config.get_spark_jdbc_lower_bound(source_name)),
            'upperBound': str(self.config.get_spark_jdbc_upper_bound(source_name)),
            'numPartitions': str(self.config.get_spark_jdbc_num_partitions(source_name)),
            'fetchsize': str(self.config.get_spark_jdbc_fetch_size(source_name)),
        })

        reader = self.spark.read.format('jdbc')
        for option, value in options.items():
            reader = reader.option(option, value)
        try:
            return reader.load()
        except Exception as exception:
            message = str(exception)
            if 'org.postgresql.Driver' in message or 'No suitable driver' in message:
                raise SparkUnsupportedFeature(
                    rml_rule['triples_map_id'],
                    'PostgreSQL JDBC driver is unavailable. Set `SPARK_JDBC_JAR` before starting Spark, '
                    'or configure `spark.jars.packages=org.postgresql:postgresql:<version>`.'
                ) from exception
            if 'partitionColumn' in message and any(word in message.lower() for word in ('numeric', 'date', 'timestamp')):
                raise SparkUnsupportedFeature(
                    rml_rule['triples_map_id'],
                    f'Spark JDBC partition column `{partition_column}` must have an integral, date, or timestamp type. '
                    'Choose a compatible column or use `spark_rdb_mode = local_preprocess`.'
                ) from exception
            raise

    def _materialize_term_value(self, data, rml_rule, position, rml_df, fnml_df, column_prefix=''):
        map_type = rml_rule[f'{position}_map_type']
        map_value = rml_rule[f'{position}_map_value']

        if map_type == RML_QUOTED_TRIPLES_MAP:
            quoted_triple = self._materialize_embedded_triple(
                data,
                rml_rule,
                position,
                map_value,
                rml_df,
                fnml_df,
                column_prefix=column_prefix,
            )
            return sf.concat(sf.lit('<< '), quoted_triple, sf.lit(' >>'))

        datatype = rml_rule['lang_datatype_map_value'] if position == 'object' else ''
        return self._materialize_term(
            data,
            map_value,
            map_type,
            rml_rule[f'{position}_termtype'],
            datatype=datatype,
            column_prefix=column_prefix,
            fnml_df=fnml_df,
        )

    def _materialize_embedded_triple(self, data, current_rml_rule, position, triples_map_id, rml_df, fnml_df, column_prefix=''):
        parent_rml_rule = get_rml_rule(rml_df, triples_map_id)

        join_conditions_key = f'{position}_join_conditions'
        join_conditions = current_rml_rule[join_conditions_key]
        if pd.notna(join_conditions) and join_conditions:
            if column_prefix:
                raise SparkUnsupportedFeature(
                    triples_map_id,
                    f'nested quoted triples with {position} join conditions are not implemented in the Spark backend yet.'
                )
            parent_prefix = f'{position}_parent_'
        else:
            parent_prefix = column_prefix

        if pd.notna(parent_rml_rule['subject_join_conditions']) and parent_rml_rule['subject_join_conditions']:
            raise SparkUnsupportedFeature(
                triples_map_id,
                'nested quoted triples with subject join conditions are not implemented in the Spark backend yet.'
            )

        if pd.notna(parent_rml_rule['object_join_conditions']) and parent_rml_rule['object_join_conditions']:
            if parent_rml_rule['object_map_type'] != RML_PARENT_TRIPLES_MAP:
                raise SparkUnsupportedFeature(
                    triples_map_id,
                    'nested quoted triples with object join conditions are not implemented in the Spark backend yet.'
                )

            grandparent_rml_rule = get_rml_rule(rml_df, parent_rml_rule['object_map_value'])
            grandparent_prefix = f'{parent_prefix}parent_'
            parent_data = self._apply_fnml_dataframe_transforms(data, parent_rml_rule, fnml_df)
            subject = self._materialize_term_value(
                parent_data,
                parent_rml_rule,
                'subject',
                rml_df,
                fnml_df,
                column_prefix=parent_prefix,
            )
            predicate = self._materialize_term(
                parent_data,
                parent_rml_rule['predicate_map_value'],
                parent_rml_rule['predicate_map_type'],
                RML_IRI,
                column_prefix=parent_prefix,
                fnml_df=fnml_df,
            )
            obj = self._materialize_term(
                parent_data,
                grandparent_rml_rule['subject_map_value'],
                grandparent_rml_rule['subject_map_type'],
                grandparent_rml_rule['subject_termtype'],
                column_prefix=grandparent_prefix,
                fnml_df=fnml_df,
            )
            return sf.concat(subject, sf.lit(' '), predicate, sf.lit(' '), obj)

        parent_data = self._apply_fnml_dataframe_transforms(data, parent_rml_rule, fnml_df)
        subject = self._materialize_term_value(parent_data, parent_rml_rule, 'subject', rml_df, fnml_df, column_prefix=parent_prefix)
        predicate = self._materialize_term(
            parent_data,
            parent_rml_rule['predicate_map_value'],
            parent_rml_rule['predicate_map_type'],
            RML_IRI,
            column_prefix=parent_prefix,
            fnml_df=fnml_df,
        )
        obj = self._materialize_term_value(parent_data, parent_rml_rule, 'object', rml_df, fnml_df, column_prefix=parent_prefix)

        if parent_rml_rule['lang_datatype'] == RML_LANGUAGE_MAP:
            lang = self._materialize_term(
                parent_data,
                parent_rml_rule['lang_datatype_map_value'],
                parent_rml_rule['lang_datatype_map_type'],
                '',
                column_prefix=parent_prefix,
                fnml_df=fnml_df,
            )
            obj = sf.concat(obj, sf.lit('@'), lang)
        elif parent_rml_rule['lang_datatype'] == RML_DATATYPE_MAP:
            datatype = self._materialize_term(
                parent_data,
                parent_rml_rule['lang_datatype_map_value'],
                parent_rml_rule['lang_datatype_map_type'],
                RML_IRI,
                column_prefix=parent_prefix,
                fnml_df=fnml_df,
            )
            obj = sf.concat(obj, sf.lit('^^'), datatype)

        return sf.concat(subject, sf.lit(' '), predicate, sf.lit(' '), obj)

    def _load_query_source_data(self, rml_rule):
        import duckdb

        pandas_df = duckdb.query(str(rml_rule['logical_source_value']).strip()).df()
        return self._create_string_dataframe_from_pandas(pandas_df)

    def _load_locally_preprocessed_file_source_data(self, rml_rule, references):
        pandas_df = get_file_data(rml_rule, references).copy()

        if references:
            missing_columns = sorted(set(references).difference(set(pandas_df.columns)))
            if missing_columns:
                raise SparkUnsupportedFeature(
                    rml_rule['triples_map_id'],
                    f'source is missing referenced columns: {missing_columns}'
                )
            pandas_df = pandas_df[references]

        return self._create_string_dataframe_from_pandas(pandas_df)

    def _load_locally_preprocessed_rdb_source_data(self, rml_rule, references):
        pandas_df = get_sql_data(self.config, rml_rule, references).copy()
        db_url = self.config.get_db_url(rml_rule['source_name'])
        if db_url.lower().startswith('oracle'):
            pandas_df = normalize_oracle_identifier_casing(pandas_df, references)

        if references:
            missing_columns = sorted(set(references).difference(set(pandas_df.columns)))
            if missing_columns:
                raise SparkUnsupportedFeature(
                    rml_rule['triples_map_id'],
                    f'source is missing referenced columns: {missing_columns}'
                )
            pandas_df = pandas_df[references]

        return self._create_string_dataframe_from_pandas(pandas_df)

    def _create_string_dataframe_from_pandas(self, pandas_df):
        rows = _pandas_df_to_string_rows(pandas_df)
        schema = StructType([StructField(column_name, StringType(), True) for column_name in pandas_df.columns])
        return self.spark.createDataFrame(rows, schema=schema)

    def _load_python_source_data(self, rml_rule, references):
        if self.python_source is None:
            raise SparkUnsupportedFeature(
                rml_rule['triples_map_id'],
                'in-memory Python sources require providing `python_source` directly to the Spark file-output backend.'
            )

        try:
            pandas_df = get_ram_data(rml_rule, references, self.python_source)
        except KeyError as e:
            raise SparkUnsupportedFeature(
                rml_rule['triples_map_id'],
                f'source is missing referenced columns: {[str(e).strip(chr(39))]}'
            ) from e

        pandas_df = pandas_df.copy()
        missing_columns = sorted(set(references).difference(set(pandas_df.columns)))
        if missing_columns:
            raise SparkUnsupportedFeature(
                rml_rule['triples_map_id'],
                f'source is missing referenced columns: {missing_columns}'
            )

        pandas_df = pandas_df[references].astype(object)
        pandas_df = pandas_df.where(pd.notna(pandas_df), None)
        rows = [
            tuple(None if value is None else str(value) for value in row)
            for row in pandas_df.itertuples(index=False, name=None)
        ]
        schema = StructType([StructField(reference, StringType(), True) for reference in references])
        return self.spark.createDataFrame(rows, schema=schema)

    def _materialize_term(self, data, template, expression_type, termtype, datatype='', column_prefix='', fnml_df=None):
        if expression_type == RML_REFERENCE:
            term = self._prepare_reference_value(data, column_prefix + template, expression_type, termtype, datatype)
        elif expression_type == RML_CONSTANT:
            term = sf.lit(template)
        elif expression_type == RML_TEMPLATE:
            term = self._materialize_template(data, template, expression_type, termtype, datatype, column_prefix, fnml_df)
        elif expression_type == RML_EXECUTION:
            output_column = self.function_executor.get_execution_output_column(template)
            if output_column in data.columns:
                term = self._column(output_column)
            else:
                term = self.function_executor.build_column(data, fnml_df, template, self)
        else:
            raise SparkUnsupportedFeature(
                template,
                f'term map type `{expression_type}` is not implemented in the Spark backend yet.'
            )

        if termtype.strip() == RML_LITERAL and expression_type in {RML_CONSTANT, RML_EXECUTION}:
            term = self._normalize_literal_value(term, datatype)

        if termtype.strip() == RML_IRI:
            return sf.concat(sf.lit('<'), term, sf.lit('>'))
        if termtype.strip() == RML_BLANK_NODE:
            return sf.concat(sf.lit('_:'), term)
        if termtype.strip() == RML_LITERAL:
            return sf.concat(sf.lit('"'), term, sf.lit('"'))
        return term

    def _materialize_template(self, data, template, expression_type, termtype, datatype, column_prefix='', fnml_df=None):
        references = get_references_in_template(template)
        template = template.replace('\\{', '{').replace('\\}', '}')

        pieces = []
        remainder = template
        for reference in references:
            split_index = remainder.find('{' + reference + '}')
            prefix = remainder[:split_index]
            if prefix:
                pieces.append(sf.lit(prefix))
            pieces.append(self._prepare_reference_value(
                data,
                column_prefix + reference,
                expression_type,
                termtype,
                datatype,
            ))
            remainder = remainder[split_index + len(reference) + 2:]

        if remainder:
            pieces.append(sf.lit(remainder))

        if not pieces:
            return sf.lit('')
        if len(pieces) == 1:
            return pieces[0]
        return sf.concat(*pieces)

    def _prepare_reference_value(self, data, reference, expression_type, termtype, datatype):
        value = self._column(reference)
        if self.config.only_write_printable_characters():
            value = _REMOVE_NON_PRINTABLE_UDF(value)

        if termtype.strip() == RML_IRI and expression_type == RML_TEMPLATE:
            if self.config.get_safe_percent_encoding():
                safe_chars = self.config.get_safe_percent_encoding()
                encode_udf = sf.udf(
                    lambda item: None if item is None else quote(item, safe=safe_chars),
                    StringType(),
                )
                return encode_udf(value)
            return _ENCODE_IRI_UDF(value)

        if termtype.strip() == RML_LITERAL:
            value = self._normalize_literal_value(value, datatype)

        return value

    def _normalize_literal_value(self, value, datatype):
        if datatype == XSD_BOOLEAN:
            value = sf.lower(value)
        elif datatype == XSD_DATETIME:
            value = sf.regexp_replace(value, ' ', 'T')
        elif datatype in [XSD_INTEGER, XSD_NONNEGATIVEINTEGER]:
            value = value.cast('double').cast('long').cast('string')

        value = sf.regexp_replace(value, r'\\', r'\\\\')
        value = sf.regexp_replace(value, '\n', r'\\n')
        value = sf.regexp_replace(value, '\r', r'\\r')
        value = sf.regexp_replace(value, '"', r'\\\"')
        for char in self.config.get_literal_escaping_chars():
            if char not in ['"', '\n', '\\', '\r']:
                replacement = f'\\{char}' if char in ['\n', '\r', '\t', '\b', '\f'] else f'\\\\{char}'
                value = sf.regexp_replace(value, re.escape(char), replacement)

        return value
