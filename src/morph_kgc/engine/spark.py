__author__ = "Ahmad Hammad"
__credits__ = ["Julián Arenas-Guerrero", "Ahmad Hammad"]

__license__ = "Apache-2.0"
__maintainer__ = "Ahmad Hammad"
__email__ = "Ahmad.Hammad@ieee.org"


import importlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from ..constants import LOGGING_NAMESPACE
from ..errors import SparkDependencyError
from ..errors import SparkUnsupportedFeature
from .base import ExecutionBackend


LOGGER = logging.getLogger(LOGGING_NAMESPACE)


class SparkExecutionBackend(ExecutionBackend):

    def __init__(self, config):
        super().__init__(config)
        self._load_pyspark()
        self._disable_local_multiprocessing()
        self._distributed_udf_tempdirs = []

    def _load_pyspark(self):
        try:
            importlib.import_module('pyspark.sql')
        except ModuleNotFoundError as e:
            raise SparkDependencyError(
                'Spark execution requires optional PySpark dependencies. '
                "Install them first, for example with `pip install 'morph-kgc[spark]'`."
            ) from e

    def _get_spark_session(self):
        from pyspark.sql import SparkSession

        builder = SparkSession.builder.appName('morph-kgc')
        jdbc_jar = os.environ.get('SPARK_JDBC_JAR')
        if jdbc_jar:
            resolved_jar = Path(jdbc_jar).expanduser().resolve()
            if not resolved_jar.is_file():
                raise FileNotFoundError(f'SPARK_JDBC_JAR does not exist: {resolved_jar}')
            builder = builder.config('spark.jars', str(resolved_jar))
        return builder.getOrCreate()

    def _disable_local_multiprocessing(self):
        if self.config.get_number_of_processes() <= 1:
            return

        LOGGER.warning(
            'Spark execution ignores `number_of_processes` and does not use local multiprocessing inside Spark tasks. '
            'Resetting `number_of_processes` to 1 for this run.'
        )
        self.config.set_number_of_processes('1')

    def _distribute_udf_dependencies(self, spark):
        udf_path = self.config.get_udfs()
        if not udf_path:
            return

        resolved_udf_path = Path(udf_path).resolve()
        if not resolved_udf_path.exists():
            raise FileNotFoundError(f'Spark UDF file does not exist: {resolved_udf_path}')

        temp_dir = tempfile.mkdtemp(prefix='morph-kgc-spark-udf-')
        self._distributed_udf_tempdirs.append(temp_dir)

        distributed_module_name = f'morph_kgc_udfs_{resolved_udf_path.stem}.py'
        distributed_module_path = Path(temp_dir) / distributed_module_name
        shutil.copyfile(resolved_udf_path, distributed_module_path)

        spark.sparkContext.addPyFile(str(distributed_module_path))

    def _cleanup_distributed_udf_dependencies(self):
        for temp_dir in self._distributed_udf_tempdirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._distributed_udf_tempdirs = []

    def _unsupported(self, feature):
        raise SparkUnsupportedFeature(
            feature,
            'the backend selection and dependency boundary are in place, '
            'but the distributed Spark materialization pipeline is still incomplete in this branch.'
        )

    @staticmethod
    def _drop_redundant_rdb_source_rows(rml_df):
        from ..constants import RDB, RML_QUERY, RML_SOURCE, RML_TABLE_NAME

        def _normalized_key_value(value):
            return None if pd.isna(value) else value

        source_rows = rml_df[
            (rml_df['source_type'] == RDB)
            & (rml_df['logical_source_type'] == RML_SOURCE)
        ]
        if source_rows.empty:
            return rml_df

        resolved_rows = rml_df[
            (rml_df['source_type'] == RDB)
            & (rml_df['logical_source_type'].isin([RML_QUERY, RML_TABLE_NAME]))
        ]
        if resolved_rows.empty:
            return rml_df

        key_columns = [
            'source_name',
            'triples_map_type',
            'subject_map_type',
            'subject_map_value',
            'subject_termtype',
            'predicate_map_type',
            'predicate_map_value',
            'object_map_type',
            'object_map_value',
            'object_termtype',
            'lang_datatype',
            'lang_datatype_map_type',
            'lang_datatype_map_value',
            'graph_map_type',
            'graph_map_value',
            'subject_join_conditions',
            'object_join_conditions',
        ]

        resolved_keys = {
            tuple(_normalized_key_value(row[column]) for column in key_columns)
            for _, row in resolved_rows.iterrows()
        }
        drop_index = [
            index
            for index, row in source_rows.iterrows()
            if tuple(_normalized_key_value(row[column]) for column in key_columns) in resolved_keys
        ]

        if not drop_index:
            return rml_df

        LOGGER.info(
            'Dropping %d redundant RDB `rml:source` mapping rows before Spark materialization.',
            len(drop_index),
        )
        return rml_df.drop(index=drop_index).reset_index(drop=True)

    def materialize_set(self, python_source=None):
        self._unsupported('materialize_set')

    def materialize_graph(self, python_source=None):
        self._unsupported('materialize')

    def materialize_oxigraph(self, python_source=None):
        self._unsupported('materialize_oxigraph')

    def materialize_kafka(self, python_source=None):
        self._unsupported('materialize_kafka')

    def materialize_to_files(self, python_source=None):
        spark = self._get_spark_session()
        try:
            from ..mapping.mapping_parser import retrieve_mappings
            from ..constants import RML_TRIPLES_MAP_CLASS
            from .spark_materializer import SparkMaterializer

            self._distribute_udf_dependencies(spark)
            rml_df, fnml_df, http_api_df = retrieve_mappings(self.config)
            rml_df = self._drop_redundant_rdb_source_rows(rml_df)
            self.config.set('CONFIGURATION', 'http_api_df', http_api_df.to_csv())
            asserted_mapping_df = rml_df.loc[rml_df['triples_map_type'] == RML_TRIPLES_MAP_CLASS]
            mapping_groups = [group for _, group in asserted_mapping_df.groupby(by='mapping_partition')]

            SparkMaterializer(spark, self.config, python_source=python_source).write_mapping_groups(
                mapping_groups,
                rml_df,
                fnml_df,
            )
            LOGGER.info('Spark materialization finished.')
        finally:
            spark.stop()
            self._cleanup_distributed_udf_dependencies()
