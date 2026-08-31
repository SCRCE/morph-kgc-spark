__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero", "Ahmad Hammad"]
__copyright__ = "Copyright © 2020 Julián Arenas-Guerrero"

__license__ = "Apache-2.0"
__maintainer__ = "Ahmad Hammad"
__email__ = "Ahmad.Hammad@ieee.org"


import sys
import time
import logging
import multiprocessing as mp

from io import BytesIO
from itertools import repeat

from ..constants import JELLY, LOGGING_NAMESPACE
from .base import ExecutionBackend


LOGGER = logging.getLogger(LOGGING_NAMESPACE)


class PandasExecutionBackend(ExecutionBackend):

    def _get_mapping_groups(self):
        from ..constants import RML_TRIPLES_MAP_CLASS
        from ..mapping.mapping_parser import retrieve_mappings

        rml_df, fnml_df, http_api_df = retrieve_mappings(self.config)
        self.config.set('CONFIGURATION', 'http_api_df', http_api_df.to_csv())

        asserted_mapping_df = rml_df.loc[rml_df['triples_map_type'] == RML_TRIPLES_MAP_CLASS]
        mapping_groups = [group for _, group in asserted_mapping_df.groupby(by='mapping_partition')]

        return rml_df, fnml_df, mapping_groups

    def materialize_set(self, python_source=None):
        from ..materializer import _materialize_mapping_group_to_set

        if 'linux' not in sys.platform:
            LOGGER.info(
                f'Parallelization is not supported for {sys.platform} when running as a library. '
                f'If you need to speed up your data integration pipeline, please run through the command line.'
            )
            self.config.set_number_of_processes('1')

        rml_df, fnml_df, mapping_groups = self._get_mapping_groups()

        if self.config.is_multiprocessing_enabled():
            LOGGER.debug(f'Parallelizing with {self.config.get_number_of_processes()} cores.')

            pool = mp.Pool(self.config.get_number_of_processes())
            triples = set().union(*pool.starmap(
                _materialize_mapping_group_to_set,
                zip(mapping_groups, repeat(rml_df), repeat(fnml_df), repeat(self.config), repeat(python_source))
            ))
            pool.close()
            pool.join()
        else:
            triples = set()
            for mapping_group in mapping_groups:
                triples.update(_materialize_mapping_group_to_set(mapping_group, rml_df, fnml_df, self.config,
                                                                python_source))

        LOGGER.info(f'Number of triples generated in total: {len(triples)}.')
        return triples

    def materialize_graph(self, python_source=None):
        from rdflib import Graph

        triples = self.materialize_set(python_source)

        graph = Graph()
        if triples:
            rdf_ntriples = '.\n'.join(triples) + '.'
            graph.parse(data=rdf_ntriples, format='nquads')

        return graph

    def materialize_oxigraph(self, python_source=None):
        from pyoxigraph import Store

        triples = self.materialize_set(python_source)

        graph = Store()
        if triples:
            rdf_ntriples = '.\n'.join(triples) + '.'
            graph.bulk_load(BytesIO(rdf_ntriples.encode()), 'application/n-quads')

        return graph

    def materialize_kafka(self, python_source=None):
        from kafka import KafkaProducer

        kafka_producer = None

        try:
            triples = self.materialize_set(python_source)
            output_kafka_server = self.config.get_output_kafka_server()
            output_kafka_topic = self.config.get_output_kafka_topic()

            if not output_kafka_server or not output_kafka_topic:
                LOGGER.error('Output Kafka server or topic is empty.')
                sys.exit()

            kafka_producer = KafkaProducer(bootstrap_servers=output_kafka_server)

            if triples:
                rdf_ntriples = '.\n'.join(triples) + '.'
                kafka_producer.send(output_kafka_topic, value=rdf_ntriples.encode('utf-8'))

            LOGGER.info(f'RDF triples materialized and sent to Kafka topic: {output_kafka_topic}.')
        except Exception as e:
            LOGGER.error(f'Error during materialization or Kafka publishing: {e}')
        finally:
            if kafka_producer:
                kafka_producer.close()

    def materialize_to_files(self, python_source=None):
        from ..materializer import _materialize_mapping_group_to_file
        from ..materializer import _materialize_mapping_group_to_kafka
        from ..utils import get_delta_time
        from ..utils import prepare_output_files

        if self.config.get_output_format() == JELLY:
            try:
                import pyjelly
            except ImportError as e:
                raise RuntimeError(
                    "JELLY output requested, but pyjelly[rdflib] is not installed. "
                    "Install: pip install 'morph-kgc[jelly]'"
                ) from e

            graph = self.materialize_graph(python_source)
            output_path = self.config.get_output_file_path(None)
            from ..utils import create_dirs_in_path
            create_dirs_in_path(output_path)
            graph.serialize(destination=output_path, format='jelly')

            LOGGER.info(f'Jelly file generated: {output_path}')
            LOGGER.info('Materialization finished.')
            return

        rml_df, fnml_df, mapping_groups = self._get_mapping_groups()
        prepare_output_files(self.config, rml_df)

        start_time = time.time()
        num_triples = 0
        if self.config.is_multiprocessing_enabled():
            LOGGER.debug(f'Parallelizing with {self.config.get_number_of_processes()} cores.')

            pool = mp.Pool(self.config.get_number_of_processes())
            if not self.config.get_output_kafka_server():
                num_triples = sum(pool.starmap(
                    _materialize_mapping_group_to_file,
                    zip(mapping_groups, repeat(rml_df), repeat(fnml_df), repeat(self.config), repeat(python_source))
                ))
            else:
                num_triples = sum(pool.starmap(
                    _materialize_mapping_group_to_kafka,
                    zip(mapping_groups, repeat(rml_df), repeat(fnml_df), repeat(self.config), repeat(python_source))
                ))
            pool.close()
            pool.join()
        else:
            for mapping_group in mapping_groups:
                if not self.config.get_output_kafka_server():
                    num_triples += _materialize_mapping_group_to_file(
                        mapping_group,
                        rml_df,
                        fnml_df,
                        self.config,
                        python_source,
                    )
                else:
                    num_triples += _materialize_mapping_group_to_kafka(
                        mapping_group,
                        rml_df,
                        fnml_df,
                        self.config,
                        python_source,
                    )

        LOGGER.info(f'Number of triples generated in total: {num_triples}.')
        LOGGER.info(f'Materialization finished in {get_delta_time(start_time)} seconds.')
