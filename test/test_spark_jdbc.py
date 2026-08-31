import pytest

from morph_kgc.args_parser import load_config_from_argument
from morph_kgc.engine.spark_jdbc import postgresql_jdbc_options


def test_spark_rdb_mode_defaults_to_local_preprocess():
    config = load_config_from_argument('[DataSource]\nmappings=/tmp/mapping.ttl\ndb_url=sqlite:///test.db\n')

    assert config.get_spark_rdb_mode('DataSource') == 'local_preprocess'


def test_spark_jdbc_requires_complete_partition_configuration():
    with pytest.raises(ValueError, match='spark_jdbc_lower_bound'):
        load_config_from_argument(
            '[DataSource]\n'
            'mappings=/tmp/mapping.ttl\n'
            'db_url=postgresql://user:password@localhost/database\n'
            'spark_rdb_mode=jdbc\n'
            'spark_jdbc_partition_column=id\n'
        )


def test_spark_jdbc_rejects_invalid_bounds():
    with pytest.raises(ValueError, match='must be less'):
        load_config_from_argument(
            '[DataSource]\n'
            'mappings=/tmp/mapping.ttl\n'
            'db_url=postgresql://user:password@localhost/database\n'
            'spark_rdb_mode=jdbc\n'
            'spark_jdbc_partition_column=id\n'
            'spark_jdbc_lower_bound=10\n'
            'spark_jdbc_upper_bound=10\n'
            'spark_jdbc_num_partitions=4\n'
        )


def test_spark_jdbc_accepts_iso_timestamp_bounds():
    config = load_config_from_argument(
        '[DataSource]\n'
        'mappings=/tmp/mapping.ttl\n'
        'db_url=postgresql://user:password@localhost/database\n'
        'spark_rdb_mode=jdbc\n'
        'spark_jdbc_partition_column=event_ts\n'
        'spark_jdbc_lower_bound=2020-01-01T00:00:00\n'
        'spark_jdbc_upper_bound=2021-01-01T00:00:00\n'
        'spark_jdbc_num_partitions=4\n'
    )

    assert config.get_spark_jdbc_lower_bound('DataSource') == '2020-01-01T00:00:00'


def test_postgresql_url_is_translated_to_jdbc_options():
    options = postgresql_jdbc_options('postgresql+psycopg://user:p%40ss@db.example:5544/benchmark')

    assert options == {
        'url': 'jdbc:postgresql://db.example:5544/benchmark',
        'driver': 'org.postgresql.Driver',
        'user': 'user',
        'password': 'p@ss',
    }


def test_non_postgresql_jdbc_source_is_explicitly_rejected():
    with pytest.raises(ValueError, match='PostgreSQL only'):
        postgresql_jdbc_options('mysql://user:password@localhost/database')
