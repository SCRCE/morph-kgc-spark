__author__ = "Ahmad Hammad"
__credits__ = ["Ahmad Hammad"]

__license__ = "Apache-2.0"
__maintainer__ = "Ahmad Hammad"
__email__ = "Ahmad.Hammad@ieee.org"


from urllib.parse import quote, urlencode

POSTGRESQL_JDBC_DRIVER = 'org.postgresql.Driver'


def postgresql_jdbc_options(db_url):
    """Translate a SQLAlchemy PostgreSQL URL into Spark JDBC connection options."""
    from sqlalchemy.engine import make_url

    parsed = make_url(db_url)
    if parsed.get_backend_name() != 'postgresql':
        raise ValueError(
            f'Spark JDBC prototype currently supports PostgreSQL only, not `{parsed.get_backend_name()}`. '
            'Use `spark_rdb_mode = local_preprocess` for this source.'
        )
    if not parsed.host or not parsed.database:
        raise ValueError('Spark JDBC requires a PostgreSQL URL with an explicit host and database name.')

    jdbc_url = (
        f'jdbc:postgresql://{parsed.host}:{parsed.port or 5432}/'
        f'{quote(parsed.database, safe="")}'
    )
    if parsed.query:
        jdbc_url = f'{jdbc_url}?{urlencode(dict(parsed.query))}'

    options = {
        'url': jdbc_url,
        'driver': POSTGRESQL_JDBC_DRIVER,
    }
    if parsed.username is not None:
        options['user'] = parsed.username
    if parsed.password is not None:
        options['password'] = parsed.password
    return options
