# Apache Spark Backend

Morph-KGC includes an optional Apache Spark backend for RDF materialization workloads that benefit from parallel source reads, distributed joins, distributed deduplication, and partitioned execution.

Pandas remains the default and reference engine. Existing configurations continue to use pandas unless Spark is selected explicitly.

## Installation

Install Morph-KGC Spark from its release tag with the optional Spark dependency:

```bash
pip install 'morph-kgc[spark] @ git+https://github.com/SCRCE/morph-kgc-spark.git@v2.10.0-spark.1'
```

A compatible Java runtime is also required. PySpark is imported only when Spark execution is selected. If it is unavailable, Morph-KGC raises a dependency error with installation guidance.

## Configuration

Set the execution engine in the main configuration section:

```ini
[CONFIGURATION]
execution_engine=spark
output_file=knowledge-graph.nt
output_format=N-TRIPLES
mapping_partitioning=PARTIAL-AGGREGATIONS

[DataSource1]
mappings=/path/to/mapping.rml.ttl
```

Run the normal Morph-KGC command:

```bash
morph_kgc config.ini
```

Use `execution_engine=pandas` to select the reference backend explicitly.

## Execution Model

Mapping parsing and planning remain shared with the pandas backend. For supported workloads, Spark DataFrames handle source records and generated RDF lines.

The backend uses:

- Spark readers for supported tabular sources;
- Spark SQL expressions for constants, references, templates, RDF terms, and native functions;
- distributed DataFrame joins for supported parent and quoted-triples mappings;
- `dropDuplicates()` for distributed RDF deduplication;
- Spark text output followed by streaming assembly into the configured `.nt` or `.nq` file.

Morph-KGC does not collect the complete RDF output into a Python set, convert it to pandas, or call full DataFrame `collect()` during supported file materialization.

Spark mode ignores `number_of_processes` values above one. Spark executors and partitions provide parallelism, and Morph-KGC does not start nested Python process pools inside Spark tasks.

## Supported Sources

| Source | Spark strategy |
| --- | --- |
| CSV and TSV | Native Spark reader |
| Parquet | Native Spark reader |
| ORC | Native Spark reader |
| JSON and XML | Shared local preprocessing to temporary Parquet, followed by Spark materialization |
| RML tabular views | Local DuckDB preprocessing to temporary Parquet, followed by Spark materialization |
| Relational databases | Compatibility mode uses local SQLAlchemy/pandas preprocessing; PostgreSQL can opt into experimental partitioned JDBC reads |
| In-memory Python data | Supported for file output when `python_source` is supplied directly |

Excel, Feather, ODS, Stata, GeoParquet, HTTP API, and property-graph sources are not supported by the Spark backend. Use pandas for those sources.

## Supported Mapping Features

The supported surface includes:

- constant, reference, and template term maps;
- IRI, literal, and blank-node terms;
- datatype and language maps;
- graph maps with N-Quads output;
- explicit parent-triples object joins, including multi-column joins;
- same-source parent-triples maps for supported no-join shapes;
- the covered RML-star quoted-triples mappings and joins;
- distributed duplicate elimination;
- deterministic literal escaping and IRI encoding compatible with pandas output.

The following remain unsupported in Spark mode:

- non-quoted subject join conditions;
- non-parent object join conditions;
- cross-source parent-triples maps without explicit join conditions;
- nested quoted-triples subject or object joins outside the supported shapes;
- APIs that return the full graph as a Python set, RDFLib graph, or Oxigraph store;
- Kafka materialization.

Unsupported mappings raise `SparkUnsupportedFeature` with the unsupported feature and a recommendation to use `execution_engine=pandas`. There is no silent fallback.

## FNML and Python UDFs

Spark function execution uses the following priority:

1. Native Spark SQL expressions for supported built-ins.
2. Vectorized pandas UDFs for scalar Python functions.
3. `mapInPandas` for selected array, list, and row-expanding functions.
4. A typed plain Python UDF when vectorized execution is not available.
5. An explicit unsupported error when a function cannot be distributed safely or matched to pandas behavior.

User UDF metadata can declare `output_type`, `cardinality`, `null_policy`, `deterministic`, `supported_backends`, and `backend_strategy`. Configured UDF files are distributed with `SparkContext.addPyFile()` and cached by worker registries instead of being imported for every row.

Functions that generate independent runtime values, such as current time, random numbers, or UUIDs, are classified as nondeterministic. Their output is tested semantically rather than by exact line comparison between independent pandas and Spark runs.

## PostgreSQL JDBC Mode

Relational sources use compatibility preprocessing by default:

```ini
spark_rdb_mode=local_preprocess
```

PostgreSQL sources can opt into experimental partitioned Spark JDBC loading:

```ini
[Database]
mappings=/path/to/mapping.ttl
db_url=postgresql://user:password@localhost/database
spark_rdb_mode=jdbc
spark_jdbc_partition_column=id
spark_jdbc_lower_bound=0
spark_jdbc_upper_bound=29999999
spark_jdbc_num_partitions=28
spark_jdbc_fetch_size=10000
```

The partition column must be an integral, date, or timestamp column. Bounds and partition count are required and validated before execution.

Provide the PostgreSQL JDBC driver with `SPARK_JDBC_JAR`:

```bash
export SPARK_JDBC_JAR=/path/to/postgresql.jar
morph_kgc config.ini
```

The driver can also be supplied through normal Spark package configuration. JDBC mode currently supports PostgreSQL only.

## Output and Parity

Spark supports N-Triples and N-Quads file materialization. Spark writes distributed text parts and Morph-KGC streams those parts into the configured output file without holding the graph in driver memory.

Spark does not guarantee RDF line order. To compare deterministic pandas and Spark output:

1. normalize line endings;
2. preserve all literal whitespace;
3. sort complete RDF lines;
4. compare statement counts and canonical hashes.

Raw file hashes are useful for serialization diagnostics but are not semantic graph hashes.

## Runtime Configuration

Spark uses the active Spark environment. For local execution, normal PySpark settings can define the master, memory, shuffle partitions, and scratch location:

```bash
export PYSPARK_PYTHON=/path/to/python
export PYSPARK_DRIVER_PYTHON=/path/to/python
export JAVA_HOME=/path/to/java
export PYSPARK_SUBMIT_ARGS='--master local[16] --driver-memory 16g --conf spark.sql.shuffle.partitions=16 pyspark-shell'

morph_kgc config.ini
```

For large jobs:

- use fast local or distributed storage for sources, scratch, and output;
- reserve disk space for temporary Parquet, shuffle spill, Spark text parts, and final output;
- tune shuffle and JDBC partition counts to the workload and available cores;
- install the same Python dependencies on all workers;
- monitor spill, garbage collection, and final output assembly time.

Spark startup and scheduling overhead can make pandas faster for small workloads. Source types that require local preprocessing also have a lower potential speedup. Benchmark representative mappings and data before selecting a production backend.

## Example

See [`examples/configuration-file/spark_config.ini`](../examples/configuration-file/spark_config.ini) for a minimal configuration.
