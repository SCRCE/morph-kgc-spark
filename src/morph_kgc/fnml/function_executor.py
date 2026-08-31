__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "Julián Arenas-Guerrero"
__email__ = "arenas.guerrero.julian@outlook.com"


import pandas as pd
import re
from ast import literal_eval

from ..constants import RML_CONSTANT
from ..constants import RML_EXECUTION
from ..constants import RML_REFERENCE
from ..constants import RML_TEMPLATE
from ..errors import SparkUnsupportedFeature
from ..utils import get_fnml_execution
from ..utils import get_references_in_template
from ..utils import remove_null_values_from_dataframe
from .function_registry import FunctionRegistry
from .function_registry import is_exploding_function


def _materialize_fnml_template(data, template):
    references = get_references_in_template(template)
    template = template.replace('\\{', '{').replace('\\}', '}')
    data['aux_fnml_template_data'] = ''

    for reference in references:
        data['reference_results'] = data[reference]
        splitted_template = template.split('{' + reference + '}')
        data['aux_fnml_template_data'] = data['aux_fnml_template_data'] + splitted_template[0] + data['reference_results']
        template = str('{' + reference + '}').join(splitted_template[1:])
    if template:
        data['aux_fnml_template_data'] = data['aux_fnml_template_data'] + template

    return data['aux_fnml_template_data']


class PandasFunctionExecutor:

    def __init__(self, config):
        self.config = config
        self.registry = FunctionRegistry(config)

    def execute(self, data: pd.DataFrame, fnml_df: pd.DataFrame, fnml_execution: str, in_recursion=False):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']

        for _, execution_rule in execution_rule_df.iterrows():
            if execution_rule['value_map_type'] == RML_EXECUTION:
                data = self.execute(data, fnml_df, execution_rule['value_map_value'], True)

        parameter_to_value_type_dict = dict(
            zip(execution_rule_df['parameter_map_value'], execution_rule_df['value_map_type'])
        )
        parameter_to_value_value_dict = dict(
            zip(execution_rule_df['parameter_map_value'], execution_rule_df['value_map_value'])
        )

        functions_with_array_parameter = fnml_df.groupby(
            ['function_execution', 'function_map_value', 'parameter_map_value']
        ).count()['value_map_value'].reset_index()
        if len(functions_with_array_parameter) > 0:
            function_params_df = functions_with_array_parameter[
                functions_with_array_parameter.function_execution == fnml_execution
            ]
            for parameter in function_params_df.parameter_map_value.to_list():
                selection = fnml_df[
                    (fnml_df.function_execution == fnml_execution) &
                    (fnml_df.parameter_map_value == parameter)
                ]
                parameter_to_value_value_dict[parameter] = selection.value_map_value.to_list()
                parameter_to_value_type_dict[parameter] = selection.value_map_type.to_list()

        registered_function = self.registry.get(function_id)
        function = registered_function.function
        function_decorator_parameters = registered_function.metadata.parameter_map

        function_param_array = []
        for function_parameter_name, function_parameter_value in function_decorator_parameters.items():
            if function_parameter_value in parameter_to_value_type_dict:
                for i in range(len(parameter_to_value_value_dict[function_parameter_value])):
                    parameter_dict = {}
                    value_type = parameter_to_value_type_dict[function_parameter_value][i]
                    value_ref = parameter_to_value_value_dict[function_parameter_value][i]
                    if value_type == RML_CONSTANT:
                        parameter_dict[function_parameter_name] = [value_ref] * len(data)
                    elif value_type == RML_TEMPLATE:
                        parameter_dict[function_parameter_name] = list(_materialize_fnml_template(data, value_ref))
                    else:
                        parameter_dict[function_parameter_name] = list(data[value_ref])
                    if i in range(len(function_param_array)):
                        function_param_array[i].update(parameter_dict)
                    else:
                        function_param_array.append(parameter_dict)

        function_params = {}
        if len(function_param_array) > 0:
            for key in function_param_array[0]:
                restructured_outer_array = []
                for j in range(len(function_param_array[0][key])):
                    restructured_inner_array = []
                    for i in range(len(function_param_array)):
                        restructured_inner_array.append(function_param_array[i][key][j])
                    restructured_outer_array.append(
                        restructured_inner_array if len(restructured_inner_array) > 1 else restructured_inner_array[0]
                    )
                function_params[key] = restructured_outer_array

        execution_results = []
        for i in range(len(data)):
            execution_params = {}
            for function_parameter_name, function_parameter_value in function_params.items():
                execution_params[function_parameter_name] = function_parameter_value[i]
            execution_results.append(function(**execution_params))

        data[fnml_execution] = execution_results
        data = remove_null_values_from_dataframe(data, self.config, fnml_execution, column=fnml_execution)

        if not in_recursion and is_exploding_function(function_id):
            data = data.explode(fnml_execution)
        elif not in_recursion:
            data = data.explode(fnml_execution)

        return data


class SparkFunctionExecutor:

    def __init__(self, config):
        self.config = config
        self.registry = FunctionRegistry(config)

    def is_native_supported(self, fnml_df: pd.DataFrame, fnml_execution: str):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)
        if registered_function.metadata.backend_strategy != 'spark-native':
            return False
        if registered_function.metadata.cardinality != 'scalar':
            return False

        if function_id == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_substring':
            for _, execution_rule in execution_rule_df.iterrows():
                if execution_rule['parameter_map_value'] in {
                    'http://users.ugent.be/~bjdmeest/function/grel.ttl#param_int_i_from',
                    'http://users.ugent.be/~bjdmeest/function/grel.ttl#param_int_i_opt_to',
                } and execution_rule['value_map_type'] != RML_CONSTANT:
                    return False

        for _, execution_rule in execution_rule_df.iterrows():
            if execution_rule['value_map_type'] == RML_EXECUTION:
                if not (
                    self.is_native_supported(fnml_df, execution_rule['value_map_value']) or
                    self.is_native_array_supported(fnml_df, execution_rule['value_map_value'])
                ):
                    return False
        return True

    def is_native_array_supported(self, fnml_df: pd.DataFrame, fnml_execution: str):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)
        if registered_function.metadata.backend_strategy != 'spark-native':
            return False
        if registered_function.metadata.cardinality != 'array':
            return False
        if function_id == 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode':
            return False

        if function_id == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split':
            separator_rows = execution_rule_df[
                execution_rule_df['parameter_map_value'] == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep'
            ]
            return len(separator_rows) == 1 and separator_rows.iloc[0]['value_map_type'] == RML_CONSTANT

        if function_id in {
            'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_sort',
            'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_reverse',
        }:
            selection = execution_rule_df[
                execution_rule_df['parameter_map_value'] == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a'
            ]
            if len(selection) != 1:
                return False
            value_type = selection.iloc[0]['value_map_type']
            if value_type != RML_EXECUTION:
                return False
            nested_execution = selection.iloc[0]['value_map_value']
            return self.is_native_array_supported(fnml_df, nested_execution)

        return False

    def is_native_exploding_supported(self, fnml_df: pd.DataFrame, fnml_execution: str):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)
        if registered_function.metadata.backend_strategy != 'spark-native':
            return False
        if registered_function.metadata.cardinality != 'array':
            return False
        if function_id != 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode':
            return False

        separator_rows = execution_rule_df[
            execution_rule_df['parameter_map_value'] == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep'
        ]
        if len(separator_rows) != 1:
            return False
        if separator_rows.iloc[0]['value_map_type'] != RML_CONSTANT:
            return False

        for _, execution_rule in execution_rule_df.iterrows():
            if execution_rule['value_map_type'] == RML_EXECUTION:
                nested_execution = execution_rule['value_map_value']
                if not (
                    self.is_native_supported(fnml_df, nested_execution) or
                    self.is_scalar_pandas_udf_supported(fnml_df, nested_execution) or
                    self.is_scalar_python_udf_supported(fnml_df, nested_execution)
                ):
                    return False
        return True

    def is_scalar_python_udf_supported(self, fnml_df: pd.DataFrame, fnml_execution: str):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)
        if registered_function.metadata.cardinality != 'scalar':
            return False

        for _, execution_rule in execution_rule_df.iterrows():
            if execution_rule['value_map_type'] == RML_EXECUTION:
                nested_execution = execution_rule['value_map_value']
                if not (
                    self.is_native_supported(fnml_df, nested_execution) or
                    self.is_batch_map_in_pandas_supported(fnml_df, nested_execution) or
                    self.is_scalar_pandas_udf_supported(fnml_df, nested_execution) or
                    self.is_scalar_python_udf_supported(fnml_df, nested_execution)
                ):
                    return False

        return True

    def is_scalar_pandas_udf_supported(self, fnml_df: pd.DataFrame, fnml_execution: str):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)
        if registered_function.metadata.cardinality != 'scalar':
            return False

        for _, execution_rule in execution_rule_df.iterrows():
            if execution_rule['value_map_type'] == RML_EXECUTION:
                nested_execution = execution_rule['value_map_value']
                if not (
                    self.is_native_supported(fnml_df, nested_execution) or
                    self.is_batch_map_in_pandas_supported(fnml_df, nested_execution) or
                    self.is_scalar_pandas_udf_supported(fnml_df, nested_execution)
                ):
                    return False

        return True

    def is_batch_map_in_pandas_supported(self, fnml_df: pd.DataFrame, fnml_execution: str):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)
        if registered_function.metadata.backend_strategy != 'spark-mapInPandas':
            return False

        for _, execution_rule in execution_rule_df.iterrows():
            if execution_rule['value_map_type'] == RML_EXECUTION:
                nested_execution = execution_rule['value_map_value']
                self._assert_registered_for_pandas_worker(fnml_df, nested_execution)

        return True

    def is_execution_deterministic(self, fnml_df: pd.DataFrame, fnml_execution: str):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)
        if not registered_function.metadata.deterministic:
            return False

        for _, execution_rule in execution_rule_df.iterrows():
            if execution_rule['value_map_type'] == RML_EXECUTION:
                if not self.is_execution_deterministic(fnml_df, execution_rule['value_map_value']):
                    return False

        return True

    def build_column(self, data, fnml_df: pd.DataFrame, fnml_execution: str, materializer):
        if self.is_native_supported(fnml_df, fnml_execution):
            return self.build_native_column(data, fnml_df, fnml_execution, materializer)
        if self.is_scalar_pandas_udf_supported(fnml_df, fnml_execution):
            try:
                return self.build_pandas_udf_column(data, fnml_df, fnml_execution, materializer)
            except (ImportError, ModuleNotFoundError, AttributeError, NotImplementedError):
                if self.is_scalar_python_udf_supported(fnml_df, fnml_execution):
                    return self.build_python_udf_column(data, fnml_df, fnml_execution, materializer)
                raise
        if self.is_scalar_python_udf_supported(fnml_df, fnml_execution):
            return self.build_python_udf_column(data, fnml_df, fnml_execution, materializer)

        self.assert_supported(fnml_df, fnml_execution)

    def get_execution_output_column(self, fnml_execution: str):
        sanitized_execution = re.sub(r'[^0-9A-Za-z_]+', '_', fnml_execution).strip('_')
        return f'__fnml_{sanitized_execution or "execution"}'

    def apply_execution(self, data, fnml_df: pd.DataFrame, fnml_execution: str, materializer, explode_batch_output=True):
        output_column = self.get_execution_output_column(fnml_execution)
        if output_column in data.columns:
            return data, output_column
        data = self._prepare_nested_batch_dependencies(data, fnml_df, fnml_execution, materializer)

        if self.is_native_exploding_supported(fnml_df, fnml_execution):
            from pyspark.sql import functions as sf

            data = data.withColumn(
                output_column,
                self.build_native_array_column(data, fnml_df, fnml_execution, materializer),
            )
            data = data.withColumn(output_column, sf.explode(sf.col(output_column)))
            return data, output_column
        if self.is_native_array_supported(fnml_df, fnml_execution):
            from pyspark.sql import functions as sf

            data = data.withColumn(
                output_column,
                self.build_native_array_column(data, fnml_df, fnml_execution, materializer),
            )
            if explode_batch_output:
                data = data.withColumn(output_column, sf.explode(sf.col(output_column)))
            return data, output_column
        if self.is_batch_map_in_pandas_supported(fnml_df, fnml_execution):
            data = self.apply_map_in_pandas_execution(
                data,
                fnml_df,
                fnml_execution,
                explode_output=explode_batch_output,
            )
            return data, output_column

        data = data.withColumn(output_column, self.build_column(data, fnml_df, fnml_execution, materializer))
        return data, output_column

    def _prepare_nested_batch_dependencies(self, data, fnml_df: pd.DataFrame, fnml_execution: str, materializer):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        for _, execution_rule in execution_rule_df.iterrows():
            if execution_rule['value_map_type'] != RML_EXECUTION:
                continue

            nested_execution = execution_rule['value_map_value']
            data = self._prepare_nested_batch_dependencies(data, fnml_df, nested_execution, materializer)
            if self.is_batch_map_in_pandas_supported(fnml_df, nested_execution):
                data, _ = self.apply_execution(
                    data,
                    fnml_df,
                    nested_execution,
                    materializer,
                    explode_batch_output=False,
                )

        return data

    def _assert_registered_for_pandas_worker(self, fnml_df: pd.DataFrame, fnml_execution: str):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        self.registry.get(function_id)

        for _, execution_rule in execution_rule_df.iterrows():
            if execution_rule['value_map_type'] == RML_EXECUTION:
                self._assert_registered_for_pandas_worker(fnml_df, execution_rule['value_map_value'])

    def apply_map_in_pandas_execution(self, data, fnml_df: pd.DataFrame, fnml_execution: str, explode_output=True):
        from pyspark.sql.types import StringType
        from pyspark.sql.types import StructField
        from pyspark.sql.types import StructType

        if not self.is_batch_map_in_pandas_supported(fnml_df, fnml_execution):
            self.assert_supported(fnml_df, fnml_execution)

        output_column = self.get_execution_output_column(fnml_execution)
        base_columns = list(data.columns)
        schema = StructType(list(data.schema.fields) + [StructField(output_column, StringType(), True)])
        config = self.config
        fnml_df_copy = fnml_df.copy()

        def batch_mapper(batch_iterator):
            executor = PandasFunctionExecutor(config)
            for batch in batch_iterator:
                result = executor.execute(
                    batch.copy(),
                    fnml_df_copy,
                    fnml_execution,
                    in_recursion=not explode_output,
                )
                result[fnml_execution] = result[fnml_execution].apply(
                    lambda value: None if value is None else str(value)
                )
                for column_name in base_columns:
                    if column_name not in result.columns:
                        result[column_name] = None
                yield result[base_columns + [fnml_execution]].rename(
                    columns={fnml_execution: output_column}
                )

        return data.mapInPandas(batch_mapper, schema=schema)

    def build_native_column(self, data, fnml_df: pd.DataFrame, fnml_execution: str, materializer):
        from pyspark.sql import functions as sf

        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)

        if not self.is_native_supported(fnml_df, fnml_execution):
            self.assert_supported(fnml_df, fnml_execution)

        bound_parameters = {}
        for function_parameter_name, function_parameter_value in registered_function.metadata.parameter_map.items():
            selection = execution_rule_df[
                execution_rule_df['parameter_map_value'] == function_parameter_value
            ]
            if len(selection) == 0:
                continue
            if len(selection) > 1:
                raise SparkUnsupportedFeature(
                    feature=fnml_execution,
                    reason=(
                        f'FNML function `{function_id}` uses repeated parameter '
                        f'`{function_parameter_value}`, which is not yet implemented for native Spark execution.'
                    ),
                )
            execution_rule = selection.iloc[0]
            value_type = execution_rule['value_map_type']
            value_ref = execution_rule['value_map_value']
            if value_type == RML_CONSTANT:
                bound_parameters[function_parameter_name] = sf.lit(value_ref)
            elif value_type == RML_TEMPLATE:
                bound_parameters[function_parameter_name] = materializer._materialize_template(
                    data,
                    value_ref,
                    value_type,
                    '',
                    '',
                    column_prefix='',
                    fnml_df=fnml_df,
                )
            elif value_type == RML_REFERENCE:
                bound_parameters[function_parameter_name] = materializer._column(value_ref)
            elif value_type == RML_EXECUTION:
                if self.is_native_array_supported(fnml_df, value_ref):
                    bound_parameters[function_parameter_name] = self.build_native_array_column(
                        data,
                        fnml_df,
                        value_ref,
                        materializer,
                    )
                else:
                    bound_parameters[function_parameter_name] = self.build_native_column(
                        data,
                        fnml_df,
                        value_ref,
                        materializer,
                    )
            else:
                raise SparkUnsupportedFeature(
                    feature=fnml_execution,
                    reason=f'Unsupported FNML value map type `{value_type}` for native Spark execution.',
                )

        if function_id == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#toUpperCase':
            return sf.upper(bound_parameters['string'])
        if function_id == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#toLowerCase':
            return sf.lower(bound_parameters['string'])
        if function_id == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_replace':
            return sf.replace(
                bound_parameters['string'],
                bound_parameters['old_substring'],
                bound_parameters['new_substring'],
            )
        if function_id == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join':
            separator_value = execution_rule_df[
                execution_rule_df['parameter_map_value'] == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep'
            ].iloc[0]['value_map_value']
            return sf.array_join(bound_parameters['array'], separator_value)
        if function_id == 'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#concat':
            separator = bound_parameters.get('separator', sf.lit(''))
            return sf.concat(bound_parameters['string1'], separator, bound_parameters['string2'])
        if function_id == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_substring':
            start_row = execution_rule_df[
                execution_rule_df['parameter_map_value'] == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#param_int_i_from'
            ].iloc[0]
            end_selection = execution_rule_df[
                execution_rule_df['parameter_map_value'] == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#param_int_i_opt_to'
            ]
            start_index = int(start_row['value_map_value'])
            if len(end_selection) == 0:
                return sf.substring(bound_parameters['string'], start_index + 1, 2147483647)
            end_index = int(end_selection.iloc[0]['value_map_value'])
            substring_length = end_index - start_index
            return sf.substring(bound_parameters['string'], start_index + 1, substring_length)

        self.assert_supported(fnml_df, fnml_execution)

    def build_native_array_column(self, data, fnml_df: pd.DataFrame, fnml_execution: str, materializer):
        from pyspark.sql import functions as sf

        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']

        if not (
            self.is_native_exploding_supported(fnml_df, fnml_execution) or
            self.is_native_array_supported(fnml_df, fnml_execution)
        ):
            self.assert_supported(fnml_df, fnml_execution)

        if function_id in {
            'https://github.com/morph-kgc/morph-kgc/function/built-in.ttl#string_split_explode',
            'http://users.ugent.be/~bjdmeest/function/grel.ttl#string_split',
        }:
            selection = execution_rule_df[
                execution_rule_df['parameter_map_value'] == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter'
            ]
            if len(selection) != 1:
                raise SparkUnsupportedFeature(
                    feature=fnml_execution,
                    reason=f'`{function_id}` requires exactly one value parameter in the Spark backend.',
                )
            execution_rule = selection.iloc[0]
            if execution_rule['value_map_type'] == RML_REFERENCE:
                string_column = materializer._column(execution_rule['value_map_value'])
            elif execution_rule['value_map_type'] == RML_CONSTANT:
                string_column = sf.lit(execution_rule['value_map_value'])
            elif execution_rule['value_map_type'] == RML_TEMPLATE:
                string_column = materializer._materialize_template(
                    data,
                    execution_rule['value_map_value'],
                    execution_rule['value_map_type'],
                    '',
                    '',
                    column_prefix='',
                    fnml_df=fnml_df,
                )
            elif execution_rule['value_map_type'] == RML_EXECUTION:
                nested_execution = execution_rule['value_map_value']
                if self.is_native_array_supported(fnml_df, nested_execution):
                    string_column = self.build_native_array_column(data, fnml_df, nested_execution, materializer)
                else:
                    string_column = self.build_column(data, fnml_df, nested_execution, materializer)
            else:
                raise SparkUnsupportedFeature(
                    feature=fnml_execution,
                    reason=f'`{function_id}` only supports constant, reference, template, or execution value parameters in the Spark backend.',
                )

            separator_value = execution_rule_df[
                execution_rule_df['parameter_map_value'] == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep'
            ].iloc[0]['value_map_value']
            return sf.split(string_column, re.escape(separator_value))

        if function_id in {
            'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_sort',
            'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_reverse',
        }:
            selection = execution_rule_df[
                execution_rule_df['parameter_map_value'] == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a'
            ]
            nested_execution = selection.iloc[0]['value_map_value']
            array_column = self.build_native_array_column(data, fnml_df, nested_execution, materializer)
            if function_id == 'http://users.ugent.be/~bjdmeest/function/grel.ttl#array_sort':
                return sf.array_sort(array_column)
            return sf.reverse(array_column)

        self.assert_supported(fnml_df, fnml_execution)

    def build_pandas_udf_column(self, data, fnml_df: pd.DataFrame, fnml_execution: str, materializer):
        from pyspark.sql import functions as sf
        from pyspark.sql.types import StringType

        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)

        if not self.is_scalar_pandas_udf_supported(fnml_df, fnml_execution):
            self.assert_supported(fnml_df, fnml_execution)

        positional_arguments = []
        provided_parameter_names = []
        argument_normalizers = []
        for function_parameter_name, function_parameter_value in registered_function.metadata.parameter_map.items():
            selection = execution_rule_df[
                execution_rule_df['parameter_map_value'] == function_parameter_value
            ]
            if len(selection) == 0:
                continue
            positional_argument, argument_normalizer = self._build_scalar_udf_parameter_binding(
                data,
                fnml_df,
                selection,
                materializer,
                fnml_execution,
                function_id,
                strategy_name='scalar Spark pandas UDF',
            )
            positional_arguments.append(positional_argument)
            argument_normalizers.append(argument_normalizer)
            provided_parameter_names.append(function_parameter_name)

        function = registered_function.function
        if not positional_arguments:
            positional_arguments = [sf.lit('')]

        @sf.pandas_udf(StringType())
        def vectorized_udf(*args):
            batch_size = len(args[0]) if args else 0
            results = []
            for row_index in range(batch_size):
                kwargs = {}
                for parameter_name, series, normalizer in zip(provided_parameter_names, args, argument_normalizers):
                    kwargs[parameter_name] = normalizer(series.iloc[row_index])
                result = function(**kwargs)
                results.append(None if result is None else str(result))
            return pd.Series(results)

        return vectorized_udf(*positional_arguments)

    def build_python_udf_column(self, data, fnml_df: pd.DataFrame, fnml_execution: str, materializer):
        from pyspark.sql import functions as sf
        from pyspark.sql.types import StringType

        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)

        if not self.is_scalar_python_udf_supported(fnml_df, fnml_execution):
            self.assert_supported(fnml_df, fnml_execution)

        positional_arguments = []
        provided_parameter_names = []
        argument_normalizers = []
        for function_parameter_name, function_parameter_value in registered_function.metadata.parameter_map.items():
            selection = execution_rule_df[
                execution_rule_df['parameter_map_value'] == function_parameter_value
            ]
            if len(selection) == 0:
                continue
            positional_argument, argument_normalizer = self._build_scalar_udf_parameter_binding(
                data,
                fnml_df,
                selection,
                materializer,
                fnml_execution,
                function_id,
                strategy_name='scalar Spark Python UDF',
            )
            positional_arguments.append(positional_argument)
            argument_normalizers.append(argument_normalizer)
            provided_parameter_names.append(function_parameter_name)

        function = registered_function.function

        def udf_wrapper(*args):
            kwargs = {}
            for parameter_name, value, normalizer in zip(provided_parameter_names, args, argument_normalizers):
                kwargs[parameter_name] = normalizer(value)
            result = function(**kwargs)
            return None if result is None else str(result)

        spark_udf = sf.udf(udf_wrapper, StringType())
        return spark_udf(*positional_arguments)

    def _build_scalar_udf_parameter_binding(
        self,
        data,
        fnml_df: pd.DataFrame,
        selection: pd.DataFrame,
        materializer,
        fnml_execution: str,
        function_id: str,
        strategy_name: str,
    ):
        from pyspark.sql import functions as sf

        expressions = []
        normalizers = []

        for _, execution_rule in selection.iterrows():
            value_type = execution_rule['value_map_type']
            value_ref = execution_rule['value_map_value']
            if value_type == RML_CONSTANT:
                expressions.append(sf.lit(value_ref))
                normalizers.append(_normalize_scalar_value)
            elif value_type == RML_TEMPLATE:
                expressions.append(materializer._materialize_template(
                    data,
                    value_ref,
                    value_type,
                    '',
                    '',
                    column_prefix='',
                    fnml_df=fnml_df,
                ))
                normalizers.append(_normalize_scalar_value)
            elif value_type == RML_REFERENCE:
                expressions.append(materializer._column(value_ref))
                normalizers.append(_normalize_scalar_value)
            elif value_type == RML_EXECUTION:
                nested_output_column = self.get_execution_output_column(value_ref)
                nested_registered_function = self.registry.get(
                    get_fnml_execution(fnml_df, value_ref).iloc[0]['function_map_value']
                )
                if nested_output_column in data.columns:
                    expressions.append(materializer._column(nested_output_column))
                else:
                    expressions.append(self.build_column(data, fnml_df, value_ref, materializer))
                if nested_registered_function.metadata.cardinality == 'array':
                    normalizers.append(_normalize_array_like_value)
                else:
                    normalizers.append(_normalize_scalar_value)
            else:
                raise SparkUnsupportedFeature(
                    feature=fnml_execution,
                    reason=f'Unsupported FNML value map type `{value_type}` for {strategy_name} execution.',
                )

        if len(expressions) == 1:
            return expressions[0], normalizers[0]

        if any(normalizer is _normalize_array_like_value for normalizer in normalizers):
            raise SparkUnsupportedFeature(
                feature=fnml_execution,
                reason=(
                    f'FNML function `{function_id}` uses repeated parameter bindings mixed with nested array-valued '
                    f'executions, which is not implemented for the {strategy_name} path.'
                ),
            )

        return sf.array(*expressions), _normalize_array_like_value

    def assert_supported(self, fnml_df: pd.DataFrame, fnml_execution: str, triples_map_id=None, position=None):
        execution_rule_df = get_fnml_execution(fnml_df, fnml_execution)
        function_id = execution_rule_df.iloc[0]['function_map_value']
        registered_function = self.registry.get(function_id)

        if self.is_native_supported(fnml_df, fnml_execution):
            return
        if self.is_native_exploding_supported(fnml_df, fnml_execution):
            return
        if self.is_native_array_supported(fnml_df, fnml_execution):
            return
        if self.is_batch_map_in_pandas_supported(fnml_df, fnml_execution):
            return
        if self.is_scalar_pandas_udf_supported(fnml_df, fnml_execution):
            return
        if self.is_scalar_python_udf_supported(fnml_df, fnml_execution):
            return

        raise SparkUnsupportedFeature(
            feature=position or fnml_execution,
            reason=(
                f'FNML function `{registered_function.metadata.function_id}` '
                f'(execution `{fnml_execution}`'
                + (f', triples map `{triples_map_id}`' if triples_map_id else '')
                + ') is not implemented in the Spark backend yet. '
                f'Registry strategy is `{registered_function.metadata.backend_strategy}` '
                f'with supported backends {registered_function.metadata.supported_backends}.'
            ),
        )


def _normalize_scalar_value(value):
    if value is None:
        return None
    if isinstance(value, (list, dict, tuple, set)):
        return value
    if hasattr(value, 'tolist') and not isinstance(value, (str, bytes)):
        try:
            converted_value = value.tolist()
        except TypeError:
            converted_value = value
        else:
            if isinstance(converted_value, list):
                return converted_value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value


def _normalize_array_like_value(value):
    value = _normalize_scalar_value(value)
    if not isinstance(value, str):
        return value

    try:
        parsed_value = literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    if isinstance(parsed_value, (list, tuple)):
        return list(parsed_value)
    return parsed_value
