__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "Julián Arenas-Guerrero"
__email__ = "arenas.guerrero.julian@outlook.com"


import pandas as pd

from .function_executor import PandasFunctionExecutor
from .function_registry import FunctionRegistry


def execute_fnml(data:pd.DataFrame, fnml_df: pd.DataFrame, fnml_execution:dict, config, in_recursion=False):
    """
    Executes an FNML (Function-based Mapping Language) transformation on the provided data.
    Args:
        data (pd.DataFrame): The input data to be transformed.
        fnml_df (pd.DataFrame): The FNML mapping definitions as a DataFrame.
        fnml_execution (dict): The execution context (an id) for the FNML transformation.
        config: Configuration object containing settings and parameters for the execution.
        in_recursion (bool, optional): Indicates whether the function is being called recursively. Defaults to False.
    Returns:
        pd.DataFrame: The transformed data after applying the FNML mappings and functions.
    Notes:
        - Handles composite functions by recursively calling itself for nested executions.
        - Supports functions with multiple parameters that need to be aggregated into arrays.
        - Dynamically loads user-defined functions (UDFs) if the function ID is not a built-in function.
        - Prepares function parameters based on their mapping type (e.g., constant, template, reference, or execution).
        - Executes the specified function for each row of the input data.
        - Removes null values from the resulting DataFrame and optionally explodes list values for outer functions.
    Raises:
        KeyError: If a required function or parameter is not found in the mappings or configuration.
        Exception: If an error occurs during function execution or data transformation.
    """
    executor = PandasFunctionExecutor(config)
    return executor.execute(data, fnml_df, fnml_execution, in_recursion=in_recursion)


def load_udfs(config):
    registry = FunctionRegistry(config)
    return {
        function_id: {
            'function': registered_function.function,
            'parameters': registered_function.metadata.parameter_map,
        }
        for function_id, registered_function in registry._functions.items()
        if registered_function.metadata.source == 'udf'
    }
