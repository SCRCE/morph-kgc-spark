__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero", "Miel Vander Sande"]

__license__ = "Apache-2.0"
__maintainer__ = "Julián Arenas-Guerrero"
__email__ = "arenas.guerrero.julian@outlook.com"


import json
import urllib.request
import xml.etree.ElementTree as et
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from urllib.parse import urlparse

import duckdb
import pandas as pd

from ..constants import *
from ..utils import normalize_hierarchical_data


def _is_http_uri(path_or_uri):
    parsed = urlparse(str(path_or_uri).strip())
    return parsed.scheme.lower() in {'http', 'https'} and bool(parsed.netloc)


def get_file_data(rml_rule, references):
    references = list(references)
    file_source_type = rml_rule['source_type']

    if rml_rule['logical_source_type'] == RML_QUERY:
        return _read_tabular_view(rml_rule)
    elif file_source_type in [CSV, TSV]:
        return _read_csv(rml_rule, references, file_source_type)
    elif file_source_type in EXCEL:
        return _read_excel(rml_rule, references)
    elif file_source_type in ODS:
        return _read_ods(rml_rule, references)
    elif file_source_type == PARQUET:
        return _read_parquet(rml_rule, references)
    elif file_source_type == GEOPARQUET:
        return _read_geoparquet(rml_rule, references)
    elif file_source_type == SHP:
        return _read_shapefile(rml_rule, references)
    elif file_source_type in FEATHER:
        return _read_feather(rml_rule, references)
    elif file_source_type == ORC:
        return _read_orc(rml_rule, references)
    elif file_source_type == STATA:
        return _read_stata(rml_rule, references)
    elif file_source_type in SAS:
        return _read_sas(rml_rule)
    elif file_source_type == SPSS:
        return _read_spss(rml_rule, references)
    elif file_source_type in JSON:
        return _read_json(rml_rule, references)
    elif file_source_type in XML:
        return _read_xml(rml_rule, references)
    else:
        raise ValueError(f'Found an invalid source type. Found value `{file_source_type}`.')


def _read_tabular_view(rml_rule):
    return duckdb.query(rml_rule['logical_source_value']).df()


def _read_csv(rml_rule, references, file_source_type):
    delimiter = ',' if file_source_type == 'CSV' else '\t'

    try:
        return pd.read_table(rml_rule['logical_source_value'],
                             sep=delimiter,
                             index_col=False,
                             encoding='utf-8',
                             encoding_errors='strict',
                             usecols=references,
                             engine='c',
                             dtype=str,
                             keep_default_na=False,
                             na_filter=False)
    except:
        # if delimiter is other than comma or tab, then infer it (issue #81)
        return pd.read_table(rml_rule['logical_source_value'],
                             index_col=False,
                             sep=None,
                             encoding='utf-8',
                             encoding_errors='strict',
                             usecols=references,
                             engine='python',
                             dtype=str,
                             keep_default_na=False,
                             na_filter=False)


def _read_parquet(rml_rule, references):
    return pd.read_parquet(rml_rule['logical_source_value'], engine='pyarrow', columns=references)


def _read_geoparquet(rml_rule, references) -> pd.DataFrame:
    import geopandas as gpd

    try:
        gdf = gpd.read_parquet(rml_rule['logical_source_value'], columns=references)
    except ValueError as e:
        if "No geometry columns are included" in str(e):
            return pd.read_parquet(rml_rule['logical_source_value'], engine='pyarrow', columns=references)
        raise e

    if isinstance(gdf, gpd.GeoDataFrame) and gdf.geometry.name in references:
        geometry = gdf.geometry
        df = gdf.drop(columns=geometry.name).pipe(pd.DataFrame)
        df[geometry.name] = geometry.to_wkt()
        return df

    return pd.DataFrame(gdf)


def _read_shapefile(rml_rule, references) -> pd.DataFrame:
    import geopandas as gpd

    gdf = gpd.read_file(rml_rule['logical_source_value'], ignore_geometry=False)

    if isinstance(gdf, gpd.GeoDataFrame) and gdf.geometry.name in references:
        geometry = gdf.geometry
        df = gdf.drop(columns=geometry.name).pipe(pd.DataFrame)
        df[geometry.name] = geometry.to_wkt()
        return df

    return pd.DataFrame(gdf)


def _read_feather(rml_rule, references):
    return pd.read_feather(rml_rule['logical_source_value'], use_threads=False, columns=references)


def _read_orc(rml_rule, references):
    return pd.read_orc(rml_rule['logical_source_value'], columns=references)


def _read_stata(rml_rule, references):
    return pd.read_stata(rml_rule['logical_source_value'],
                         columns=references,
                         convert_dates=False,
                         convert_categoricals=False,
                         convert_missing=False,
                         preserve_dtypes=False,
                         order_categoricals=False)


def _read_sas(rml_rule):
    return pd.read_sas(rml_rule['logical_source_value'], encoding='utf-8')


def _read_spss(rml_rule, references):
    return pd.read_spss(rml_rule['logical_source_value'], usecols=references, convert_categoricals=False)


def _read_excel(rml_rule, references):
    return pd.read_excel(rml_rule['logical_source_value'],
                         sheet_name=0,
                         engine='openpyxl',
                         usecols=references,
                         dtype=str,
                         keep_default_na=False,
                         na_filter=False)


def _read_ods(rml_rule, references):
    try:
        return pd.read_excel(rml_rule['logical_source_value'],
                             sheet_name=0,
                             engine='odf',
                             usecols=references,
                             dtype=str,
                             keep_default_na=False,
                             na_filter=False)
    except ImportError:
        return _read_ods_without_odfpy(rml_rule, references)


def _read_ods_without_odfpy(rml_rule, references):
    namespaces = {
        'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
        'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0',
        'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0',
    }

    with ZipFile(rml_rule['logical_source_value']) as ods_file:
        root = et.fromstring(ods_file.read('content.xml'))

    sheet = root.find('.//table:table', namespaces)
    if sheet is None:
        return pd.DataFrame(columns=references)

    rows = []
    for row in sheet.findall('table:table-row', namespaces):
        repeated_rows = int(row.attrib.get(f"{{{namespaces['table']}}}number-rows-repeated", '1'))
        row_values = []
        for cell in row.findall('table:table-cell', namespaces):
            repeated_columns = int(cell.attrib.get(f"{{{namespaces['table']}}}number-columns-repeated", '1'))
            cell_text = '\n'.join(''.join(paragraph.itertext()) for paragraph in cell.findall('text:p', namespaces))
            row_values.extend([cell_text] * repeated_columns)
        trimmed_values = list(row_values)
        while trimmed_values and trimmed_values[-1] == '':
            trimmed_values.pop()
        for _ in range(repeated_rows):
            rows.append(list(trimmed_values))

    if not rows:
        return pd.DataFrame(columns=references)

    headers = rows[0]
    if not headers:
        return pd.DataFrame(columns=references)

    projected_rows = []
    for values in rows[1:]:
        if not any(values):
            continue
        padded_values = values + [''] * max(0, len(headers) - len(values))
        projected_rows.append(dict(zip(headers, padded_values[:len(headers)])))

    ods_df = pd.DataFrame(projected_rows)
    missing_references = [reference for reference in references if reference not in ods_df.columns]
    if missing_references:
        ods_df[missing_references] = None
    return ods_df[references]


def _read_json(rml_rule, references):
    logical_source_value = rml_rule['logical_source_value'].strip()

    if _is_http_uri(logical_source_value):
        with urllib.request.urlopen(logical_source_value) as json_url:
            json_data = json.loads(json_url.read().decode())
    else:
        json_data = json.loads(Path(logical_source_value).read_bytes())

    try:
        from jsonpath import JSONPath

        jsonpath_expression = rml_rule['iterator'] + '.('
        # add top level object of the references to reduce intermediate results (THIS IS NOT STRICTLY NECESSARY)
        for reference in references:
            jsonpath_expression += reference.split('.')[0] + ','
        jsonpath_expression = jsonpath_expression[:-1] + ')'

        jsonpath_result = JSONPath(jsonpath_expression).parse(json_data)
        # normalize and remove nulls
        json_df = pd.json_normalize([
            json_object
            for json_object in normalize_hierarchical_data(jsonpath_result)
            if None not in json_object.values()
            and all(reference.split('.')[0] in json_object for reference in references)
        ])
    except ModuleNotFoundError:
        json_df = _read_json_without_jsonpath(json_data, rml_rule['iterator'], references)

    # add columns with null values for those references in the mapping rule that are not present in the data file
    missing_references_in_df = list(set(references).difference(set(json_df.columns)))
    json_df[missing_references_in_df] = None
    json_df.dropna(axis=0, how='any', inplace=True)

    return json_df


def _read_json_without_jsonpath(json_data, iterator, references):
    records = _apply_simple_json_iterator(json_data, iterator)
    normalized_records = []
    for record in records:
        projected_record = _project_json_record(record, references)
        normalized_records.extend(normalize_hierarchical_data(projected_record))

    json_df = pd.json_normalize(normalized_records) if normalized_records else pd.DataFrame()

    return json_df


def _apply_simple_json_iterator(json_data, iterator):
    iterator = iterator.strip()
    if iterator == '$':
        return [json_data]

    if not iterator.startswith('$'):
        raise ValueError('Simple JSON fallback only supports iterators that start with `$`.')

    current_nodes = [json_data]
    path_expression = iterator[1:]
    if path_expression.startswith('.'):
        path_expression = path_expression[1:]
    if not path_expression:
        return current_nodes

    for token in path_expression.split('.'):
        next_nodes = []
        for node in current_nodes:
            next_nodes.extend(_advance_simple_json_nodes(node, token))
        current_nodes = next_nodes

    return current_nodes


def _advance_simple_json_nodes(node, token):
    if token == '*':
        if isinstance(node, dict):
            return list(node.values())
        if isinstance(node, list):
            return list(node)
        return []

    if token.endswith('[*]'):
        key = token[:-3]
        targets = _advance_simple_json_nodes(node, key) if key else [node]
        expanded_targets = []
        for target in targets:
            if isinstance(target, list):
                expanded_targets.extend(target)
        return expanded_targets

    if isinstance(node, dict) and token in node:
        return [node[token]]

    return []


def _project_json_record(record, references):
    if not isinstance(record, dict):
        return record

    projected_record = {}
    for reference in references:
        top_level_key = reference.split('.')[0]
        if top_level_key == '*':
            continue
        if top_level_key in record:
            projected_record[top_level_key] = record[top_level_key]

    return projected_record


def _resolve_simple_json_reference(record, reference):
    current = record
    for part in reference.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _read_xml(rml_rule, references):
    logical_source_value = rml_rule['logical_source_value'].strip()

    if _is_http_uri(logical_source_value):
        with urllib.request.urlopen(logical_source_value) as xml_url:
            xml_string = xml_url.read()
        # Turn into file object for compatibility with iterparse
        with BytesIO(xml_string) as xml_file:
            return _parse_xml_file(xml_file, rml_rule, references)
    else:
        with Path(logical_source_value).open(encoding='utf-8') as xml_file:
            return _parse_xml_file(xml_file, rml_rule, references)


def _parse_xml_file(xml_file, rml_rule, references):
    try:
        import elementpath
        from elementpath.xpath3 import XPath3Parser

        # Collect namespaces from XML document
        namespaces = {}
        for event, element in et.iterparse(xml_file, events=['end', 'start-ns']):
            if event == "start-ns":
                namespaces[element[0]] = element[1]
            elif event == "end":
                el = element
        parsed = et.ElementTree(el)
        xml_root = parsed.getroot()
        xpath_result = elementpath.iter_select(xml_root, rml_rule['iterator'], namespaces=namespaces, parser=XPath3Parser)

        # we need to retrieve both ELEMENTS and ATTRIBUTES in the XML
        data_records = []
        for e in xpath_result:
            data_record = []
            for reference in references:
                data_value = []
                reference = reference.replace('/@', '@')  # deals with `route/stop/@id`

                if reference.startswith('@'):
                    element = None
                    attribute = reference
                elif '@' in reference:
                    element = reference.split('@')[0]
                    attribute = reference.split('@')[1]
                else:
                    element = reference
                    attribute = None

                if element:
                    for r in e.findall(element, namespaces=namespaces):
                        if attribute:
                            data_value.append(r.get(attribute))
                        else:
                            data_value.append(r.text)
                else:
                    attribute = attribute[1:]  # do not use the starting @ from the attribute
                    data_value.append(e.attrib[attribute])
                data_record.append(data_value)
            data_records.append(data_record)

        xml_df = pd.DataFrame.from_records(data_records, columns=references)
    except ModuleNotFoundError:
        xml_df = _parse_xml_file_without_elementpath(xml_file, rml_rule, references)

    # add columns with null values for those references in the mapping rule that are not present in the data file
    missing_references_in_df = list(set(references).difference(set(xml_df.columns)))
    xml_df[missing_references_in_df] = None
    xml_df.dropna(axis=0, how='any', inplace=True)

    for reference in references:
        xml_df = xml_df.explode(reference)

    return xml_df


def _parse_xml_file_without_elementpath(xml_file, rml_rule, references):
    xml_root = et.parse(xml_file).getroot()
    xpath_result = _select_simple_xml_elements(xml_root, rml_rule['iterator'])

    data_records = []
    for element in xpath_result:
        data_record = []
        for reference in references:
            values = []
            reference = reference.replace('/@', '@')
            if reference.startswith('@'):
                values.append(element.attrib.get(reference[1:]))
            elif '@' in reference:
                element_path, attribute = reference.split('@', 1)
                for child in element.findall(element_path):
                    values.append(child.attrib.get(attribute))
            else:
                for child in element.findall(reference):
                    values.append(child.text)
            data_record.append(values)
        data_records.append(data_record)

    return pd.DataFrame.from_records(data_records, columns=references)


def _select_simple_xml_elements(xml_root, iterator):
    iterator = iterator.strip()
    if iterator == '.':
        return [xml_root]

    if iterator == '/*':
        return [xml_root]

    if not iterator.startswith('/'):
        return xml_root.findall(iterator)

    path_parts = [part for part in iterator.split('/') if part]
    if not path_parts:
        return [xml_root]

    if path_parts[0] == xml_root.tag:
        path_parts = path_parts[1:]

    if not path_parts:
        return [xml_root]

    return xml_root.findall('/'.join(path_parts))
