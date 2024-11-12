from typing import Optional, List, Dict
import re
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# number of chars to consider based on a resolution
char_dict = {"NUTS3": 5, "NUTS2": 4, "NUTS1": 3, "NUTS0": 2}


def solve_dfs(df_1: pd.DataFrame, df_2: pd.DataFrame, operator: str) -> pd.DataFrame:
    """
    Performs arithmetic operations on the 'value' columns of `df_1` and `df_2`.

    :param df_1: Dataframe containing a 'value' column
    :type df_1: pd.DataFrame

    :param df_2: Dataframe containing a 'value' column
    :type df_2: pd.DataFrame

    :param operator: Indicates the operation to be performed
    :type operator: str

    :returns: final_df
    :rtype: pd.DataFrame
    """

    result = pd.merge(
        df_1,
        df_2,
        on=["region_code"],
        how="inner",
    )

    result["value_x"].replace([np.inf, np.nan], 0, inplace=True)
    result["value_y"].replace([np.inf, np.nan], 0, inplace=True)

    if operator == "+":
        result["value"] = result["value_x"] + result["value_y"]

    elif operator == "/":
        result["value"] = result["value_x"] / result["value_y"]

        # NOTE: there are some 0s in the data that lead to NAs/infinity in the calculation due to divide by 0 problem
        # for now these are set to 0
        if np.isinf(result["value"].values).any() or result["value"].isna().any():
            warnings.warn("INFs/NAs present in calculated data. These are set to 0")
            result["value"].replace([np.inf, np.nan], 0, inplace=True)

    elif operator == "*":
        result["value"] = result["value_x"].mul(result["value_y"])

    else:
        raise ValueError("Unknown operation")

    # take minimum of two value_confidence_level
    result["value_confidence_level"] = result[
        ["value_confidence_level_x", "value_confidence_level_y"]
    ].min(axis=1)

    result.drop(
        columns=[
            "value_x",
            "value_y",
            "value_confidence_level_x",
            "value_confidence_level_y",
        ],
        inplace=True,
    )

    return result


def get_proxy_var_list(proxy_equation: str) -> List[str]:
    """
    Splits the `proxy_equation` to get a list of proxy vars.

    :param proxy_equation: String containing the equation.
    :type proxy_equation: str

    :returns: proxy_vars
    :rtype: list
    """
    proxy_equation = "".join(proxy_equation.split())

    pattern = r"[\+\*\|\-\/]"

    # Splitting the string using the defined pattern
    split_result = re.split(pattern, proxy_equation)

    # Filtering out empty strings and digits
    proxy_vars = [item for item in split_result if item and not is_float(item)]

    return proxy_vars


def solve_proxy_equation(
    equation: str, proxy_data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Performs the arithmetic operations specified in the `equation` on the 'value'
    column in the dataframe contained in `proxy_data_dict`.

    .. note::
        Currently covers the following cases:
        0. simple proxy: var_1
        1. several proxies added without weighting: var_1 + var_2 + var_3 ...
        2. 1 proxy divided by the other: var_1/var_2
        3. several proxies added with weighting: 2*var_1 |+ 3*var_2 | .....
        4. sum of proxies (or divide or multiply two proxies), divided by a proxy: var_1 + var_2 ... |/ var_n
        5. multiply two proxies: var_1 * var_2

    :param equation: String containing the equation.
    :type equation: str

    :param proxy_data_dict: A dictionary containing the variable names specified in the `equation`
    and the corresponding dataframes.
    :type proxy_data_dict: dict

    :returns: result
    :rtype: pd.DataFrame
    """

    equation = "".join(equation.split())

    # normalise value column before performing arithmetic operations
    proxy_data_dict_normalized = {}
    for var_name, data_df in proxy_data_dict.items():
        _df = data_df.copy()
        # If there is no variance in data, we cannot normailize it. So everything is just set to 0
        if len(_df["value"].unique()) == 1:
            _df["value"] = 0
        else:
            _df["value"] = (
                _df["value"] / _df["value"].max()
            )  # normalizing this way to retain true 0s in the normalized data

        proxy_data_dict_normalized[var_name] = _df

    def _calculate(_eq):
        if "/" in _eq:
            [var_1, var_2] = _eq.split("/")

            var_1_df = proxy_data_dict_normalized[var_1]
            var_2_df = proxy_data_dict_normalized[var_2]

            result = solve_dfs(var_1_df, var_2_df, "/")

        elif "+" in _eq:
            proxy_vars = _eq.split("+")

            for i, var_name in enumerate(proxy_vars):
                var_data = proxy_data_dict_normalized[var_name]

                if i == 0:
                    result = var_data

                else:
                    result = solve_dfs(result, var_data, "+")

        elif "*" in _eq:
            [var_1, var_2] = _eq.split("*")

            if is_float(var_1):
                result = proxy_data_dict_normalized[var_2]
                result["value"] = result["value"] * float(var_1)

            else:
                var_1_df = proxy_data_dict_normalized[var_1]
                var_2_df = proxy_data_dict_normalized[var_2]

                result = solve_dfs(var_1_df, var_2_df, "*")

        else:
            result = proxy_data_dict_normalized[_eq]

        return result

    eq_parts = equation.split("|")

    for i, eq_part in enumerate(eq_parts):
        if i == 0:
            result = _calculate(eq_part)

        else:
            operator = eq_part[0]
            _eq_part = eq_part[1:]

            _result = _calculate(_eq_part)

            result = solve_dfs(result, _result, operator)

    return result


def match_source_target_resolutions(
    source_resolution: str, lau_regions: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds a 'match_region_code' column to `lau_regions`. This column contains
    regions from `source_resolution` that correspond to the `lau_regions`.

    :param source_resolution: The resolution of the source value.
    :type source_resolution: str

    :param lau_regions: LAU regions data
    :type lau_regions: pd.DataFrame

    :returns: lau_regions
    :rtype: pd.DataFrame
    """
    n_char = char_dict[source_resolution]
    if "code" in lau_regions.columns:
        lau_regions["match_region_code"] = lau_regions["code"].str[:n_char]
    else:
        lau_regions["match_region_code"] = lau_regions["region_code"].str[:n_char]

    return lau_regions


def disaggregate_value(target_value: float, proxy_data: pd.DataFrame) -> pd.DataFrame:
    """
    Performs the arithmetic operations specified in the `equation` on the 'value'
    column in the dataframe contained in `proxy_data_dict`.

    .. note::
        Currently covers the following cases:
        0. simple proxy: var_1
        1. several proxies added without weighting: var_1 + var_2 + var_3 ...
        2. 1 proxy divided by the other: var_1/var_2
        3. several proxies added with weighting: 2*var_1 |+ 3*var_2 | .....
        4. sum of proxies (or divide or multiply two proxies), divided by a proxy: var_1 + var_2 ... |/ var_n
        5. multiply two proxies: var_1 * var_2

    :param equation: String containing the equation.
    :type equation: str

    :param proxy_data_dict: A dictionary containing the variable names specified in the `equation`
    and the corresponding dataframes.
    :type proxy_data_dict: dict

    :returns: result
    :rtype: pd.DataFrame
    """
    disagg_data = proxy_data.copy(deep=True)

    total = disagg_data["value"].values.sum()

    if total == 0:
        if target_value == 0:
            disagg_data["disagg_value"] = 0

            # clean up columns
            disagg_data = disagg_data.drop(columns=["value"]).rename(
                columns={"disagg_value": "value"}
            )
        else:
            print(target_value)
            print(proxy_data["region_code"].values)
            raise ValueError("The proxy is not suitable. has all 0s")

    else:
        # disaggregte
        disagg_data["share"] = disagg_data["value"] / total
        disagg_data["disagg_value"] = disagg_data["share"] * target_value

        # clean up columns
        disagg_data = disagg_data.drop(columns=["value", "share"]).rename(
            columns={"disagg_value": "value"}
        )

    return disagg_data


def disaggregate_data(
    target_data: pd.DataFrame, proxy_data: pd.DataFrame, proxy_confidence_level: int
) -> pd.DataFrame:
    """
    Disaggregates values in `target_data` to the spatial resolution of the `proxy_data` based on the
    proportion of the values in `proxy_data`.

    :param target_data: Dataframe with values at the source spatial level.
    :type target_data: pd.DataFrame

    :param proxy_data: Dataframe with values at the target spatial level.
    :type proxy_data: pd.DataFrame

    :param proxy_confidence_level: The confidence level in the way the data is disaggregated
    :type proxy_confidence_level: int

    :returns: final_disagg_df
    :rtype: pd.DataFrame
    """
    # disaggregate value in each source region to the corresponding target regions
    disagg_df_list = []

    for key_row in target_data.iterrows():
        row = key_row[1]
        _proxy_data = proxy_data[proxy_data["match_region_code"] == row["region_code"]]

        disagg_df = disaggregate_value(row["value"], _proxy_data)

        # calculate value confidence level --> min(proxy confidence level, target data confidence level, proxy data confidence level)
        _confidence_level = min(
            proxy_confidence_level, row["value_confidence_level"]
        )  # min of target data and proxy_confidence_level

        disagg_df[
            "value_confidence_level"
        ] = np.minimum(  # min of above and proxy data confidence level
            disagg_df["value_confidence_level"],
            _confidence_level,
        )

        disagg_df_list.append(disagg_df)

    final_disagg_df = pd.concat(disagg_df_list)

    return final_disagg_df


def get_confidence_level(r2) -> int:
    """
    Returns 'confidence_level' corresponding to the R-squared error value obained during
    missing value imputation.

    :param r2: float value.
    :type r2: float

    :returns: confidence_level
    :rtype: int
    """
    if r2 > 0.8:
        confidence_level = 4  #  HIGH
    elif (r2 > 0.5) and (r2 <= 0.8):
        confidence_level = 3  # MEDIUM
    elif (r2 > 0.2) and (r2 <= 0.5):
        confidence_level = 2  # LOW
    elif r2 <= 0.2:
        confidence_level = 1  # VERY LOW

    return confidence_level
