from typing import Optional, List, Dict
import re
import warnings
import numpy as np
import pandas as pd


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# number of chars to consider based on a resolution
char_dict = {"NUTS3": 5, "NUTS2": 4, "NUTS1": 3, "NUTS0": 2}


def get_proxy_var_list(proxy_equation: str) -> List[str]:
    """
    Splits the `proxy_equation` to get a list of proxy vars.

    :param proxy_equation: String containing the equation.
    :type proxy_equation: str

    :returns: proxy_vars
    :rtype: list
    """
    operators = r"[\+\-\*\%\(\)\/\n]"

    # Splitting the string using the defined pattern
    split_result = re.split(operators, proxy_equation)

    # Filtering out empty strings and digits
    var_list = [
        part.strip()
        for part in split_result
        if part.strip() and not part.strip().isdigit() and not is_float(part.strip())
    ]

    return var_list


def solve_proxy_equation(
    equation: str, proxy_data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Performs the arithmetic operations specified in the `equation` on the 'value'
    column in the dataframe contained in `proxy_data_dict`.

    :param equation: String containing the equation.
    :type equation: str

    :param proxy_data_dict: A dictionary containing the variable names specified in the `equation`
    and the corresponding dataframes.
    :type proxy_data_dict: dict

    :returns: result
    :rtype: pd.DataFrame
    """
    result = None

    for proxy_var, proxy_data in proxy_data_dict.items():
        # If there is no variance in data, we cannot normailize it. So everything is just set to 0
        if len(proxy_data["value"].unique()) == 1:
            proxy_data["value"] = 0
        else:
            proxy_data["value"] = (
                proxy_data["value"] / proxy_data["value"].max()
            )  # normalizing this way to retain true 0s in the normalized data

        proxy_data.rename(columns={"value": proxy_var}, inplace=True)

        if result is None:
            result = proxy_data
        else:
            result = pd.merge(result, proxy_data, on=["region_code"])

            # merged value_confidence_level
            # NOTE: depends on the poorest quality rating, Hence min
            result["value_confidence_level"] = result[
                ["value_confidence_level_x", "value_confidence_level_y"]
            ].min(axis=1)

            result.drop(
                columns=[
                    "value_confidence_level_x",
                    "value_confidence_level_y",
                ],
                inplace=True,
            )

    result = result.eval(f"value = {equation}")
    result["value"] = result["value"].replace([np.inf, np.nan], 0)

    result = result[["region_code", "value", "value_confidence_level"]].copy()

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
