"""Functions to help disaggregate values to LAU and populate DB with data."""
import numpy as np
import pandas as pd

from energyemissionsregio import utils


def distribute_data_equally(
    target_data: pd.DataFrame,
    source_resolution: str,
    target_regions: pd.DataFrame,
    proxy_confidence_level: int,
) -> pd.DataFrame:
    """
    Assigns the same value as in `target_data` to all target regions.

    :param target_data: Dataframe containing target data
    :type target_data: pd.DataFrame

    :param source_resolution: The original resolution of the target data
    :type source_resolution: str

    :param target_regions: Dataframe containing target regions
    :type target_regions: pd.DataFrame

    :param proxy_confidence_level: The confidence level in the way the data is disaggregated
    :type proxy_confidence_level: int

    :returns: final_df
    :rtype: pd.DataFrame
    """

    regions_df = utils.match_source_target_resolutions(
        source_resolution, target_regions
    )

    final_df = pd.merge(
        regions_df,
        target_data,
        left_on="match_region_code",
        right_on="region_code",
        how="right",
    )

    final_df.drop(
        columns=[
            "region_code",
            "match_region_code",
        ],
        inplace=True,
    )

    final_df.rename(columns={"code": "region_code"}, inplace=True)

    # confidence level
    final_df[
        "value_confidence_level"
    ] = np.minimum(  # min of target data and proxy confidence level
        final_df["value_confidence_level"],
        proxy_confidence_level,
    )

    return final_df


def perform_proxy_based_disaggregation(
    target_data,
    proxy_data,
    source_resolution,
    proxy_confidence_level,
    round_to_int=False,
) -> pd.DataFrame:
    """
    Disaggregates data to target regions based on the proportion of the proxy values.

    :param target_data: Dataframe containing target data
    :type target_data: pd.DataFrame

    :param proxy_data: Dataframe containing proxy data
    :type proxy_data: pd.DataFrame

    :param source_resolution: The original resolution of the target data
    :type source_resolution: str

    :param proxy_confidence_level: The confidence level in the way the data is disaggregated
    :type proxy_confidence_level: int

    **Default arguments:**

    :param round_to_int: Indicates if the resulting disaggregated values should be converted
    to int or not
        |br| * the default value is 'False'
    :type round_to_int: bool

    :returns: final_df
    :rtype: pd.DataFrame
    """

    proxy_data = utils.match_source_target_resolutions(source_resolution, proxy_data)

    final_df = utils.disaggregate_data(target_data, proxy_data, proxy_confidence_level)

    final_df.drop(columns=["match_region_code"], inplace=True)

    if round_to_int:
        final_df["value"] = final_df["value"].astype(int)

    return final_df
