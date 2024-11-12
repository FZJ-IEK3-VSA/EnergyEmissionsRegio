import pytest
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from energyemissionsregio.config import SHP_PATH


@pytest.fixture
def test_data():
    my_dict = {
        "region_code": ["DE600", "DE142"],
        "value": [1, 15],
        "value_confidence_level": [3, 3],
    }
    data = pd.DataFrame(my_dict)
    return data


@pytest.fixture
def proxy_data_1():
    regions_gdf = gpd.read_file(os.path.join(SHP_PATH, "LAU.shp"))
    reg_codes = list(
        regions_gdf[regions_gdf["code"].str.startswith(("DE600", "DE142"))]["code"]
    )

    my_dict = {
        "region_code": reg_codes,
        "value": [
            800,
            0,
            100,
            400,
            700,
            600,
            630,
            600,
            300,
            408,
            504,
            677,
            800,
            0,
            400,
            1000,
        ],
        "value_confidence_level": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    }
    data = pd.DataFrame(my_dict)
    return data


@pytest.fixture
def proxy_data_2():
    regions_gdf = gpd.read_file(os.path.join(SHP_PATH, "LAU.shp"))
    reg_codes = list(
        regions_gdf[regions_gdf["code"].str.startswith(("DE600", "DE142"))]["code"]
    )

    my_dict = {
        "region_code": reg_codes,
        "value": [1, 14, 10, 4, 7, 14, 6, 6, 3, 4, 5, 6, 8, 9, 4, 1],
        "value_confidence_level": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    }
    data = pd.DataFrame(my_dict)
    return data


@pytest.fixture
def proxy_data_3():
    regions_gdf = gpd.read_file(os.path.join(SHP_PATH, "LAU.shp"))
    reg_codes = list(
        regions_gdf[regions_gdf["code"].str.startswith(("DE600", "DE142"))]["code"]
    )

    my_dict = {
        "region_code": reg_codes,
        "value": [0, 4, 6, 6, 6, 8, 10, 3, 4, 5, 6, 8, 9, 10, 4, 7],
        "value_confidence_level": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    }
    data = pd.DataFrame(my_dict)
    return data


@pytest.fixture
def test_lau_regions():
    regions_gdf = gpd.read_file(os.path.join(SHP_PATH, "LAU.shp"))
    sub_regions_df = regions_gdf[
        regions_gdf["code"].str.startswith(("DE600", "DE142"))
    ][["code"]].copy()

    return sub_regions_df


@pytest.fixture
def imputation_data():
    # Set seed for reproducibility
    np.random.seed(42)

    # Number of samples
    n_samples = 100

    column_A = np.random.randn(n_samples)
    column_B = column_A * 0.95 + np.random.normal(
        0, 0.1, n_samples
    )  # High correlation with some noise
    column_B[
        np.random.choice(n_samples, 20, replace=False)
    ] = np.nan  # Introduce missing values

    # Generate moderately correlated column D
    column_C = 0.3 * column_A + np.random.normal(
        0, 1, n_samples
    )  # Moderate correlation

    column_D = np.random.randn(n_samples)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "Column_A": column_A,
            "Column_B": column_B,  # This column contains missing values
            "Column_C": column_C,
            "Column_D": column_D,
        }
    )

    return data
