import pytest
from energyemissionsregio.missing_value_imputation import impute_data


def test_impute_data(imputation_data):
    r2, imputed_df = impute_data(imputation_data, impute_col="Column_B")

    pytest.set_trace()
