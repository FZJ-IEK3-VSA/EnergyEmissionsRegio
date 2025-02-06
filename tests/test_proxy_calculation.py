import pytest
import numpy as np
from energyemissionsregio.utils import solve_proxy_equation, get_proxy_var_list


@pytest.mark.parametrize(
    "proxy_equation, expected_list",
    [
        ("var_1", ["var_1"]),
        ("var_1+var_2", ["var_1", "var_2"]),
        ("var_1+var_2+var_3", ["var_1", "var_2", "var_3"]),
        ("var_1/var_2", ["var_1", "var_2"]),
        ("(2.14 * var_1) + (3 * var_2) + (6*var_3)", ["var_1", "var_2", "var_3"]),
        ("(var_1+var_2) / var_3", ["var_1", "var_2", "var_3"]),
        ("(var_1 * var_2) / var_3", ["var_1", "var_2", "var_3"]),
        ("var_1*var_3", ["var_1", "var_3"]),
    ],
)
def test_get_proxy_var_list(proxy_equation, expected_list):

    output_list = get_proxy_var_list(proxy_equation)

    assert output_list == expected_list


@pytest.mark.parametrize(
    "proxy_equation, expected_value_region_1, expected_value_region_2",
    [
        ("var_1", 0.8, 0),
        ("var_3", 0, 0.4),
        ("var_1+var_2", 0.87, 1.00),
        ("var_1+var_2+var_3", 0.87, 1.4),
        ("var_1/var_2", 11.2, 0),
        ("(2.14 * var_1) + (3 * var_2) + (6 * var_3)", 1.93, 5.4),
        ("(var_1+var_2) /var_3", 0, 2.5),
        ("(var_1*var_2) / var_3", 0, 0),
        ("var_1*var_3", 0, 0),
    ],
)
def test_solve_proxy_equation(
    proxy_data_1,
    proxy_data_2,
    proxy_data_3,
    proxy_equation,
    expected_value_region_1,
    expected_value_region_2,
):

    proxy_data_dict = {
        "var_1": proxy_data_1,
        "var_2": proxy_data_2,
        "var_3": proxy_data_3,
    }
    proxy_data = solve_proxy_equation(proxy_equation, proxy_data_dict)

    output_value_region_1 = proxy_data[proxy_data["region_code"] == "DE600_02000000"][
        "value"
    ].item()
    output_value_region_2 = proxy_data[proxy_data["region_code"] == "DE142_08416006"][
        "value"
    ].item()

    output_value_region_1 = np.round(output_value_region_1, 2)
    output_value_region_2 = np.round(output_value_region_2, 2)

    assert output_value_region_1 == expected_value_region_1

    assert output_value_region_2 == expected_value_region_2
