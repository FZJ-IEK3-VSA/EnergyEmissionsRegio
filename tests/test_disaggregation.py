import numpy as np
from energyemissionsregio.utils import solve_proxy_equation
from energyemissionsregio.disaggregation import (
    distribute_data_equally,
    perform_proxy_based_disaggregation,
)


def test_distribute_data_equally(test_data, test_lau_regions):

    disagg_data = distribute_data_equally(test_data, "NUTS3", test_lau_regions)

    assert (
        disagg_data[disagg_data["region_code"] == "DE142_08416018"]["value"].item()
        == 15
    )


def test_proxy_based_disaggregation_single_proxy(test_data, proxy_data_2):

    disagg_data = perform_proxy_based_disaggregation(
        test_data, proxy_data_2, "NUTS3", 4
    )

    output_value = disagg_data[disagg_data["region_code"] == "DE142_08416006"][
        "value"
    ].item()
    excepted_value = 0.14 * 15  # (share of proxy * disaggregated value)

    assert np.round(output_value, 1) == np.round(excepted_value, 1)
    assert (
        disagg_data[disagg_data["region_code"] == "DE142_08416006"][
            "value_confidence_level"
        ].item()
        == 3
    )  # (min(proxy confidence level, proxy value confidence level, source value confidence level))


def test_proxy_based_disaggregation_multiple_proxies(
    test_data, proxy_data_1, proxy_data_2
):

    proxy_data_dict = {"var_1": proxy_data_1, "var_2": proxy_data_2}

    solved_proxy_data = solve_proxy_equation("var_1+var_2", proxy_data_dict)

    disagg_data = perform_proxy_based_disaggregation(
        test_data, solved_proxy_data, "NUTS3", 3
    )

    value_a = disagg_data[disagg_data["region_code"] == "DE142_08416006"][
        "value"
    ].item()
    value_b = disagg_data[disagg_data["region_code"] == "DE142_08416050"][
        "value"
    ].item()

    assert value_a < value_b

    value_a = disagg_data[disagg_data["region_code"] == "DE142_08416018"][
        "value"
    ].item()
    value_b = disagg_data[disagg_data["region_code"] == "DE142_08416022"][
        "value"
    ].item()

    assert value_a > value_b

    disagg_data[disagg_data["region_code"] == "DE142_08416050"][
        "value_confidence_level"
    ] == 3
