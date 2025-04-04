SHP_PATH = r"C:\Users\s.patil\OneDrive - Forschungszentrum Jülich GmbH\Documents\code\ETHOS.RegionData\data\input\02_processed\shapefiles"

DATA_PATH = r"C:\Users\s.patil\OneDrive - Forschungszentrum Jülich GmbH\Documents\code\ETHOS.RegionData\data\input\03_imported\collected_data"
RAW_DATA_PATH = r"C:\Users\s.patil\OneDrive - Forschungszentrum Jülich GmbH\Documents\code\ETHOS.RegionData\data\input\01_raw"

CLIMATE_DATA_PATH = r"C:\Users\s.patil\OneDrive - Forschungszentrum Jülich GmbH\Documents\code\ETHOS.RegionData\data\input\03_imported\climate_projections"

units = {
    "de_employment_in_agriculture": "number",
    "es_utilized_agricultural_area": "square kilometer",
    "de_employment_in_textile_and_leather_manufacturing": "number",
    "industrial_or_commercial_units_cover": "square kilometer",
    "de_employment_in_food_and_beverage_manufacturing": "number",
    "de_employment_in_mechanical_and_automotive_engineering": "number",
    "de_employment_in_mechatronics_energy_and_electrical": "number",
    "de_employment_in_wood_processing": "number",
    "employment_in_construction": "number",
    "construction_sites_cover": "square kilometer",
    "road_transport_of_freight": "Mt",
    "road_network": "kilometer",
    "population": "number",
    "area": "square kilometer",
    "es_number_of_dwellings": "number",
    "continuous_urban_fabric_cover": "square kilometer",
    "discontinuous_urban_fabric_cover": "square kilometer",
    "industrial_or_commercial_units_cover": "square kilometer",
    "port_areas_cover": "square kilometer",
    "airports_cover": "square kilometer",
    "mineral_extraction_sites_cover": "square kilometer",
    "dump_sites_cover": "square kilometer",
    "construction_sites_cover": "square kilometer",
    "green_urban_areas_cover": "square kilometer",
    "sport_and_leisure_facilities_cover": "square kilometer",
    "non_irrigated_arable_land_cover": "square kilometer",
    "permanently_irrigated_land_cover": "square kilometer",
    "rice_fields_cover": "square kilometer",
    "vineyards_cover": "square kilometer",
    "fruit_trees_and_berry_plantations_cover": "square kilometer",
    "olive_groves_cover": "square kilometer",
    "pastures_cover": "square kilometer",
    "permanent_crops_cover": "square kilometer",
    "complex_cultivation_patterns_cover": "square kilometer",
    "agriculture_with_natural_vegetation_cover": "square kilometer",
    "agro_forestry_areas_cover": "square kilometer",
    "broad_leaved_forest_cover": "square kilometer",
    "coniferous_forest_cover": "square kilometer",
    "mixed_forest_cover": "square kilometer",
    "natural_grasslands_cover": "square kilometer",
    "moors_and_heathland_cover": "square kilometer",
    "sclerophyllous_vegetation_cover": "square kilometer",
    "transitional_woodland_shrub_cover": "square kilometer",
    "beaches_dunes_and_sand_cover": "square kilometer",
    "bare_rocks_cover": "square kilometer",
    "sparsely_vegetated_areas_cover": "square kilometer",
    "burnt_areas_cover": "square kilometer",
    "glaciers_and_perpetual_snow_cover": "square kilometer",
    "inland_marshes_cover": "square kilometer",
    "peat_bogs_cover": "square kilometer",
    "salt_marshes_cover": "square kilometer",
    "salines_cover": "square kilometer",
    "intertidal_flats_cover": "square kilometer",
    "water_courses_cover": "square kilometer",
    "water_bodies_cover": "square kilometer",
    "coastal_lagoons_cover": "square kilometer",
    "estuaries_cover": "square kilometer",
    "sea_and_ocean_cover": "square kilometer",
    "de_number_of_passenger_cars_emission_group_euro_1": "number",
    "de_number_of_passenger_cars_emission_group_euro_2": "number",
    "de_number_of_passenger_cars_emission_group_euro_3": "number",
    "de_number_of_passenger_cars_emission_group_euro_4": "number",
    "de_number_of_passenger_cars_emission_group_euro_5": "number",
    "de_number_of_passenger_cars_emission_group_euro_6r": "number",
    "de_number_of_passenger_cars_emission_group_euro_6dt": "number",
    "de_number_of_passenger_cars_emission_group_euro_6d": "number",
    "de_number_of_passenger_cars_emission_group_euro_other": "number",
    "de_residential_building_living_area": "square kilometer",
    "employment_in_manufacturing": "number",
    "de_non_residential_building_living_area": "square kilometer",
    "employment_in_agriculture_forestry_and_fishing": "number",
    "es_number_of_commerical_and_service_companies": "number",
    "es_average_daily_traffic_light_duty_vehicles": "number",
    "number_of_cattle": "number",
    "number_of_pigs": "number",
    "number_of_buffaloes": "number",
    "number_of_motorcycles": "number",
    "number_of_paper_and_printing_industries": "number",
    "number_of_cement_industries": "number",
    "number_of_iron_and_steel_industries": "number",
    "number_of_refineries": "number",
    "number_of_non_metallic_minerals_industries": "number",
    "number_of_non_ferrous_metals_industries": "number",
    "number_of_chemical_industries": "number",
    "number_of_glass_industries": "number",
    "average_air_pollution_due_to_pm25": "ug/m3",
    "average_air_pollution_due_to_no2": "ug/m3",
    "average_air_pollution_due_to_o3": "ug/m3",
    "average_air_pollution_due_to_pm10": "ug/m3",
    "number_of_buildings": "number",
    "air_transport_of_freight": "Mt",
    "air_transport_of_passengers": "number",
    "ghg_emissions_from_fc_in_manufacture_of_iron_and_steel": "Mt",
    "ghg_emissions_from_fc_in_manufacture_of_non_ferrous_metals": "Mt",
    "ghg_emissions_from_fc_in_manufacture_of_chemicals": "Mt",
    "ghg_emissions_from_fc_in_manufacture_of_non_metallic_mineral_products": "Mt",
    "ghg_emissions_from_fc_in_manufacture_of_pulp_paper_and_printing": "Mt",
    "cproj_annual_mean_temperature_heating_degree_days": "heating_degree_days",
    "railway_network": "kilometer",
    "ghg_emissions_from_fc_in_food_beverages_and_tobacco_industries": "Mt",
    "ghg_emissions_from_fc_in_rail_transport": "Mt",
    "final_energy_consumption_in_wood_and_wood_products_industries": "MWh",
    "final_energy_consumption_in_agriculture_and_forestry": "MWh",
    "gross_domestic_product": "million Euros",
}

confidence_level_mapping = {
    1: "VERY LOW",
    2: "LOW",
    3: "MEDIUM",
    4: "HIGH",
    5: "VERY HIGH",
}
