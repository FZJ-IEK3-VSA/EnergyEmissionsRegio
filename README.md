EnergyEmissionsRegio: Spatial disaggregation of final energy consumption and emissions data from NUTS0 to LAU level.
==============================

A workflow tool that:
1. Use spatial proxies from various databases, by imputing missing values if any.
2. Disaggregates the energy consumption and emissions from differnt sectors based on proxy specifications.
3. Evaulates confidence level of the disaggregated values
--------

Installation steps 
------------

0. Before you begin:

Please make sure you have mamba installed in your base environment
    ```bash
    conda install mamba -c conda-forge
    ```

1. Clone this repository:
    ```bash
    git clone https://github.com/FZJ-IEK3-VSA/EnergyEmissionsRegio.git
    ```

2. Install dependencies and the repo in a clean conda environment:
    ```bash
    cd EnergyEmissionsRegio
    mamba env create -n eeregio --file=requirements.yml
    conda activate eeregio
    pip install -e .
    ```

The experiments can be found under the experiments folder.