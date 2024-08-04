import pandas as pd
import numpy as np
from project import load_data, impute_row, compute_growth_rates, apply_growth_rates

# Test the load_data function to ensure that the demo file is loaded successfully
def test_load_data():
    df = load_data(use_demo=True)
    assert df is not None, "Demo file loaded successfully"

# Test the impute_row function to ensure that missing values are imputed correctly based on growth rate
def test_impute_row():
    row = pd.Series([1, np.nan, 3, np.nan, 5], index=['Year1', 'Year2', 'Year3', 'Year4', 'Year5'])
    expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=['Year1', 'Year2', 'Year3', 'Year4', 'Year5'])
    imputed = impute_row(row, 0, 4, None)
    assert imputed.equals(expected), "The row was not imputed correctly"

# Test the compute_growth_rates function to ensure that growth rates are computed correctly for each row
def test_compute_growth_rates():
    data = {
        'Year1': [1, 2, np.nan],
        'Year2': [np.nan, 3, 4],
        'Year3': [3, np.nan, 5]
    }
    df = pd.DataFrame(data)
    start_col = 0
    end_col = 2
    growth_rates = compute_growth_rates(df, start_col, end_col)

    expected_growth_rates = {0: 2.0, 1: 1.0, 2: 1.0}
    assert growth_rates == expected_growth_rates, "The growth rates were not computed correctly"


