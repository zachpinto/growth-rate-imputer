import streamlit as st
import pandas as pd
import numpy as np

# Load data from file
def load_data(use_demo):
    if use_demo:
        return pd.read_excel('demo.xlsx')
    uploaded_file = st.sidebar.file_uploader("1. Upload your data file", type=['csv', 'xlsx'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    return None

# Impute a single row based on given growth rate
def impute_row(row, start_col, end_col, growth_rate):
    imputed_row = row.copy()
    columns = row.index[start_col:end_col + 1]
    values = row[columns].dropna()

    # Compute growth rate if possible
    if len(values) > 1:
        first_value, last_value = values.iloc[0], values.iloc[-1]
        num_periods = len(columns) - 1
        growth_rate = (last_value - first_value) / num_periods

    if pd.isna(growth_rate):
        return imputed_row

    # Impute missing values based on growth rate
    for col in columns:
        if pd.isna(imputed_row[col]):
            prev_col = columns[columns.get_loc(col) - 1]
            if columns.get_loc(col) > 0 and not pd.isna(imputed_row[prev_col]):
                imputed_row[col] = round(imputed_row[prev_col] + growth_rate)
            else:
                next_value = values.iloc[0] if len(values) > 0 else None
                if next_value is not None:
                    next_col = values.index[0]
                    periods_before = columns.get_loc(next_col) - columns.get_loc(col)
                    imputed_row[col] = round(next_value - (growth_rate * periods_before))
    return imputed_row

# Compute growth rates for each row
def compute_growth_rates(df, start_col, end_col):
    growth_rates = {}
    columns = df.columns[start_col:end_col + 1]
    for idx, row in df.iterrows():
        values = row[columns].dropna()
        if len(values) > 1:
            first_value, last_value = values.iloc[0], values.iloc[-1]
            num_periods = len(values) - 1
            growth_rates[idx] = (last_value - first_value) / num_periods
    return growth_rates

# Apply growth rates to impute missing values
def apply_growth_rates(df, start_col, end_col, growth_rates, category_col=None):
    if category_col:
        category_means = df.groupby(category_col).apply(
            lambda x: np.nanmean([growth_rates.get(idx, np.nan) for idx in x.index])
        )
        df = df.apply(
            lambda row: impute_row(row, start_col, end_col, category_means.get(row[category_col], np.nan)), axis=1
        )
    else:
        mean_growth_rate = np.nanmean(list(growth_rates.values()))
        df = df.apply(lambda row: impute_row(row, start_col, end_col, mean_growth_rate), axis=1)
    return df

# Main function to create Streamlit app
def main():
    st.title("Growth Rate Data Imputation Tool")
    st.write("This tool imputes missing data points in your dataset based on a calculated growth rate. "
             "Follow the sidebar instructions to upload your data and configure the imputation settings.")
    st.write("**Author:** Zachary Pinto")
    st.write("**GitHub:** zachpinto")
    st.write("**edX:** zachwalker98")
    st.write("**City and Country:** Ballston Spa, NY, United States")
    st.write("**Date:** August 4th, 2024")

    use_demo = st.sidebar.checkbox("2. Use demo file (demo.xlsx)")
    df = load_data(use_demo)

    if df is not None:
        st.write("Uploaded Data Preview")
        st.dataframe(df.head())

        # Select columns for imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        start_year = st.sidebar.selectbox("3. Select Start Year Column:", numeric_cols)
        end_year = st.sidebar.selectbox("4. Select End Year Column:", numeric_cols)
        handling_method = st.sidebar.radio(
            "5. Handling Method for Undefined Growth Rates:",
            ['Average of ALL rows', 'Average of rows with SHARED CATEGORY']
        )
        category_col = None
        if handling_method == 'Average of rows with SHARED CATEGORY':
            category_col = st.sidebar.selectbox("6. Select the Category Column:", df.columns)

        # Impute data based on selected columns
        if st.sidebar.button("7. Impute Data"):
            start_idx = df.columns.get_loc(start_year)
            end_idx = df.columns.get_loc(end_year)
            growth_rates = compute_growth_rates(df, start_idx, end_idx)
            df = apply_growth_rates(df, start_idx, end_idx, growth_rates, category_col)

            st.write("Imputed Data")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download imputed data as CSV", csv, "imputed_data.csv", "text/csv")

# Run the app
if __name__ == "__main__":
    main()
