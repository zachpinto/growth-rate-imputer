import streamlit as st
import pandas as pd
import base64
import numpy as np


class DataImputer:
    def __init__(self, dataframe, start_year, end_year, category_col=None, interpolation_method="Linear"):
        # model parameters
        self.dataframe = dataframe
        self.start_year = start_year
        self.end_year = end_year
        self.category_col = category_col
        self.interpolation_method = interpolation_method

    def impute_data(self):
        if self.category_col:
            if self.interpolation_method == "Linear":
                return self.impute_based_on_category(self.linear_interpolate)
            else:  # Exponential interpolation
                return self.impute_based_on_category(self.exponential_interpolate)
        else:
            if self.interpolation_method == "Linear":
                return self.impute_across_all(self.linear_interpolate)
            else:  # Exponential interpolation
                return self.impute_across_all(self.exponential_interpolate)

    def impute_based_on_category(self, interpolation_func):
        # Apply interpolation to each category group
        return self.dataframe.groupby(self.category_col, group_keys=False).apply(interpolation_func)

    def impute_across_all(self, interpolation_func):
        # Apply interpolation to the whole dataset
        return interpolation_func(self.dataframe)

    def linear_interpolate(self, df):
        # Interpolation logic to handle "backward" filling from the end year
        start_idx = df.columns.get_loc(self.start_year)
        end_idx = df.columns.get_loc(self.end_year) + 1

        # Backward fill from the end year to handle rows with data only in the end year
        df.iloc[:, start_idx:end_idx] = df.iloc[:, start_idx:end_idx].bfill(axis=1)

        # Now apply linear interpolation
        df.iloc[:, start_idx:end_idx] = df.iloc[:, start_idx:end_idx].interpolate(method='linear', axis=1)

        return df

    def exponential_interpolate(self, df):
        start_idx = df.columns.get_loc(self.start_year)
        end_idx = df.columns.get_loc(self.end_year) + 1

        # Compute growth rates for rows with both start and end year data
        rates = []
        for i, row in df.iterrows():
            if not pd.isna(row[self.start_year]) and not pd.isna(row[self.end_year]):
                rate = (row[self.end_year] / row[self.start_year]) ** (1 / (end_idx - start_idx - 1)) - 1
                rates.append(rate)

        avg_rate = np.mean(rates) if rates else 0  # Calculate average rate

        # Apply growth rate to all rows
        for i, row in df.iterrows():
            if not pd.isna(row[self.start_year]):
                # If there's a start year value, apply the specific or average growth rate
                rate = rates.pop(0) if rates else avg_rate
                for j in range(start_idx + 1, end_idx):
                    df.at[i, df.columns[j]] = df.at[i, df.columns[j - 1]] * (1 + rate)
            else:
                # For rows with only end year data, apply the average rate backwards
                for j in range(end_idx - 2, start_idx - 1, -1):
                    df.at[i, df.columns[j]] = df.at[i, df.columns[j + 1]] / (1 + avg_rate)

        return df


def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading data: {e}")
    return None


def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="imputed_data.csv">Download imputed data as CSV</a>'
    return href

st.write("## Growth Rate Imputations for Time Series Data")
st.write("""
##### Directions
This application is designed to impute missing values in datasets across time series data based on implied growth rates. 
The directions are clearly outlined on the sidebar.""")

st.write("""
First, upload your data in csv or xlsx format. Next, select your start and end periods (ie. years). 
Then, select how you would like to impute null values. 
Then, select either linear or exponential interpolation for filling these missing values. 
Finally, click the 'Impute Data' button to see the imputed data. Scroll down the page to preview the imputed data.
You can also click the 'Download imputed data as CSV' link to download the imputed data.""")

st.write("""
As the imputations are based on the implied growth rates between selected periods 
(or the average rates of all or some records), 
not all records need to have non-null values for the start and end periods, 
however each record must have data for either the start or end periods in order to complete the imputations. 
The application will impute the missing values based on the average growth rate of the available data.
""")

with st.sidebar:
    st.write("### 1.) Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

df = None

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())

        # Sidebar for inputs
        with st.sidebar:
            st.write("### 2.) Column Selection")
            column_names = df.columns.tolist()
            start_year = st.selectbox("Select the Start Year Column:", column_names, key='start_year')
            end_year = st.selectbox("Select the End Year Column:", column_names, key='end_year')

            # Check if the selected columns contain non-numerical non-null data
            if not pd.to_numeric(df[start_year].dropna(), errors='coerce').notnull().all() or \
               not pd.to_numeric(df[end_year].dropna(), errors='coerce').notnull().all():
                st.error("Selected start or end year columns contain non-numerical, non-null data. "
                         "Please select different columns.")
                df = None  # Prevent further processing

            # Continue with other inputs if the data checks are passed
            if df is not None:
                st.write("### 3.) Missing Data Handling")
                option = st.radio(
                    "For years without any clear growth rates, "
                    "would you like their growth rates to be calculated using:",
                    ('An average of ALL rows', 'An average of rows with a SHARED CATEGORY'),
                    key='data_handling_option'
                )

                category_col = None
                if option == 'An average of rows with a SHARED CATEGORY':
                    category_col = st.selectbox("Select the column to categorize by:", column_names,
                                                index=0,
                                                key='category_column')

                st.write("### 4.) Interpolation Method")
                interpolation_method = st.selectbox("Select the interpolation method:", ["Linear", "Exponential"],
                                                    key='interpolation_method')

                if st.button("Impute Data", key='impute_data'):
                    st.session_state['impute'] = True
                else:
                    st.session_state['impute'] = st.session_state.get('impute', False)

# Outside the sidebar, check if the button was pressed and display the imputed data
if 'impute' in st.session_state and st.session_state['impute'] and df is not None:
    # Check if df is defined and the impute button has been pressed
    imputer = DataImputer(df, start_year, end_year, category_col, interpolation_method)
    imputed_df = imputer.impute_data()

    if imputed_df is not None and not imputed_df.empty:
        st.write("### Imputed Data Preview")
        st.dataframe(imputed_df.head())
        st.markdown(get_table_download_link(imputed_df), unsafe_allow_html=True)
    else:
        st.error("Data imputation failed or no data to display.")
