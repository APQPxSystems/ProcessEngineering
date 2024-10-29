# Import libraries
import streamlit as st
import pandas as pd
from io import BytesIO

# Streamlit Configurations
st.set_page_config(page_title="ME3 Apps", layout="wide")
hide_st_style = """
                <style>
                #MainMenu {visibility:hidden;}
                footer {visibility:hidden;}
                header {visibility:hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Remove top white space
st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
st.title("Prime or Reassy Identifier")
st.write("____________________________________________")

# Upload defect details
st.subheader("Raw Data Configurations")
raw_data = st.file_uploader("Upload excel file of raw data of internal defects")
if raw_data is not None:
    raw_data = pd.read_excel(raw_data)

    # Create Product Name, Lot Number, Serial Number variables from column selection
    raw_data_cols = raw_data.columns.tolist()
    
    # Use selected column names to index the DataFrame rather than assigning strings directly to the columns
    product_name_col = st.selectbox("Select Product Name column:", raw_data_cols)
    lot_number_col = st.selectbox("Select Lot Number column:", raw_data_cols)
    serial_number_col = st.selectbox("Select Serial Number column:", raw_data_cols)
    date_detected_col = st.selectbox("Select Date Detected column:", raw_data_cols)
    
    # Create concatenated column of Product Name, Lot Number, Serial Number
    raw_data["PN+LN+SN"] = "PN:" + raw_data[product_name_col].astype(str) + " LN:" + raw_data[lot_number_col].astype(str) + " SN:" + raw_data[serial_number_col].astype(str)
    
    method = st.selectbox("Select Method", ["Method 1", "Method 2"])
    
    if method == "Method 1":
        # Method 1
        # Add a Prime or Reassy Column
        # First occurrence of each PN + LN + SN is PRIME.
        # Next occurrences are REASSY.
        
        st.write("Assuming that first occurrence of Product Name, Lot Number, and Serial Number is PRIME while succeeding occurrences are REASSY.")
        raw_data["Prime/Reassy"] = raw_data["PN+LN+SN"].duplicated(keep='first').apply(lambda x: 'Reassy' if x else 'Prime')
    
        st.write("____________________________________________")
        st.subheader("Output Data")
        st.dataframe(raw_data)
    
    if method == "Method 2":
        st.write("Assuming that occurrences of a Product Name, Lot Number, and Serial Number on the same day are all PRIME while succeeding occurrences on other dates are REASSY.")
    
        # Sort by Product Code and Date Detected to ensure chronological order
        raw_data = raw_data.sort_values(by=['PN+LN+SN', date_detected_col]).reset_index(drop=True)

        # Define a function to mark 'Prime' for the first date, and 'Reassy' for later dates of each Product Code
        def label_status(group):
            # Mark the first date for each product code as 'Prime' and others as 'Reassy'
            first_date = group[date_detected_col].iloc[0]
            group['Status'] = group[date_detected_col].apply(lambda x: 'Prime' if x == first_date else 'Reassy')
            return group

        # Apply the function by grouping with Product Code
        raw_data = raw_data.groupby('PN+LN+SN', group_keys=False).apply(label_status)
        
        st.write("____________________________________________")
        st.subheader("Output Data")
        st.dataframe(raw_data)
    
    # Code to download the generated DataFrame as Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        raw_data.to_excel(writer, index=False, sheet_name='Output Data')
        writer.close()
    
    output.seek(0)
    
    st.download_button(
        label="Download Output as Excel",
        data=output,
        file_name="Prime_Reassy_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
