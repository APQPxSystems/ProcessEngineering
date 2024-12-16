import streamlit as st
import pandas as pd

# Streamlit Configurations
st.set_page_config(page_title="PRIME/REASSY Identifier | KentK.", layout="wide")

# Hide Streamlit elements like the menu and footer
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

# Function to determine Prime or Reassy status with sequential Reassy labeling
def determine_status(df, product_col, lot_col, serial_col, issue_col):
    # Sort by selected columns
    df = df.sort_values(by=[product_col, lot_col, serial_col, issue_col]).reset_index(drop=True)
    df['Prime/Reassy'] = None  # Initialize column
    
    # Create a unique identifier for each product
    df['Unique Identifier'] = (
        "PN: " + df[product_col].astype(str) + 
        " LN: " + df[lot_col].astype(str) + 
        " SN: " + df[serial_col].astype(str)
    )
    
    # Process each unique identifier group
    for uid, group in df.groupby('Unique Identifier'):
        reassy_count = 0  # Counter for Reassy numbers
        current_status = 'Prime'  # Start with Prime
        prev_issue_number = None  # Track previous issue number

        for idx in group.index:
            issue_number = df.at[idx, issue_col]
            
            if prev_issue_number is None or issue_number - prev_issue_number > 1:
                if prev_issue_number is None:
                    current_status = 'Prime'
                else:
                    reassy_count += 1
                    current_status = f'Reassy{reassy_count}'
            # Assign the determined status
            df.at[idx, 'Prime/Reassy'] = current_status
            prev_issue_number = issue_number

    return df

# Streamlit app layout
st.title("Defect Analysis AI: Prime and Reassy Identifier")
password = st.text_input("Input app key to continue:")
if password in ["kent", "gian", "tato"]:
    st.write("""This defect analysis tool uses the Reworking Issue Number data to identify the Prime and Reassy Harness.
          For every unique Harness (identified by the combination of Product Number, Lot Number, and Serial Number),
          the Reworking Issue Number will be analyzed. The first occurrence of consecutive Reworking Issue Number/s will
          be identified as PRIME and the next occurrences will be REASSY. This helps to identify the number of defects
          per inspection more accurately instead of relying on the defect detection date.""")
  
    with st.sidebar:
        st.title("ProcessEngineering")
        st.write("_________________________")
        uploaded_file = st.file_uploader("Upload your defect data Excel/CSV file", type=["xlsx", "csv"])
  
    if uploaded_file:
        # Load data
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data:")
        st.dataframe(df)
      
        # Column selection
        product_col = st.selectbox("Select the Product Name column:", df.columns)
        lot_col = st.selectbox("Select the Lot Number column:", df.columns)
        serial_col = st.selectbox("Select the Serial Number column:", df.columns)
        issue_col = st.selectbox("Select the Reworking Issue Number column:", df.columns)
  
        if st.button("Process Data"):
            # Process data
            df = determine_status(df, product_col, lot_col, serial_col, issue_col)

            # Display processed data
            st.write("Processed Data with Prime/Reassy Status:")
            st.dataframe(df)

            # Create a pivot table
            pivot_df = df.pivot_table(index='Unique Identifier', columns='Prime/Reassy', aggfunc='size', fill_value=0)
            pivot_df['Total Count'] = pivot_df.sum(axis=1)

            # Add column for the number of repairs
            pivot_df['Times Repaired'] = (pivot_df.loc[:, pivot_df.columns != 'Total Count'] > 0).sum(axis=1)
            pivot_df = pivot_df.sort_values(by='Times Repaired', ascending=False)

            # Display pivot table
            st.write("Defect Distribution Analysis (Pivot Table):")
            st.dataframe(pivot_df)
          
            # Save results to an Excel file
            with pd.ExcelWriter("processed_defect_data_with_pivot.xlsx") as writer:
                df.to_excel(writer, sheet_name="Processed Data", index=False)
                pivot_df.to_excel(writer, sheet_name="Defect Distribution Analysis")
          
            # Option to download the file
            with open("processed_defect_data_with_pivot.xlsx", "rb") as file:
                st.download_button(
                    label="Download Processed Data",
                    data=file,
                    file_name="processed_defect_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.subheader("How to use:")
        st.write("1. Drag and drop the details of defect Excel or CSV file on the sidebar.")
        st.write("2. Select the columns for Product Name, Lot Number, Serial Number, and Reworking Issue Number.")
        st.write("3. Click on the Process Data button.")
        st.write("4. Scroll down and click on the Download Processed Data button to download the processed Excel file.")
        st.subheader("Download sample Excel template:")
        with open("PrimeReAssy/template.xlsx", "rb") as file:
            st.download_button(
                label="Download Excel Template",
                data=file,
                file_name="template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
