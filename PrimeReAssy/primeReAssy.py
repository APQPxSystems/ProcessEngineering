import streamlit as st
import pandas as pd

# Streamlit Configurations
st.set_page_config(page_title="PRIME/REASSY Identifier | KentK.", layout="wide")
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
    df = df.sort_values(by=[product_col, lot_col, serial_col, issue_col])
    df['Prime/Reassy'] = None  # Initialize column
    
    # Identify each unique product by combining Product Number, Lot Number, and Serial Number
    df['Unique Identifier'] = "PN: " + df[product_col].astype(str) + " LN: " + df[lot_col].astype(str) + " SN: " + df[serial_col].astype(str)

    # Iterate over each unique product to check Reworking Issue Numbers
    for uid in df['Unique Identifier'].unique():
        product_rows = df[df['Unique Identifier'] == uid]
        prev_issue_number = None
        reassy_count = 0  # Counter for Reassy labels
        is_prime = True   # First set of issues will be Prime
        
        for idx, row in product_rows.iterrows():
            current_issue_number = row[issue_col]
            
            # Check if the current issue number is consecutive from the previous
            if prev_issue_number is None or current_issue_number == prev_issue_number + 1:
                # If this is the first set, mark as Prime; otherwise, mark as current Reassy
                if is_prime:
                    df.at[idx, 'Prime/Reassy'] = 'Prime'
                else:
                    df.at[idx, 'Prime/Reassy'] = f'Reassy{reassy_count}'
            else:
                # If non-consecutive, start a new Reassy label
                reassy_count += 1
                df.at[idx, 'Prime/Reassy'] = f'Reassy{reassy_count}'
                is_prime = False  # Subsequent sets are all Reassy

            prev_issue_number = current_issue_number
    
    return df

# Streamlit app layout
st.title("Defect Analysis AI: Prime and Reassy Identifier")
password = st.text_input("Input app key to continue:")
if password == "kent":
  st.write("""This defect analysis tool uses the Reworking Issue Number data to identfy the Prime and Reassy Harness.
          For every unique Harness (identified by the combination of Product Number, Lot Number, and Serial Number),
          the Reworking Issue Number will be analyzed. The first occurrence of consecutive Reworking Issue Number/s will
          be identified as PRIME and the next occurrences will be REASSY. This would help us to identify the number of defects
          per inspection more accurately instead of relying to the defect detection date. The app can also identify 
          the number of times that each harness undergo REASSY (labeled as Reassy1, Reassy2, Reassy3, and so on) and the number of
          NG everytime the harness is inspected.
          """)
  
  with st.sidebar:
      st.title("ProcessEngineering")
      st.write("_________________________")
      uploaded_file = st.file_uploader("Upload your defect data Excel file", type=["xlsx"])
  
  if uploaded_file:
      # Load data
      df = pd.read_excel(uploaded_file)
      st.write("Uploaded Data:")
      st.dataframe(df)
      
      # Column selection
      product_col = st.selectbox("Select the Product Name column:", df.columns)
      lot_col = st.selectbox("Select the Lot Number column:", df.columns)
      serial_col = st.selectbox("Select the Serial Number column:", df.columns)
      issue_col = st.selectbox("Select the Reworking Issue Number column:", df.columns)
  
      if st.button("Process Data"):
          # Process data to add Unique Identifier and Prime/Reassy columns
          df = determine_status(df, product_col, lot_col, serial_col, issue_col)

          # Display processed data
          st.write("Processed Data with Prime/Reassy Status:")
          st.dataframe(df)

          # Create the pivot table and add a total count column
          pivot_df = df.pivot_table(index='Unique Identifier', columns='Prime/Reassy', aggfunc='size', fill_value=0)
          pivot_df['Total Count'] = pivot_df.sum(axis=1)

          # Rename columns to add "NG" to each Prime/Reassy label
          pivot_df = pivot_df.rename(columns={col: f"{col} NG" for col in pivot_df.columns if col not in ['Total Count']})

          # Add a new column for the number of repairs, counting non-zero values excluding 'Total Count'
          pivot_df['Times Repaired'] = (pivot_df.loc[:, pivot_df.columns != 'Total Count'] > 0).sum(axis=1)

          # Sort pivot table by No. of Repair in descending order
          pivot_df = pivot_df.sort_values(by='Times Repaired', ascending=False)

          # Display pivot table
          st.write("Defect Distribution Analysis (Pivot Table):")
          st.dataframe(pivot_df)
          
          # Save both sheets to an Excel file for download
          with pd.ExcelWriter("processed_defect_data_with_pivot.xlsx") as writer:
              df.to_excel(writer, sheet_name="Processed Data", index=False)
              pivot_df.to_excel(writer, sheet_name="Defect Distribution Analysis")
          
          # Option to download the modified data and pivot table as an Excel file
          with open("processed_defect_data_with_pivot.xlsx", "rb") as file:
              st.download_button(
                  label="Download Processed Data",
                  data=file,
                  file_name="processed_defect_data.xlsx",
                  mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
              )
  else:
      st.subheader("How to use:")
      st.write("1. Drag and drop the details of defect Excel file on the sidebar.")
      st.write("2. Select the columns for Product Name, Lot Number, Serial Number, and Reworking Issue Number.")
      st.write("3. Click on the Process Data button.")
      st.write("4. Scroll down and click on the Download Processed Data button to download the processed Excel file.")
      st.subheader("Download sample Excel template:")
      file_path = "PrimeReAssy/template.xlsx"
      with open(file_path, "rb") as file:
          st.download_button(
              label="Download Excel Template",
              data=file,
              file_name="template.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          )
