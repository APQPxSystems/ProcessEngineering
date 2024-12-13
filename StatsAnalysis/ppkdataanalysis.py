import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Streamlit app title
st.title("PPK Data Analysis")

# File upload widget
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    # Load the Excel file
    excel_data = pd.ExcelFile(uploaded_file)

    # Get sheet names from the Excel file
    sheet_names = excel_data.sheet_names

    # User selects a sheet to analyze
    sheet_name = st.selectbox("Select a sheet to analyze", sheet_names)
    
    # Read the selected sheet into a DataFrame
    raw_data = excel_data.parse(sheet_name)

    # Visualization selection
    visualization = "Normal Curve with Histogram"

    # If the user selects 'Normal Curve with Histogram'
    if visualization == "Normal Curve with Histogram":
        usl = st.number_input("Specify Upper Specification Limit (USL)", min_value=1)
        lsl = st.number_input("Specify Lower Specification Limit (LSL)", min_value=1)
        bins = st.number_input("Specify number of histogram bins", min_value=1)

        st.write("Normal Curve with Histogram:")

        # Calculate the median line as the average of USL and LSL
        median_line = (usl + lsl) / 2

        # Loop through columns of the selected sheet (data)
        for column in raw_data.columns:
            # Only consider numeric columns for histograms
            if pd.api.types.is_numeric_dtype(raw_data[column]):
                # Create a new figure and axes for each plot
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot histogram
                ax.hist(raw_data[column], bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black', label='Histogram')

                # Fit a normal distribution to the data
                mu, sigma = stats.norm.fit(raw_data[column])
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mu, sigma)

                # Plot the normal curve
                ax.plot(x, p, 'k', linewidth=2, label='Normal Curve')

                # Plot vertical lines for mean (green), USL, LSL (red), +/- 3 sigma (blue), and median line (orange)
                ax.axvline(mu, color='green', linestyle='--', label='Mean')  # Mean in green
                ax.axvline(usl, color='red', linestyle='--', label='USL')    # USL in red
                ax.axvline(lsl, color='red', linestyle='--', label='LSL')    # LSL in red
                ax.axvline(median_line, color='orange', linestyle='-', label='Median Line (Avg of USL & LSL)')  # Median Line in orange
                ax.axvline(mu + 3*sigma, color='blue', linestyle=':', label='+3 Sigma')  # +3 Sigma in blue
                ax.axvline(mu - 3*sigma, color='blue', linestyle=':', label='-3 Sigma')  # -3 Sigma in blue

                ax.set_xlabel(column)
                ax.set_ylabel('Density')
                ax.legend()

                # Show plot
                st.pyplot(fig)

                # Analysis for the description
                bias = ((mu - median_line) / (usl - lsl)) * 100
                within_tolerance = (mu - 3 * sigma >= lsl) and (mu + 3 * sigma <= usl)
                lower_outside = mu - 3 * sigma < lsl
                upper_outside = mu + 3 * sigma > usl

                # Generate the detailed description
                description = f"""
                - Mean is biased {'high' if bias > 0 else 'low'} by {abs(bias):.2f}% relative to the tolerance range.
                - The +/- 3 sigma range is {'inside' if within_tolerance else 'outside'} the tolerance range.
                """
                if not within_tolerance:
                    if lower_outside:
                        description += "- The lower 3 sigma limit is outside the tolerance range.\n"
                    if upper_outside:
                        description += "- The upper 3 sigma limit is outside the tolerance range.\n"

                # Display the detailed description
                st.write(f"### Analysis for {column}")
                st.write(description)

# Apply custom CSS (optional)
with open('StatsAnalysis/style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
