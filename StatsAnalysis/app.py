# Statistical Analysis Tool
# For Analysis of Design of Experiments Data
# By Kent Katigbak -- Systems Engineering/ Process Engineering Staff

# Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Streamlit Configurations
st.set_page_config(page_title="PE-StatsAnalysis | KentK.", layout="wide")
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

st.set_option('deprecation.showPyplotGlobalUse', False)

# Sidebar Configurations
with st.sidebar:
    st.write("## Stats Analysis Tool for DFA")
    st.write("## How to Use:")
    st.write("#### 1. Select an action: Perform a statistical test, visualize your data, check assy boards.")
    st.write("#### 2. Upload your data using the requested file type.")
    st.write("#### 3. Select a test to perform or a visualization tool to use.")
    st.write("#### 4. Fill the required inputs if there are any.")
    st.write("__________________________________")
    st.write("### Kent Katigbak | Process Engineering")

# App Title and Description
st.write("# STATISTICAL ANALYSIS TOOL FOR DFA")
st.write("""#### This statistical analysis tool is specialized for the analysis of datasets gathered from Process Engineering's Design of Experiments and the Quality Improvement Team's Defect Failure Analysis.""")
st.write("__________________________________________")

# Action Selection
action = st.selectbox("What do you want to do:", ["Perform a statistical test", "Visualize data", "Check Assy Boards"])

# Perform Statistical Test
if action == "Perform a statistical test":
    
    # File Uploader
    raw_data = st.file_uploader("Please upload the CSV file of your DOE dataset", type="csv")
    if raw_data is not None:
        raw_data = pd.read_csv(raw_data)
        st.write("#### Preview of DOE Dataset")
        st.dataframe(raw_data)
        st.write("_________________________________________________")
    
        test = st.selectbox("Select a statistical test:", ["One Sample t Test", "Two Sample t Test", "Paired t Test", "ANOVA", "Chi Square"])
        
        # One Sample t Test
        if test == "One Sample t Test":
            test_column = st.selectbox("Select a column to test.", raw_data.columns)
            population_mean = st.number_input("Population Mean:")
            
            # Extract column data
            sample_data = raw_data[test_column]
            
            # Perform one sample t test
            t_statistic, p_value, = stats.ttest_1samp(sample_data, population_mean)
            
            # Descriptive statistics
            sample_mean = sample_data.mean()
            sample_std = sample_data.std(ddof=1)
            sample_size = len(sample_data)
            
            # Calculate the confidence interval (95%)
            confidence_level = 0.95
            degrees_freedom = sample_size - 1
            confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, stats.sem(sample_data))

            # Determine if we reject the null hypothesis
            alpha = 0.05  # Significance level
            reject_null = p_value < alpha

            # Display the results
            st.write("#### Descriptive Statistics")
            st.write(f"{test_column} Sample Size: {sample_size}")
            st.write(f"{test_column} Sample Mean: {sample_mean:.2f}")
            st.write(f"{test_column} Sample Standard Deviation: {sample_std:.2f}")

            st.write("#### One Sample t-Test Results")
            st.write(f"t-Statistic: {t_statistic:.2f}")
            st.write(f"p-Value: {p_value:.4f}")
            st.write(f"Confidence Interval: {confidence_interval}")

            if reject_null:
                st.write("#### Decision: Reject the null hypothesis (The sample mean is significantly different from the population mean at α = 0.05).")
            else:
                st.write("#### Decision: Fail to reject the null hypothesis (The sample mean is not significantly different from the population mean at α = 0.05).")
                    
        # Two Sample t Test
        if test == "Two Sample t Test":
            test_column1 = st.selectbox("Select the first column to test.", raw_data.columns)
            test_column2 = st.selectbox("Select the second column to test.", raw_data.columns)
            
            # Ensure the two columns are not the same
            if test_column1 == test_column2:
                st.error("You cannot test the same column!")
            else:
                # Extract the column data
                sample_data1 = raw_data[test_column1]
                sample_data2 = raw_data[test_column2]
                
                # Perform the Two Sample t Test
                t_statistic, p_value = stats.ttest_ind(sample_data1, sample_data2)
                
                # Calculate descriptive statistics for both samples
                sample1_mean = sample_data1.mean()
                sample1_std = sample_data1.std(ddof=1)
                sample1_size = len(sample_data1)

                sample2_mean = sample_data2.mean()
                sample2_std = sample_data2.std(ddof=1)
                sample2_size = len(sample_data2)

                # Determine if we reject the null hypothesis
                alpha = 0.05  # Significance level
                reject_null = p_value < alpha

                # Display the results
                st.write("#### Descriptive Statistics")
                st.write(f"{test_column1} Sample Size: {sample1_size}")
                st.write(f"{test_column1} Sample Mean: {sample1_mean:.2f}")
                st.write(f"{test_column1} Sample Standard Deviation: {sample1_std:.2f}")
                st.write(f"{test_column2} Sample Size: {sample2_size}")
                st.write(f"{test_column2} Sample Mean: {sample2_mean:.2f}")
                st.write(f"{test_column2} Sample Standard Deviation: {sample2_std:.2f}")

                st.write("#### Two Sample t-Test Results")
                st.write(f"t-Statistic: {t_statistic:.2f}")
                st.write(f"p-Value: {p_value:.4f}")

                if reject_null:
                    st.write("#### Decision: Reject the null hypothesis (The means of the two samples are significantly different at α = 0.05).")
                else:
                    st.write("#### Decision: Fail to reject the null hypothesis (The means of the two samples are not significantly different at α = 0.05).")
        
        # Paired t Test
        if test == "Paired t Test":
            test_column1 = st.selectbox("Select the first column to test.", raw_data.columns)
            test_column2 = st.selectbox("Select the second column to test.", raw_data.columns)
            
            # Ensure the selected columns are not the same
            if test_column1 == test_column2:
                st.error("You cannot test the same column.")
            else:
                # Extract the column data
                sample_data1 = raw_data[test_column1]
                sample_data2 = raw_data[test_column2]
                
                # Perform the Paired Sample t-Test
                t_statistic, p_value = stats.ttest_rel(sample_data1, sample_data2)

                # Calculate descriptive statistics for both samples
                sample1_mean = sample_data1.mean()
                sample1_std = sample_data1.std(ddof=1)
                sample1_size = len(sample_data1)

                sample2_mean = sample_data2.mean()
                sample2_std = sample_data2.std(ddof=1)
                sample2_size = len(sample_data2)

                # Determine if we reject the null hypothesis
                alpha = 0.05  # Significance level
                reject_null = p_value < alpha

                # Display the results
                st.write("#### Descriptive Statistics")
                st.write(f"{test_column1} Sample Size: {sample1_size}")
                st.write(f"{test_column1} Sample Mean: {sample1_mean:.2f}")
                st.write(f"{test_column1} Sample Standard Deviation: {sample1_std:.2f}")
                st.write(f"{test_column2} Sample Size: {sample2_size}")
                st.write(f"{test_column2} Sample Mean: {sample2_mean:.2f}")
                st.write(f"{test_column2} Sample Standard Deviation: {sample2_std:.2f}")

                st.write("#### Paired Sample t-Test Results")
                st.write(f"t-Statistic: {t_statistic:.2f}")
                st.write(f"p-Value: {p_value:.4f}")

                if reject_null:
                    st.write("#### Decision: Reject the null hypothesis (There is a significant difference between the paired samples at α = 0.05).")
                else:
                    st.write("#### Decision: Fail to reject the null hypothesis (There is no significant difference between the paired samples at α = 0.05).")
        
        # ANOVA
        if test == "ANOVA":
            columns = st.multiselect("Select columns to test.", raw_data.columns)
            
            # Ensure that at least three columns are selected
            if len(columns) < 3:
                st.error("Please select three or more columns.")
            else:
                # Extract the column data
                sample_data = [raw_data[column] for column in columns]
                
                # Perform ANOVA
                f_statistic, p_value = stats.f_oneway(*sample_data)
                
                # Calculate descriptive statistics for each sample
                descriptive_stats = {}
                for column in columns:
                    descriptive_stats[column] = {
                        "mean": raw_data[column].mean(),
                        "std": raw_data[column].std(ddof=1),
                        "size": len(raw_data[column])
                    }

                # Determine if we reject the null hypothesis
                alpha = 0.05  # Significance level
                reject_null = p_value < alpha

                # Display the results
                st.write("#### Descriptive Statistics")
                for column, stats in descriptive_stats.items():
                    st.write(f"Column: {column}")
                    st.write(f"Size: {stats['size']}")
                    st.write(f"Mean: {stats['mean']:.2f}")
                    st.write(f"Standard Deviation: {stats['std']:.2f}")
                    st.write("---")

                st.write("#### ANOVA Test Results")
                st.write(f"F-Statistic: {f_statistic:.2f}")
                st.write(f"p-Value: {p_value:.4f}")

                if reject_null:
                    st.write("#### Decision: Reject the null hypothesis (At least one sample mean is significantly different from the others at α = 0.05).")
                else:
                    st.write("#### Decision: Fail to reject the null hypothesis (No significant difference between the sample means at α = 0.05).")
        
        # Chi Square Test
        if test == "Chi Square":
            pass
        
    else:
        st.write("#### No dataset has been uploaded.")

# Visualize data
if action == "Visualize data":
    
    # File Uploader
    raw_data = st.file_uploader("Please upload the CSV file of your DOE dataset", type="csv")
    if raw_data is not None:
        raw_data = pd.read_csv(raw_data)
        st.write("#### Preview of DOE Dataset")
        st.dataframe(raw_data)
        st.write("_________________________________________________")    
    
        visualization = st.selectbox("Select a visualization tool:", ["Histogram", "Normal Curve", 
                                                                    "Normal Curve with Histogram", "Normal Curve with ± 3 Sigma",
                                                                    "Scatter Plot", "NC with ± 3 Sigma and Scatter", "Scatter Plot with ± 3 Sigma", 
                                                                    "Box Plot", "Box Plot with Limits", "Pearson Correlation"])
        
        # Histogram
        if visualization == "Histogram":
            st.write("#### Generated Histogram/s")
            bins = st.number_input("Specify number of histogram bins", min_value=1)
            for column in raw_data.columns:
                plt.figure(figsize=(8, 6))
                plt.hist(raw_data[column], bins=bins, edgecolor='black')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                st.pyplot()

        # Normal Curve
        if visualization == "Normal Curve":
            usl = st.number_input("Specify Upper Specification Limit (USL)", min_value=1)
            lsl = st.number_input("Specify Lower Specification Limit (LSL)", min_value=1)

            st.write("Normal Curves with Mean, USL, LSL:")
            for column in raw_data.columns:
                plt.figure(figsize=(8, 6))

                # Fit a normal distribution to the data
                mu, sigma = stats.norm.fit(raw_data[column])
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mu, sigma)

                # Plot the normal curve
                plt.plot(x, p, 'k', linewidth=2, label='Normal Curve')  

                # Plot vertical lines for mean, USL, LSL
                plt.axvline(mu, color='blue', linestyle='--', label='Mean')
                plt.axvline(usl, color='red', linestyle='--', label='USL')
                plt.axvline(lsl, color='green', linestyle='--', label='LSL')

                plt.xlabel(column)
                plt.ylabel('Density')
                plt.legend()

                # Show plot
                st.pyplot()

        # Normal Curve with Histogram
        if visualization == "Normal Curve with Histogram":
            usl = st.number_input("Specify Upper Specification Limit (USL)", min_value=1)
            lsl = st.number_input("Specify Lower Specification Limit (LSL)", min_value=1)
            bins = st.number_input("Specify number of histogram bins", min_value=1)
            st.write("Histograms with Normal Curve, Mean, USL, LSL:")
            for column in raw_data.columns:
                plt.figure(figsize=(10, 6))

                # Plot histogram
                plt.hist(raw_data[column], bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black', label='Histogram')

                # Fit a normal distribution to the data
                mu, sigma = stats.norm.fit(raw_data[column])
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mu, sigma)

                # Plot the normal curve
                plt.plot(x, p, 'k', linewidth=2, label='Normal Curve')

                # Plot vertical lines for mean, USL, LSL
                plt.axvline(mu, color='red', linestyle='--', label='Mean')
                plt.axvline(usl, color='green', linestyle='--', label='USL')
                plt.axvline(lsl, color='orange', linestyle='--', label='LSL')

                plt.xlabel(column)
                plt.ylabel('Density')
                plt.legend()

                # Show plot
                st.pyplot()

        # Normal Curve with ± 3 Sigma
        if visualization == "Normal Curve with ± 3 Sigma":
            usl = st.number_input("Specify Upper Specification Limit (USL)", min_value=1)
            lsl = st.number_input("Specify Lower Specification Limit (LSL)", min_value=1)

            # Calculate the median
            median = (usl + lsl) / 2

            # Calculate global xmin and xmax
            global_min = float('inf')
            global_max = float('-inf')
            
            good_count = 0
            no_good_count = 0
            good_columns = []
            no_good_columns = []

            for col in raw_data.columns:
                mean = raw_data[col].mean()
                std_dev = raw_data[col].std()
                col_min = mean - 4 * std_dev
                col_max = mean + 4 * std_dev
                if col_min < global_min:
                    global_min = col_min
                if col_max > global_max:
                    global_max = col_max

                # Determine if the column is GOOD or NO GOOD
                if (mean - 3 * std_dev >= lsl) and (mean + 3 * std_dev <= usl):
                    good_count += 1
                    good_columns.append(col)
                else:
                    no_good_count += 1
                    no_good_columns.append(col)

            # Display counts
            st.write(f"GOOD: {good_count}, NO GOOD: {no_good_count}")
            
            # Display tables of GOOD and NO GOOD columns with index starting from 1
            g_col, ng_col = st.columns([1,1])

            with g_col: 
                good_df = pd.DataFrame(good_columns, columns=["LIST OF GOOD"])
                good_df.index += 1
                st.table(good_df)

            with ng_col:
                no_good_df = pd.DataFrame(no_good_columns, columns=["LIST OF NO GOOD"])
                no_good_df.index += 1
                st.table(no_good_df)

            st.write("_______________________________________")
            # Display normal distribution with reference lines for each column
            st.write("Normal Distribution with ± 3 Sigma Reference Lines for Each Column:")

            for col in raw_data.columns:
                plt.figure(figsize=(8, 6))

                # Fit a normal distribution to the data
                mean = raw_data[col].mean()
                std_dev = raw_data[col].std()

                # Plot normal distribution
                x = np.linspace(global_min, global_max, 100)
                p = stats.norm.pdf(x, mean, std_dev)
                plt.plot(x, p, label='Normal Curve')

                # Add reference lines for USL, LSL, ±3 sigma, and median
                for val, color, label in [(usl, 'r', 'USL'),
                                            (lsl, 'r', 'LSL'),
                                            (mean + 3 * std_dev, 'b', '+3σ'),
                                            (mean - 3 * std_dev, 'b', '-3σ'),
                                            (median, 'g', 'Median')]:
                    plt.axvline(x=val, color=color, linestyle='--', label=label)

                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.title(f"Normal Distribution - {col}")
                plt.legend()
                st.pyplot(plt)

        # Scatter Plot
        if visualization == "Scatter Plot":
            st.write("Scatter Plots for all Pair Combinations:")
            columns = raw_data.columns
            num_cols = len(columns)

            for i in range(num_cols):
                for j in range(i + 1, num_cols):
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=raw_data, x=columns[i], y=columns[j])
                    plt.xlabel(columns[i])
                    plt.ylabel(columns[j])
                    plt.title(f"{columns[i]} vs {columns[j]}")
                    st.pyplot()

        # Normal Curve with ± 3 Sigma and Scatter Plot
        if visualization == "NC with ± 3 Sigma and Scatter":
            usl = st.number_input("Specify Upper Specification Limit (USL)", min_value=1)
            lsl = st.number_input("Specify Lower Specification Limit (LSL)", min_value=1)
            # Display scatter plot with normal distribution and reference lines for each column
            st.write("Scatter Plot with Normal Distribution and Reference Lines for Each Column:")

            for col in raw_data.columns:
                plt.figure(figsize=(10, 6))

                # Scatter plot of actual data points
                plt.scatter(raw_data.index, raw_data[col], color='skyblue', label='Data Points')

                # Fit a normal distribution to the data
                mean = raw_data[col].mean()
                std_dev = raw_data[col].std()

                # Plot normal distribution
                xmin = mean - 4 * std_dev
                xmax = mean + 4 * std_dev
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mean, std_dev)
                plt.plot(x, p, 'k', label='Normal Curve')

                # Add reference lines for USL, LSL, and ±3 sigma
                for val, color, label in [(usl, 'r', 'USL'),
                                        (lsl, 'g', 'LSL'),
                                        (mean + 3 * std_dev, 'b', '+3σ'),
                                        (mean - 3 * std_dev, 'b', '-3σ')]:
                    plt.axvline(x=val, color=color, linestyle='--', label=label)

                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.title(f"Scatter Plot with Normal Distribution - {col}")
                plt.legend()
                st.pyplot()
                
        # Scatter Plot with ± 3 sigma
        if visualization == "Scatter Plot with ± 3 Sigma":
            usl = st.number_input("Specify Upper Specification Limit (USL)", min_value=1)
            lsl = st.number_input("Specify Lower Specification Limit (LSL)", min_value=1)
            # Display scatter plot with reference lines for ±3 sigma for each column
            st.write("Scatter Plot with ±3 Sigma Reference Lines for Each Column:")

            for col in raw_data.columns:
                plt.figure(figsize=(10, 6))

                # Scatter plot of actual data points
                plt.scatter(raw_data.index, raw_data[col], color='skyblue', label='Data Points')

                # Calculate mean and standard deviation
                mean = raw_data[col].mean()
                std_dev = raw_data[col].std()

                # Add reference lines for ±3 sigma
                upper_3_sigma = mean + 3 * std_dev
                lower_3_sigma = mean - 3 * std_dev

                plt.axhline(y=upper_3_sigma, color='r', linestyle='--', label='+3σ')
                plt.axhline(y=lower_3_sigma, color='g', linestyle='--', label='-3σ')

                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.title(f"Scatter Plot with ±3 Sigma - {col}")
                plt.legend()
                st.pyplot()

        # Box Plot
        if visualization == "Box Plot":
            # Display vertical boxplots for all columns
            st.write("Vertical Boxplots for Each Column:")
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=raw_data)
            plt.ylabel("Value")
            plt.title("Boxplot for Each Column")
            plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
            st.pyplot()

        # Box Plot with Limits
        if visualization == "Box Plot with Limits":
            median = st.number_input("Specify the Median Specification", min_value=1)
            usl = st.number_input("Specify Upper Specification Limit (USL)", min_value=1)
            lsl = st.number_input("Specify Lower Specification Limit (LSL)", min_value=1)

            # Display vertical boxplots with USL and LSL reference lines for all columns
            st.write("Vertical Boxplots with USL and LSL Reference Lines:")
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=raw_data)
            
            # Add Median, USL and LSL reference lines
            plt.axhline(y=median, color='g', linestyle='--', label='Median')
            plt.axhline(y=usl, color='r', linestyle='--', label='USL')
            plt.axhline(y=lsl, color='r', linestyle='--', label='LSL')

            plt.ylabel("Value")
            plt.title("Vertical Boxplot with USL and LSL Reference Lines")
            plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
            plt.legend()
            st.pyplot()

        # Pearson Correlation
        if visualization == "Pearson Correlation":
            corr_matrix = raw_data.corr()

            # Display Pearson correlation table with shading
            st.write("Pearson Correlation Table:")
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            st.pyplot()

    else:
        st.write("#### No dataset has been uploaded.")
        
# Perform Statistical Test
if action == "Check Assy Boards":

    # File Uploader
    raw_data = st.file_uploader("Please upload the Excel file of your DOE dataset", type=["xlsx", "xls"])
    if raw_data is not None:
        try:
            # Load all sheets into a dictionary
            sheets_dict = pd.read_excel(raw_data, sheet_name=None)
            
            # List the sheet names
            sheet_names = list(sheets_dict.keys())
            
            # Selectbox for choosing a sheet to display
            sheet_to_display = st.selectbox("Select the sheet to display", sheet_names)
            
            # Display the selected sheet
            st.write(f"#### Preview of {sheet_to_display} Sheet")
            df = sheets_dict[sheet_to_display]
            st.dataframe(df)
            st.write("_________________________________________________")

            usl = st.number_input("Specify Upper Specification Limit (USL)", min_value=1)
            lsl = st.number_input("Specify Lower Specification Limit (LSL)", min_value=1)

            # Calculate the median
            median = (usl + lsl) / 2

            # Combine all data into a single series
            combined_data = pd.concat([df[col] for col in df.columns])
            
            # Calculate global xmin and xmax for combined data
            mean_combined = combined_data.mean()
            std_dev_combined = combined_data.std()
            global_min = mean_combined - 4 * std_dev_combined
            global_max = mean_combined + 4 * std_dev_combined

            # First section: Combined data normal curve
            st.write("### General Analysis | Line Condition")
            
            plt.figure(figsize=(8, 6))

            # Plot normal distribution for combined data
            x_combined = np.linspace(global_min, global_max, 100)
            p_combined = stats.norm.pdf(x_combined, mean_combined, std_dev_combined)
            plt.plot(x_combined, p_combined, label='Normal Curve')

            # Add reference lines for USL, LSL, ±3 sigma, and median
            for val, color, label in [(usl, 'r', 'USL'),
                                        (lsl, 'r', 'LSL'),
                                        (mean_combined + 3 * std_dev_combined, 'b', '+3σ'),
                                        (mean_combined - 3 * std_dev_combined, 'b', '-3σ'),
                                        (median, 'g', 'Median')]:
                plt.axvline(x=val, color=color, linestyle='--', label=label)

            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.title("Normal Distribution - Combined Data")
            plt.legend()
            st.pyplot(plt)

            # Add conditional description for the median
            range_tolerance = usl - lsl
            bias_percentage_combined = abs(mean_combined - median) / range_tolerance * 100
            median_description_combined = ""
            if mean_combined == median:
                median_description_combined = "The center of data is on the median value."
            elif mean_combined < median:
                median_description_combined = f"The center of data is biased on the short dimension by {bias_percentage_combined:.2f}%."
            else:
                median_description_combined = f"The center of data is biased on the long dimension by {bias_percentage_combined:.2f}%."
            
            # Add conditional description for ±3 Sigma
            plus_3_sigma_combined = mean_combined + 3 * std_dev_combined
            minus_3_sigma_combined = mean_combined - 3 * std_dev_combined
            sigma_description_combined = ""
            if minus_3_sigma_combined >= lsl and plus_3_sigma_combined <= usl:
                sigma_description_combined = "±3 Sigma is inside the tolerance range."
            elif minus_3_sigma_combined < lsl and plus_3_sigma_combined > usl:
                sigma_description_combined = "±3 Sigma both goes beyond the tolerance range."
            elif plus_3_sigma_combined > usl:
                sigma_description_combined = "+3 Sigma goes beyond Maximum Tolerance."
            elif minus_3_sigma_combined < lsl:
                sigma_description_combined = "-3 Sigma goes beyond Minimum Tolerance."

            st.write(median_description_combined)
            st.write(sigma_description_combined)

            st.write("___________________________________________________")

            # Second section: Individual column analysis
            st.write("### Drill-Down Analysis | Individual Boards:")

            good_count = 0
            no_good_count = 0
            good_columns = []
            no_good_columns = []

            for col in df.columns:
                mean = df[col].mean()
                std_dev = df[col].std()
                col_min = mean - 4 * std_dev
                col_max = mean + 4 * std_dev
                if col_min < global_min:
                    global_min = col_min
                if col_max > global_max:
                    global_max = col_max

                # Determine if the column is GOOD or NO GOOD
                if (mean - 3 * std_dev >= lsl) and (mean + 3 * std_dev <= usl):
                    good_count += 1
                    good_columns.append(col)
                else:
                    no_good_count += 1
                    no_good_columns.append(col)

            # Display counts
            st.write(f"GOOD: {good_count}, NO GOOD: {no_good_count}")

            # Display tables of GOOD and NO GOOD columns with index starting from 1
            g_col, ng_col = st.columns([1, 1])

            with g_col: 
                good_df = pd.DataFrame(good_columns, columns=["LIST OF GOOD"])
                good_df.index += 1
                st.table(good_df)

            with ng_col:
                no_good_df = pd.DataFrame(no_good_columns, columns=["LIST OF NO GOOD"])
                no_good_df.index += 1
                st.table(no_good_df)

            st.write("_______________________________________")
            
            # Display normal distribution with reference lines for each column
            st.write("Normal Distribution with ± 3 Sigma Reference Lines for Each Column:")

            columns = st.columns(2)
            for idx, col in enumerate(df.columns):
                with columns[idx % 2]:
                    plt.figure(figsize=(8, 6))

                    # Fit a normal distribution to the data
                    mean = df[col].mean()
                    std_dev = df[col].std()

                    # Plot normal distribution
                    x = np.linspace(global_min, global_max, 100)
                    p = stats.norm.pdf(x, mean, std_dev)
                    plt.plot(x, p, label='Normal Curve')

                    # Add reference lines for USL, LSL, ±3 sigma, and median
                    for val, color, label in [(usl, 'r', 'USL'),
                                                (lsl, 'r', 'LSL'),
                                                (mean + 3 * std_dev, 'b', '+3σ'),
                                                (mean - 3 * std_dev, 'b', '-3σ'),
                                                (median, 'g', 'Median')]:
                        plt.axvline(x=val, color=color, linestyle='--', label=label)

                    plt.xlabel("Value")
                    plt.ylabel("Density")
                    plt.title(f"Normal Distribution - {col}")
                    plt.legend()
                    st.pyplot(plt)

                    # Add conditional description for the median
                    bias_percentage = abs(mean - median) / range_tolerance * 100
                    median_description = ""
                    if mean == median:
                        median_description = "The center of data is on the median value."
                    elif mean < median:
                        median_description = f"The center of data is biased on the short dimension by {bias_percentage:.2f}%."
                    else:
                        median_description = f"The center of data is biased on the long dimension by {bias_percentage:.2f}%."
                    
                    # Add conditional description for ±3 Sigma
                    plus_3_sigma = mean + 3 * std_dev
                    minus_3_sigma = mean - 3 * std_dev
                    sigma_description = ""
                    if minus_3_sigma >= lsl and plus_3_sigma <= usl:
                        sigma_description = "±3 Sigma is inside the tolerance range."
                    elif minus_3_sigma < lsl and plus_3_sigma > usl:
                        sigma_description = "±3 Sigma both goes beyond the tolerance range."
                    elif plus_3_sigma > usl:
                        sigma_description = "+3 Sigma goes beyond Maximum Tolerance."
                    elif minus_3_sigma < lsl:
                        sigma_description = "-3 Sigma goes beyond Minimum Tolerance."

                    st.write(f"#### {col}")
                    st.write(median_description)
                    st.write(sigma_description)
        
        except UnicodeDecodeError as e:
            st.error(f"UnicodeDecodeError: {e}. Please ensure the file is a valid Excel file.")
        except Exception as e:
            st.error(f"An error occurred: {e}. Please ensure the file is a valid Excel file.")
            
    else:
        st.write("#### No dataset has been uploaded.")
      
with open('StatsAnalysis/style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

