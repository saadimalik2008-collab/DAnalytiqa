import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from sqlalchemy import create_engine
import sqlite3

# --- Initial Setup and Configuration ---
st.set_page_config(layout="wide", page_title="Data Analytics Platform", page_icon="📈")

# Initialize session state variables
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'cleaning_log' not in st.session_state:
    st.session_state.cleaning_log = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'task_instructions' not in st.session_state:
    st.session_state.task_instructions = ""
if 'period_data' not in st.session_state:
    st.session_state.period_data = {}

# --- Main App Title and Navigation ---
st.title("🚀 Smart Data Analysis & Reporting Platform")
st.markdown("A multi-step platform for data processing, analysis, visualization, and report generation.")

# --- Tab Definitions ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1. Upload & Instructions", "2. Quick Overview", "3. Data Cleaning", 
    "4. Custom Analysis", "5. Visualizations", "6. Compare Periods",
    "7. SQL Query Interface", "8. Reports & Downloads"
])

# --- Tab 1: Upload & Instructions (Part 1/4) ---
with tab1:
    st.header("Step 1: Upload Data & Set Instructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your data file (CSV, Excel, JSON, Parquet)",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet']
        )
        
        if uploaded_file is not None:
            try:
                # Determine file type and read data
                file_name = uploaded_file.name
                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif file_name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                elif file_name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                else:
                    st.error("Unsupported file format.")
                    df = None

                if df is not None:
                    # Reset cleaned data and log upon new upload
                    st.session_state.uploaded_data = df
                    st.session_state.cleaned_data = None
                    st.session_state.cleaning_log = []
                    st.success(f"✅ Data uploaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                    st.dataframe(df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    with col2:
        st.subheader("📝 Analysis Instructions")
        st.info("Define the objectives for your analysis. This will be included in the final report.")
        
        st.session_state.task_instructions = st.text_area(
            "Enter your analysis objectives or business questions here:",
            value=st.session_state.task_instructions,
            height=200,
            key="task_instructions_input"
        )
        if st.session_state.task_instructions:
            st.success("Task instructions set.")

# --- Tab 2: Quick Overview (Part 1/4) ---
with tab2:
    st.header("Step 2: Quick Overview")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data in the 'Upload & Instructions' tab first.")
    else:
        df = st.session_state.uploaded_data
        st.subheader("Dataset Summary")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Duplicate Rows", df.duplicated().sum())

        st.subheader("Column Information")
        # Get data types and missing values
        data_info = pd.DataFrame({
            'Dtype': df.dtypes,
            'Non-Null Count': df.count(),
            'Missing Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        data_info['Missing %'] = (data_info['Missing Count'] / len(df)) * 100
        data_info = data_info.sort_values(by='Missing Count', ascending=False)
        st.dataframe(data_info, use_container_width=True)
        
        st.subheader("Statistical Description (Numeric Columns)")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().T.style.format('{:.2f}'), use_container_width=True)
        else:
            st.info("No numeric columns found for statistical description.")

# --- Tab 3: Data Cleaning (Part 2/4) ---
with tab3:
    st.header("Step 3: Data Cleaning & Preprocessing")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data in the 'Upload & Instructions' tab first.")
    else:
        df = st.session_state.uploaded_data.copy()
        
        if st.session_state.cleaned_data is not None:
            df = st.session_state.cleaned_data.copy()

        st.info(f"Current Data: {df.shape[0]} rows, {df.shape[1]} columns.")
        
        st.subheader("Cleaning Operations")
        
        # 1. Handle Missing Values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            st.write("---")
            st.write("### 1. Missing Values Imputation/Removal")
            col_to_impute = st.selectbox("Select column for missing value handling:", missing_cols, key="impute_col")
            
            if col_to_impute:
                missing_count = df[col_to_impute].isnull().sum()
                st.info(f"Column '{col_to_impute}' has **{missing_count}** missing values.")
                
                # Check column data type for appropriate imputation methods
                if df[col_to_impute].dtype in ['int64', 'float64']:
                    method = st.radio("Select method:", ["Fill with Mean", "Fill with Median", "Remove Rows"], key="missing_numeric_method")
                    if st.button(f"Apply {method} to {col_to_impute}", key="apply_impute_numeric"):
                        original_count = len(df)
                        if method == "Fill with Mean":
                            fill_value = df[col_to_impute].mean()
                            df[col_to_impute] = df[col_to_impute].fillna(fill_value)
                            st.session_state.cleaning_log.append(f"• Filled {missing_count} missing values in '{col_to_impute}' with the Mean ({fill_value:.2f}).")
                        elif method == "Fill with Median":
                            fill_value = df[col_to_impute].median()
                            df[col_to_impute] = df[col_to_impute].fillna(fill_value)
                            st.session_state.cleaning_log.append(f"• Filled {missing_count} missing values in '{col_to_impute}' with the Median ({fill_value:.2f}).")
                        elif method == "Remove Rows":
                            df.dropna(subset=[col_to_impute], inplace=True)
                            rows_removed = original_count - len(df)
                            st.session_state.cleaning_log.append(f"• Removed {rows_removed} rows where '{col_to_impute}' was missing.")
                        
                        st.session_state.cleaned_data = df
                        st.success(f"✅ Operation applied. New rows: {len(df)}")
                        st.experimental_rerun()
                        
                else: # Categorical or Object type
                    method = st.radio("Select method:", ["Fill with Mode", "Fill with Custom Value", "Remove Rows"], key="missing_categorical_method")
                    if st.button(f"Apply {method} to {col_to_impute}", key="apply_impute_categorical"):
                        original_count = len(df)
                        if method == "Fill with Mode":
                            fill_value = df[col_to_impute].mode()[0]
                            df[col_to_impute] = df[col_to_impute].fillna(fill_value)
                            st.session_state.cleaning_log.append(f"• Filled {missing_count} missing values in '{col_to_impute}' with the Mode ({fill_value}).")
                        elif method == "Fill with Custom Value":
                            custom_value = st.text_input("Enter custom fill value:", key="custom_fill_value")
                            if custom_value:
                                df[col_to_impute] = df[col_to_impute].fillna(custom_value)
                                st.session_state.cleaning_log.append(f"• Filled {missing_count} missing values in '{col_to_impute}' with custom value '{custom_value}'.")
                        elif method == "Remove Rows":
                            df.dropna(subset=[col_to_impute], inplace=True)
                            rows_removed = original_count - len(df)
                            st.session_state.cleaning_log.append(f"• Removed {rows_removed} rows where '{col_to_impute}' was missing.")
                            
                        st.session_state.cleaned_data = df
                        st.success(f"✅ Operation applied. New rows: {len(df)}")
                        st.experimental_rerun()
        else:
            st.success("✅ No missing values found in the current dataset.")

        # 2. Handle Duplicates
        st.write("---")
        st.write("### 2. Duplicate Rows")
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            st.warning(f"Found **{duplicate_count}** duplicate rows.")
            if st.button("🧼 Remove Duplicate Rows"):
                df.drop_duplicates(inplace=True)
                st.session_state.cleaning_log.append(f"• Removed **{duplicate_count}** duplicate rows.")
                st.session_state.cleaned_data = df
                st.success(f"✅ Duplicates removed. New rows: {len(df)}")
                st.experimental_rerun()
        else:
            st.info("No duplicate rows found.")

        # 3. Data Type Conversion
        st.write("---")
        st.write("### 3. Data Type Conversion")
        cols_to_convert = df.columns.tolist()
        col_to_change = st.selectbox("Select column to change data type:", cols_to_convert, key="col_to_change")
        
        if col_to_change:
            current_dtype = str(df[col_to_change].dtype)
            new_dtype = st.selectbox(
                f"Current Dtype: {current_dtype}. Select new Dtype:",
                ['int', 'float', 'str', 'datetime'],
                key="new_dtype"
            )
            
            if st.button("🔄 Apply Dtype Conversion"):
                try:
                    original_dtype = str(df[col_to_change].dtype)
                    if new_dtype == 'datetime':
                        # Try to infer format for robust conversion
                        df[col_to_change] = pd.to_datetime(df[col_to_change], errors='coerce', infer_datetime_format=True)
                    elif new_dtype == 'str':
                        df[col_to_change] = df[col_to_change].astype(str)
                    elif new_dtype == 'int':
                        # Convert to float first to handle NaNs, then to int
                        df[col_to_change] = pd.to_numeric(df[col_to_change], errors='coerce').astype(float).astype(pd.Int64Dtype())
                    elif new_dtype == 'float':
                        df[col_to_change] = pd.to_numeric(df[col_to_change], errors='coerce')

                    st.session_state.cleaned_data = df
                    st.session_state.cleaning_log.append(f"• Converted data type of column '{col_to_change}' from {original_dtype} to {new_dtype}.")
                    st.success(f"✅ Dtype converted to {new_dtype}. New Dtype: {df[col_to_change].dtype}")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Conversion failed: {str(e)}. Ensure the column data is compatible with the target type.")

        st.write("---")
        st.subheader("Cleaning Log")
        if st.session_state.cleaning_log:
            for log in st.session_state.cleaning_log:
                st.code(log)
        else:
            st.info("No cleaning operations performed yet.")

# --- Tab 4: Custom Analysis (Part 2/4) ---
with tab4:
    st.header("Step 4: Custom Analysis & Aggregation")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first.")
    else:
        # Use cleaned data if available, otherwise use uploaded data
        df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.uploaded_data
        
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.subheader("📈 Descriptive Statistics by Group")
        
        col1, col2 = st.columns(2)
        with col1:
            group_col = st.selectbox("Group By Column (Categorical):", [None] + categorical_cols, key="group_by_col")
        with col2:
            value_col = st.selectbox("Value Column (Numeric):", numeric_cols, key="value_col")
            
        if group_col and value_col:
            agg_type = st.radio("Select Aggregation:", ["Count", "Mean", "Median", "Sum", "Min", "Max"], horizontal=True)
            
            if st.button("Calculate Grouped Statistics", type="primary"):
                try:
                    if agg_type == "Count":
                        result = df.groupby(group_col)[value_col].count().reset_index(name='Count')
                        title = f"Count of {value_col} by {group_col}"
                    else:
                        agg_func = {'Mean': 'mean', 'Median': 'median', 'Sum': 'sum', 'Min': 'min', 'Max': 'max'}[agg_type]
                        result = df.groupby(group_col)[value_col].agg(agg_func).reset_index(name=agg_type)
                        title = f"{agg_type} of {value_col} by {group_col}"
                    
                    st.success(f"✅ Calculation complete: {title}")
                    st.dataframe(result, use_container_width=True)
                    st.session_state.analysis_results[title] = result # Store for potential report
                    
                    # Visualization of Grouped Result
                    fig = px.bar(result, x=group_col, y=result.columns[-1], title=title)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during aggregation: {str(e)}")
        else:
            st.info("Select a Group By column (Categorical) and a Value column (Numeric) to perform grouped analysis.")

        st.write("---")
        st.subheader("🔍 Filter Data")
        
        filter_col = st.selectbox("Select column to filter:", all_cols, key="filter_col")
        
        if filter_col:
            col_type = df[filter_col].dtype
            
            if col_type in ['object', 'category']:
                # Categorical filter
                unique_values = df[filter_col].unique().tolist()
                selected_values = st.multiselect("Select values to keep:", unique_values, key="filter_values")
                
                if st.button("Apply Categorical Filter"):
                    if selected_values:
                        filtered_df = df[df[filter_col].isin(selected_values)]
                        st.success(f"✅ Filtered data: {len(filtered_df)} rows remaining.")
                        st.dataframe(filtered_df.head(10), use_container_width=True)
                        if st.button("💾 Save Filtered Data as Cleaned Data"):
                            st.session_state.cleaned_data = filtered_df
                            st.session_state.cleaning_log.append(f"• Filtered data: Kept rows where '{filter_col}' is one of {selected_values}.")
                            st.success("Filtered data saved to cleaned dataset.")
                            st.experimental_rerun()
            
            elif col_type in ['int64', 'float64']:
                # Numeric filter
                col1, col2 = st.columns(2)
                with col1:
                    operator = st.selectbox("Operator:", ['>', '<', '==', '>=', '<=', 'between'], key="numeric_operator")
                with col2:
                    if operator != 'between':
                        filter_value = st.number_input("Value:", key="numeric_value")
                    else:
                        min_val = st.number_input("Min Value:", key="numeric_min")
                        max_val = st.number_input("Max Value:", key="numeric_max")

                if st.button("Apply Numeric Filter"):
                    if operator == '>':
                        filtered_df = df[df[filter_col] > filter_value]
                    elif operator == '<':
                        filtered_df = df[df[filter_col] < filter_value]
                    elif operator == '==':
                        filtered_df = df[df[filter_col] == filter_value]
                    elif operator == '>=':
                        filtered_df = df[df[filter_col] >= filter_value]
                    elif operator == '<=':
                        filtered_df = df[df[filter_col] <= filter_value]
                    elif operator == 'between':
                        filtered_df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]
                        
                    st.success(f"✅ Filtered data: {len(filtered_df)} rows remaining.")
                    st.dataframe(filtered_df.head(10), use_container_width=True)
                    if st.button("💾 Save Filtered Data as Cleaned Data", key="save_filtered_numeric"):
                        st.session_state.cleaned_data = filtered_df
                        st.session_state.cleaning_log.append(f"• Filtered data: Kept rows where '{filter_col}' {operator} {filter_value if operator != 'between' else f'is between {min_val} and {max_val}'}.")
                        st.success("Filtered data saved to cleaned dataset.")
                        st.experimental_rerun()
            
            else:
                st.info(f"Filtering for data type {col_type} is not yet supported.")

# --- Tab 5: Visualizations (Part 3/4) ---
with tab5:
    st.header("Step 5: Interactive Visualizations")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first.")
    else:
        df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.uploaded_data
        
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart", "Heatmap (Correlation)", "Multi-Chart Dashboard"],
            key="viz_type_select"
        )
        
        st.divider()
        
        if viz_type == "Line Chart":
            st.subheader("📈 Line Chart (Trend Analysis)")
            if len(numeric_cols) > 0:
                # Prioritize date columns for x-axis, otherwise use all columns
                x_options = date_cols + [c for c in all_cols if c not in date_cols]
                
                x_col = st.selectbox("X-axis (Trend/Date):", x_options, key="line_x")
                y_col = st.selectbox("Y-axis (Value):", numeric_cols, key="line_y")
                
                # Check for date conversion, just in case
                if df[x_col].dtype != 'datetime64[ns]':
                    try:
                        temp_df = df.copy()
                        temp_df[x_col] = pd.to_datetime(temp_df[x_col], errors='raise')
                        df = temp_df # Use converted df for chart
                    except:
                        st.info("X-axis is not a recognized date format. Treating as a regular series.")

                try:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines+markers', name=y_col))
                    fig.update_layout(title=f"Trend Analysis", xaxis_title=x_col, yaxis_title="Value", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Unable to create chart: {str(e)}. Try selecting a different X-axis column (numeric or date).")
            else:
                st.warning("⚠️ No numeric columns found. Line charts require numeric data for the Y-axis.")
        
        elif viz_type == "Bar Chart":
            st.subheader("📊 Bar Chart")
            if len(numeric_cols) > 0:
                x_col = st.selectbox("X-axis:", all_cols, key="bar_x")
                y_col = st.selectbox("Y-axis:", numeric_cols, key="bar_y")
                color_col = st.selectbox("Color by (optional):", [None] + categorical_cols, key="bar_color")
                
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No numeric columns found. Bar charts require numeric data for the Y-axis.")
        
        elif viz_type == "Scatter Plot":
            st.subheader("🔵 Scatter Plot")
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
                color_col = st.selectbox("Color by (optional):", [None] + categorical_cols, key="scatter_color")
                size_col = st.selectbox("Size by (optional):", [None] + numeric_cols, key="scatter_size")
                
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, 
                                 title=f"{y_col} vs {x_col}", height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Need at least 2 numeric columns for scatter plot. Please upload data with multiple numeric columns.")
        
        elif viz_type == "Histogram":
            st.subheader("📊 Distribution - Histogram")
            if len(numeric_cols) > 0:
                col = st.selectbox("Select column:", numeric_cols, key="hist_col")
                bins = st.slider("Number of bins:", 10, 100, 30)
                
                fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No numeric columns found. Histograms require numeric data.")
        
        elif viz_type == "Box Plot":
            st.subheader("📦 Box Plot - Outlier Detection")
            if len(numeric_cols) > 0:
                y_col = st.selectbox("Value column:", numeric_cols, key="box_y")
                x_col = st.selectbox("Group by (optional):", [None] + categorical_cols, key="box_x")
                
                fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col}")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No numeric columns found. Box plots require numeric data.")
        
        elif viz_type == "Pie Chart":
            st.subheader("🥧 Pie Chart")
            if len(categorical_cols) > 0:
                names_col = st.selectbox("Category column:", categorical_cols, key="pie_names")
                values_col = st.selectbox("Values column (optional):", [None] + numeric_cols, key="pie_values")
                
                if values_col:
                    fig = px.pie(df, names=names_col, values=values_col, title=f"Distribution by {names_col}")
                else:
                    value_counts = df[names_col].value_counts()
                    fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution by {names_col}")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No categorical columns found. Pie charts require categorical data.")
        
        elif viz_type == "Heatmap (Correlation)":
            st.subheader("🔥 Correlation Heatmap")
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", 
                                 title="Correlation Matrix",
                                 color_continuous_scale='RdBu_r')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation heatmap.")
        
        elif viz_type == "Multi-Chart Dashboard":
            st.subheader("📊 Multi-Chart Dashboard")
            
            if len(numeric_cols) >= 1:
                try:
                    valid_numeric = df[numeric_cols[0]].dropna()
                    if len(valid_numeric) == 0:
                        st.warning(f"⚠️ Column '{numeric_cols[0]}' has no valid numeric data.")
                    else:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Distribution**")
                            fig1 = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            st.write("**Trend**")
                            fig2 = px.line(df, y=numeric_cols[0], title=f"Trend of {numeric_cols[0]}")
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        if len(numeric_cols) >= 2:
                            st.write("**Correlation Scatter**")
                            fig3 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                               title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
                            st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Unable to create dashboard: {str(e)}. Ensure numeric columns contain valid data.")
            else:
                st.warning("⚠️ No numeric columns found. Multi-chart dashboard requires numeric data.")

# --- Tab 6: Compare Periods (Part 3/4) ---
with tab6:
    st.header("Step 5: Compare Periods")
    
    st.info("💡 **Compare Periods**: Upload multiple datasets representing different time periods and compare metrics across them.")
    
    st.subheader("📁 Upload Period Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        period_name = st.text_input("Period Name (e.g., Q1 2024, January, Week 1)", key="period_name_input")
        period_file = st.file_uploader(
            "Upload period data file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            key="period_file_uploader"
        )
        
        if st.button("➕ Add Period", type="primary"):
            if period_name and period_file is not None:
                try:
                    if period_file.name.endswith('.csv'):
                        period_df = pd.read_csv(period_file)
                    elif period_file.name.endswith(('.xlsx', '.xls')):
                        period_df = pd.read_excel(period_file)
                    elif period_file.name.endswith('.json'):
                        period_df = pd.read_json(period_file)
                    elif period_file.name.endswith('.parquet'):
                        period_df = pd.read_parquet(period_file)
                    else:
                        st.error("Unsupported file format")
                        period_df = None
                    
                    if period_df is not None:
                        st.session_state.period_data[period_name] = period_df
                        st.success(f"✅ Added period '{period_name}' with {period_df.shape[0]} rows and {period_df.shape[1]} columns")
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
            else:
                st.warning("⚠️ Please provide both a period name and upload a file.")
    
    with col2:
        st.write("**Loaded Periods:**")
        if st.session_state.period_data:
            for period_name, period_df in st.session_state.period_data.items():
                st.write(f"✅ {period_name} ({period_df.shape[0]} rows)")
        else:
            st.info("No periods loaded yet")
        
        if st.session_state.period_data and st.button("🗑️ Clear All Periods"):
            st.session_state.period_data = {}
            st.experimental_rerun()
    
    if len(st.session_state.period_data) >= 2:
        st.divider()
        
        st.subheader("📊 Compare Metrics")
        
        period_names = list(st.session_state.period_data.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            period_1 = st.selectbox("Select first period:", period_names, key="compare_period_1")
        with col2:
            period_2 = st.selectbox("Select second period:", [p for p in period_names if p != period_1], key="compare_period_2")
        
        df1 = st.session_state.period_data[period_1]
        df2 = st.session_state.period_data[period_2]
        
        common_numeric_cols = list(set(df1.select_dtypes(include=[np.number]).columns) & 
                                    set(df2.select_dtypes(include=[np.number]).columns))
        
        if common_numeric_cols:
            selected_metrics = st.multiselect(
                "Select metrics to compare:",
                common_numeric_cols,
                default=common_numeric_cols[:min(3, len(common_numeric_cols))]
            )
            
            if selected_metrics:
                st.subheader("📈 Comparison Summary")
                
                comparison_data = []
                for metric in selected_metrics:
                    # Note: Using .sum() as the aggregation function for comparison
                    value_1 = df1[metric].sum()
                    value_2 = df2[metric].sum()
                    absolute_change = value_2 - value_1
                    percentage_change = ((value_2 - value_1) / value_1 * 100) if value_1 != 0 else 0
                    
                    comparison_data.append({
                        'Metric': metric,
                        f'{period_1}': value_1,
                        f'{period_2}': value_2,
                        'Absolute Change': absolute_change,
                        'Percentage Change (%)': percentage_change
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.style.format({
                    f'{period_1}': '{:.2f}',
                    f'{period_2}': '{:.2f}',
                    'Absolute Change': '{:.2f}',
                    'Percentage Change (%)': '{:.2f}'
                }), use_container_width=True)
                
                st.subheader("📊 Visual Comparison")
                
                
                fig = go.Figure()
                
                x_pos = list(range(len(selected_metrics)))
                
                fig.add_trace(go.Bar(
                    name=period_1,
                    x=selected_metrics,
                    y=[df1[metric].sum() for metric in selected_metrics],
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    name=period_2,
                    x=selected_metrics,
                    y=[df2[metric].sum() for metric in selected_metrics],
                    marker_color='lightcoral'
                ))
                
                fig.update_layout(
                    title=f"Comparison: {period_1} vs {period_2}",
                    xaxis_title="Metrics",
                    yaxis_title="Values",
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📥 Download Comparison")
                
                csv_data = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Comparison Table (CSV)",
                    data=csv_data,
                    file_name=f"comparison_{period_1}_vs_{period_2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                excel_buffer = BytesIO()
                comparison_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                st.download_button(
                    label="📊 Download Comparison Table (Excel)",
                    data=excel_buffer,
                    file_name=f"comparison_{period_1}_vs_{period_2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠️ No common numeric columns found between the two periods. Please ensure the datasets have matching numeric columns.")
        elif len(st.session_state.period_data) == 1:
            st.info("ℹ️ Please upload at least one more period to enable comparison.")
        else:
            st.info("ℹ️ Upload at least 2 periods to start comparing.")

# --- Tab 7: SQL Query Interface (Part 3/4 & 4/4) ---
with tab7:
    st.header("Step 5: SQL Query Interface")
    
    if st.session_state.cleaned_data is None and st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first.")
    else:
        df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.uploaded_data
        
        st.info("💡 **SQL Query Interface**: Query your data using SQL syntax. The data is available as a table named **'data'**.")
        
        # Setup in-memory SQLite database
        engine = create_engine('sqlite:///:memory:')
        df.to_sql('data', engine, index=False, if_exists='replace')
        
        st.subheader("Quick Queries")
        
        if 'sql_quick_result' not in st.session_state:
            st.session_state.sql_quick_result = None
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Show All Data"):
                try:
                    query = "SELECT * FROM data LIMIT 100"
                    st.code(query, language="sql")
                    result_df = pd.read_sql_query(query, engine)
                    st.success(f"✅ Returned {len(result_df)} rows")
                    st.dataframe(result_df, use_container_width=True)
                    st.session_state.sql_quick_result = result_df
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=result_df.to_csv(index=False),
                        file_name="all_data.csv",
                        mime="text/csv",
                        key="download_all"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        with col2:
            if st.button("Count Rows"):
                try:
                    query = "SELECT COUNT(*) as total_rows FROM data"
                    st.code(query, language="sql")
                    result_df = pd.read_sql_query(query, engine)
                    st.success(f"✅ Total rows: {result_df['total_rows'].iloc[0]}")
                    st.dataframe(result_df, use_container_width=True)
                    st.session_state.sql_quick_result = result_df
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=result_df.to_csv(index=False),
                        file_name="row_count.csv",
                        mime="text/csv",
                        key="download_count"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        with col3:
            if st.button("Show Columns"):
                try:
                    query = "SELECT name, type FROM pragma_table_info('data')"
                    st.code(query, language="sql")
                    result_df = pd.read_sql_query(query, engine)
                    # Add non-null count from original DataFrame for better info
                    result_df['non_null_count'] = [df[col].count() for col in df.columns] 
                    st.success(f"✅ Returned {len(result_df)} columns")
                    st.dataframe(result_df, use_container_width=True)
                    st.session_state.sql_quick_result = result_df
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=result_df.to_csv(index=False),
                        file_name="column_info.csv",
                        mime="text/csv",
                        key="download_cols"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        st.subheader("Custom SQL Query")
        
        sql_query = st.text_area(
            "Enter your SQL query:",
            value="SELECT * FROM data LIMIT 10",
            height=150,
            help="Example: SELECT column1, COUNT(*) FROM data GROUP BY column1"
        )
        
        if st.button("🚀 Execute Query", type="primary"):
            try:
                result_df = pd.read_sql_query(sql_query, engine)
                st.success(f"✅ Query executed successfully! Returned {len(result_df)} rows.")
                st.dataframe(result_df, use_container_width=True)
                
                st.download_button(
                    label="📥 Download Query Results (CSV)",
                    data=result_df.to_csv(index=False),
                    file_name="query_results.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ Query Error: {str(e)}")
        
        st.divider()
        
        st.subheader("Example Queries")
        st.code("""
-- Group by and aggregate
SELECT column_name, COUNT(*), AVG(numeric_column)
FROM data
GROUP BY column_name
ORDER BY COUNT(*) DESC;

-- Filter and sort
SELECT *
FROM data
WHERE numeric_column > 100
ORDER BY numeric_column DESC
LIMIT 20;

-- Join example (if multiple tables)
SELECT a.*, b.column
FROM data a
JOIN other_table b ON a.id = b.id;
        """, language="sql")

# --- Tab 8: Reports & Downloads (Part 4/4) ---
with tab8:
    st.header("Step 6: Reports & Downloads")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first.")
    else:
        df_original = st.session_state.uploaded_data
        df_cleaned = st.session_state.cleaned_data
        
        st.subheader("📋 Analysis Summary Report")
        
        st.write("**Task Instructions:**")
        if st.session_state.task_instructions:
            st.info(st.session_state.task_instructions)
        else:
            st.warning("No task instructions provided.")
        
        st.divider()
        
        st.write("**Data Processing Summary:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Rows", df_original.shape[0])
            st.metric("Original Columns", df_original.shape[1])
        with col2:
            if df_cleaned is not None:
                st.metric("Cleaned Rows", df_cleaned.shape[0])
                st.metric("Cleaned Columns", df_cleaned.shape[1])
            else:
                st.metric("Cleaned Rows", "Not cleaned yet")
                st.metric("Cleaned Columns", "Not cleaned yet")
        
        if st.session_state.cleaning_log:
            st.write("**Cleaning Operations Performed:**")
            for log in st.session_state.cleaning_log:
                st.write(log)
        
        st.divider()
        
        st.subheader("💡 Automated Insights")
        
        df_to_analyze = df_cleaned if df_cleaned is not None else df_original
        numeric_cols = df_to_analyze.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            insights = []
            
            st.write("**Key Trends & Patterns:**")
            
            top_values = []
            for col in numeric_cols:
                mean_val = df_to_analyze[col].mean()
                max_val = df_to_analyze[col].max()
                min_val = df_to_analyze[col].min()
                top_values.append({
                    'column': col,
                    'mean': mean_val,
                    'max': max_val,
                    'min': min_val,
                    'range': max_val - min_val
                })
            
            sorted_by_mean = sorted(top_values, key=lambda x: x['mean'], reverse=True)
            sorted_by_range = sorted(top_values, key=lambda x: x['range'], reverse=True)
            
            insights.append(f"• **Highest average value**: '{sorted_by_mean[0]['column']}' with a mean of {sorted_by_mean[0]['mean']:.2f}")
            if len(sorted_by_mean) > 1:
                insights.append(f"• **Lowest average value**: '{sorted_by_mean[-1]['column']}' with a mean of {sorted_by_mean[-1]['mean']:.2f}")
            insights.append(f"• **Widest range**: '{sorted_by_range[0]['column']}' ranges from {sorted_by_range[0]['min']:.2f} to {sorted_by_range[0]['max']:.2f} (range: {sorted_by_range[0]['range']:.2f})")
            
            for insight in insights:
                st.write(insight)
            
            if len(numeric_cols) > 1:
                st.write("\n**Correlations:**")
                
                corr_matrix = df_to_analyze[numeric_cols].corr()
                
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]
                            direction = "positive" if corr_value > 0 else "negative"
                            strong_correlations.append({
                                'col1': col1,
                                'col2': col2,
                                'value': corr_value,
                                'direction': direction
                            })
                
                if strong_correlations:
                    strong_correlations = sorted(strong_correlations, key=lambda x: abs(x['value']), reverse=True)
                    for corr in strong_correlations[:3]:
                        st.write(f"• **Strong {corr['direction']} correlation** ({corr['value']:.2f}) between **'{corr['col1']}'** and **'{corr['col2']}'**")
                else:
                    st.write("• No strong correlations detected (threshold: |r| > 0.7)")
            
            st.write("\n**Outliers Detected:**")
            
            outlier_info = []
            for col in numeric_cols:
                # Use IQR method for outlier detection
                Q1 = df_to_analyze[col].quantile(0.25)
                Q3 = df_to_analyze[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df_to_analyze[(df_to_analyze[col] < lower_bound) | (df_to_analyze[col] > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outlier_percentage = (outlier_count / len(df_to_analyze)) * 100
                    outlier_info.append({
                        'column': col,
                        'count': outlier_count,
                        'percentage': outlier_percentage
                    })
            
            if outlier_info:
                outlier_info = sorted(outlier_info, key=lambda x: x['count'], reverse=True)
                for info in outlier_info[:3]:
                    st.write(f"• **'{info['column']}'** has **{info['count']} outliers** ({info['percentage']:.1f}% of data)")
            else:
                st.write("• No significant outliers detected using IQR method")
            
            st.write("\n**Data Quality Insights:**")
            
            missing_data = df_to_analyze.isnull().sum()
            cols_with_missing = missing_data[missing_data > 0]
            
            if len(cols_with_missing) > 0:
                worst_missing = cols_with_missing.idxmax()
                worst_missing_pct = (cols_with_missing.max() / len(df_to_analyze)) * 100
                st.write(f"• **Missing data**: **'{worst_missing}'** has the most missing values ({cols_with_missing.max()} values, {worst_missing_pct:.1f}%)")
            else:
                st.write("• **Complete data**: No missing values detected across all columns")
            
            duplicate_count = df_to_analyze.duplicated().sum()
            if duplicate_count > 0:
                dup_percentage = (duplicate_count / len(df_to_analyze)) * 100
                st.write(f"• **Duplicates**: {duplicate_count} duplicate rows found ({dup_percentage:.1f}% of data)")
            else:
                st.write("• **No duplicates**: All rows are unique")
        else:
            st.info("ℹ️ No numeric columns available for automated insights. Upload data with numeric values to generate insights.")
        
        st.divider()
        
        st.subheader("📥 Download Deliverables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Download Cleaned Data:**")
            
            if df_cleaned is not None:
                csv_data = df_cleaned.to_csv(index=False)
                st.download_button(
                    label="📄 Download Cleaned Data (CSV)",
                    data=csv_data,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                excel_buffer = BytesIO()
                df_cleaned.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                st.download_button(
                    label="📊 Download Cleaned Data (Excel)",
                    data=excel_buffer,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("Clean data first to enable downloads.")
        
        with col2:
            st.write("**Download Analysis Report:**")
            
            df_to_analyze = df_cleaned if df_cleaned is not None else df_original
            
            summary_stats = df_to_analyze.describe(include='all').to_csv()
            st.download_button(
                label="📊 Download Statistical Summary (CSV)",
                data=summary_stats,
                file_name=f"statistical_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.divider()
        
        st.subheader("📄 Full Analysis Report")
        
        if st.button("📝 Generate Complete Report", type="primary"):
            report_text = f"""
DATA ANALYSIS & REPORTING
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
TASK INSTRUCTIONS
{'='*80}
{st.session_state.task_instructions if st.session_state.task_instructions else 'No task instructions provided'}

{'='*80}
DATA SUMMARY
{'='*80}
Original Dataset:
 - Rows: {df_original.shape[0]}
 - Columns: {df_original.shape[1]}

"""
            
            if df_cleaned is not None:
                report_text += f"""
Cleaned Dataset:
 - Rows: {df_cleaned.shape[0]}
 - Columns: {df_cleaned.shape[1]}
 - Rows Removed: {df_original.shape[0] - df_cleaned.shape[0]}

{'='*80}
CLEANING OPERATIONS
{'='*80}
"""
                for log in st.session_state.cleaning_log:
                    report_text += f"{log}\n"
            
            df_to_analyze = df_cleaned if df_cleaned is not None else df_original
            
            report_text += f"""

{'='*80}
STATISTICAL SUMMARY
{'='*80}

{df_to_analyze.describe(include='all').to_string()}

{'='*80}
KEY FINDINGS (Automated Insights)
{'='*80}
"""
            
            # Re-run automated insights calculation to embed in report
            numeric_cols = df_to_analyze.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                report_text += "\n--- Key Trends & Patterns ---\n"
                top_values = []
                for col in numeric_cols:
                    mean_val = df_to_analyze[col].mean()
                    max_val = df_to_analyze[col].max()
                    min_val = df_to_analyze[col].min()
                    top_values.append({
                        'column': col,
                        'mean': mean_val,
                        'max': max_val,
                        'min': min_val,
                        'range': max_val - min_val
                    })
                
                sorted_by_mean = sorted(top_values, key=lambda x: x['mean'], reverse=True)
                sorted_by_range = sorted(top_values, key=lambda x: x['range'], reverse=True)
                report_text += f"• Highest average value: '{sorted_by_mean[0]['column']}' with a mean of {sorted_by_mean[0]['mean']:.2f}\n"
                if len(sorted_by_mean) > 1:
                    report_text += f"• Lowest average value: '{sorted_by_mean[-1]['column']}' with a mean of {sorted_by_mean[-1]['mean']:.2f}\n"
                report_text += f"• Widest range: '{sorted_by_range[0]['column']}' ranges from {sorted_by_range[0]['min']:.2f} to {sorted_by_range[0]['max']:.2f} (range: {sorted_by_range[0]['range']:.2f})\n"

                report_text += "\n--- Data Quality Insights ---\n"
                missing_data = df_to_analyze.isnull().sum()
                cols_with_missing = missing_data[missing_data > 0]
                if len(cols_with_missing) > 0:
                    worst_missing = cols_with_missing.idxmax()
                    worst_missing_pct = (cols_with_missing.max() / len(df_to_analyze)) * 100
                    report_text += f"• Missing data: '{worst_missing}' has the most missing values ({cols_with_missing.max()} values, {worst_missing_pct:.1f}%)\n"
                else:
                    report_text += "• Complete data: No missing values detected across all columns\n"
                
                duplicate_count = df_to_analyze.duplicated().sum()
                if duplicate_count > 0:
                    dup_percentage = (duplicate_count / len(df_to_analyze)) * 100
                    report_text += f"• Duplicates: {duplicate_count} duplicate rows found ({dup_percentage:.1f}% of data)\n"
                else:
                    report_text += "• No duplicates: All rows are unique\n"
            else:
                report_text += "No numeric data available for automated insights.\n"
            
            report_text += f"""

{'='*80}
DELIVERABLES STATUS
{'='*80}
✅ Cleaned Dataset: Available for download
✅ Statistical Summary: Available for download
✅ Interactive Visualizations: Available in Visualizations tab
✅ SQL Query Interface: Available in SQL Query tab

{'='*80}
NEXT STEPS
{'='*80}
All data processing is complete and ready for client delivery.
Use the download buttons above to retrieve final deliverables.

"""
            
            st.text_area("Full Report:", report_text, height=400)
            
            st.download_button(
                label="📥 Download Full Report (TXT)",
                data=report_text,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            st.success("✅ Report generated successfully!")

# --- Sidebar Status and Quick Actions (Part 4/4) ---
st.sidebar.title("📊 Platform Status")
st.sidebar.divider()

if st.session_state.uploaded_data is not None:
    st.sidebar.success("✅ Data Uploaded")
else:
    st.sidebar.warning("⚠️ No Data Uploaded")

if st.session_state.cleaned_data is not None:
    st.sidebar.success("✅ Data Cleaned")
else:
    st.sidebar.info("ℹ️ Data Not Cleaned")

if st.session_state.task_instructions:
    st.sidebar.success("✅ Task Instructions Set")
else:
    st.sidebar.info("ℹ️ No Task Instructions")

st.sidebar.divider()
st.sidebar.subheader("Quick Actions")

if st.sidebar.button("🔄 Reset All Data"):
    # Reset all session state variables
    st.session_state.uploaded_data = None
    st.session_state.cleaned_data = None
    st.session_state.cleaning_log = []
    st.session_state.analysis_results = {}
    st.session_state.task_instructions = ""
    st.session_state.period_data = {} # Reset period comparison data
    st.experimental_rerun()

st.sidebar.divider()
st.sidebar.info("💡 **Workflow:** Upload → Clean → Analyze → Visualize → Report")
