# karachi-grocery-price-analytics
Interactive data science project that analyzes and compares grocery prices across Karachi stores using Python, EDA, and Linear Regression, with a Streamlit dashboard for visualizing price trends and store-wise comparisons.
🛒 Grocery Price Comparison & Trend Analysis for Karachi
📌 Project Overview

Grocery prices in Karachi vary significantly across different online stores, making it difficult for consumers to identify cost-effective options for essential items.
This project focuses on collecting, cleaning, analyzing, and visualizing grocery price data from multiple online retailers operating in Karachi and presenting insights through an interactive Streamlit dashboard.

The project is developed as part of the Introduction to Data Science Lab (CSL-495) at Bahria University, Karachi Campus.

🎯 Objectives

Collect grocery price data from multiple online stores

Clean and standardize price and unit information

Perform Exploratory Data Analysis (EDA) to compare prices

Visualize price trends across stores and categories

Apply a basic Linear Regression model to observe short-term price trends

Build an interactive Streamlit web app for data exploration

🧰 Technologies & Tools

Python

Pandas & NumPy – Data cleaning and processing

Matplotlib – Data visualization

Scikit-learn – Linear Regression model

Streamlit – Interactive dashboard

CSV Dataset – Price data storage

📊 Key Features
🔹 Data Collection

Manual price collection of essential grocery items

Limited web scraping from publicly available product pages

Unified dataset stored in CSV format

🔹 Data Cleaning & Processing

Removal of duplicate records

Conversion of units (kg, g, ml, pcs) into standard grams

Calculation of price per kg for fair comparison

🔹 Exploratory Data Analysis (EDA)

Store-wise price comparison

Category-wise price analysis

Visualizations including:

Bar charts

Line charts

Box plots

🔹 Price Trend Prediction

Linear Regression applied on selected grocery items

Visualization of historical prices vs predicted trends

Model evaluation using MSE and R² Score

🔹 Streamlit Dashboard

Interactive filters for:

Store

Category

Item

Real-time charts and statistics

Download filtered dataset as CSV

🖥️ Streamlit App Preview

The project includes a fully functional Streamlit web application that allows users to:

Compare grocery prices across stores

Analyze category-wise pricing

Visualize historical price trends

Predict short-term price movements

To run the app locally:

streamlit run app.py
