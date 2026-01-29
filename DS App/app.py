import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Grocery Price Analysis", layout="wide")

# Title
st.title("🛒 Grocery Price Analysis Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("grocery.csv")
    
    # Data cleaning
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df['Price_PKR'] = pd.to_numeric(df['Price_PKR'])
    
    # Unit conversion
    def convert_to_grams(unit):
        unit = str(unit).lower().strip()
        
        # Remove extra spaces
        unit = ''.join(unit.split())
        
        try:
            if 'kg' in unit:
                return float(unit.replace('kg','')) * 1000
            elif 'g' in unit and 'kg' not in unit:
                return float(unit.replace('g',''))
            elif 'l' in unit or 'ml' in unit:
                if 'ml' in unit:
                    return float(unit.replace('ml',''))
                else:
                    return float(unit.replace('l','')) * 1000
            elif 'pcs' in unit or 'sheets' in unit or 'rolls' in unit:
                # For items sold by pieces, use 1000g as standard
                return 1000.0
            else:
                # Try to extract just the number
                import re
                numbers = re.findall(r'\d+\.?\d*', unit)
                if numbers:
                    return float(numbers[0]) * 1000  # Assume kg if no unit
                return np.nan
        except:
            return np.nan
    
    df['quantity_grams'] = df['Unit'].apply(convert_to_grams)
    df['price_per_kg'] = (df['Price_PKR'] / df['quantity_grams']) * 1000
    
    # Clean data
    df = df.dropna(subset=['price_per_kg'])
    df = df.drop_duplicates()
    
    return df

try:
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    categories = st.sidebar.multiselect(
        "Select Categories",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )
    
    stores = st.sidebar.multiselect(
        "Select Stores",
        options=df['Store'].unique(),
        default=df['Store'].unique()
    )
    
    # Filter data
    filtered_df = df[
        (df['Category'].isin(categories)) & 
        (df['Store'].isin(stores))
    ]
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Items", len(filtered_df['Item'].unique()))
    with col2:
        st.metric("Avg Price/KG", f"PKR {filtered_df['price_per_kg'].mean():.2f}")
    with col3:
        st.metric("Date Range", f"{filtered_df['Date'].min().date()} to {filtered_df['Date'].max().date()}")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Store Comparison", "📈 Category Analysis", "🔮 Price Prediction", "📋 Data View"])
    
    with tab1:
        st.subheader("Average Price per KG by Store")
        
        store_avg = filtered_df.groupby('Store')['price_per_kg'].mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        store_avg.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_ylabel("Price per KG (PKR)")
        ax.set_xlabel("Store")
        ax.set_title("Store-wise Price Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Store statistics
        st.subheader("Store Statistics")
        store_stats = filtered_df.groupby('Store')['price_per_kg'].describe()
        st.dataframe(store_stats)
    
    with tab2:
        st.subheader("Average Price per KG by Category")
        
        category_avg = filtered_df.groupby('Category')['price_per_kg'].mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        category_avg.plot(kind='bar', ax=ax, color='coral')
        ax.set_ylabel("Price per KG (PKR)")
        ax.set_xlabel("Category")
        ax.set_title("Category-wise Price Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Category statistics
        st.subheader("Category Statistics")
        category_stats = filtered_df.groupby('Category')['price_per_kg'].describe()
        st.dataframe(category_stats)
    
    with tab3:
        st.subheader("Price Trend Prediction")
        
        # Item selection
        items = filtered_df['Item'].unique()
        selected_item = st.selectbox("Select Item for Price Prediction", items)
        
        item_df = filtered_df[filtered_df['Item'] == selected_item].copy()
        
        if len(item_df) > 1:
            # Price trend visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for store in item_df['Store'].unique():
                s_data = item_df[item_df['Store'] == store]
                ax.plot(s_data['Date'], s_data['price_per_kg'], marker='o', label=store)
            
            ax.set_title(f"{selected_item} Price Trend Across Stores")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price per KG (PKR)")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Linear regression
            st.subheader("Price Prediction Model")
            
            item_df['date_ordinal'] = item_df['Date'].map(pd.Timestamp.toordinal)
            X = item_df[['date_ordinal']]
            y = item_df['price_per_kg']
            
            model = LinearRegression()
            model.fit(X, y)
            
            item_df['predicted_price'] = model.predict(X)
            
            # Model metrics
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.2f}")
            with col2:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Regression visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(item_df['Date'], item_df['price_per_kg'], label='Actual', alpha=0.6)
            ax.plot(item_df['Date'], item_df['predicted_price'], color='red', label='Regression Line', linewidth=2)
            ax.set_title(f"Linear Regression: {selected_item} Price Trend")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price per KG (PKR)")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Not enough data points for prediction. Please select an item with multiple records.")
    
    with tab4:
        st.subheader("Filtered Data View")
        
        # Search functionality
        search_term = st.text_input("Search in data", "")
        
        if search_term:
            display_df = filtered_df[
                filtered_df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)
            ]
        else:
            display_df = filtered_df
        
        # Sort options
        sort_by = st.selectbox("Sort by", ['Date', 'Item', 'Store', 'price_per_kg'])
        ascending = st.checkbox("Ascending", value=True)
        
        display_df = display_df.sort_values(by=sort_by, ascending=ascending)
        
        st.dataframe(display_df[['Date', 'Store', 'Category', 'Item', 'Brand', 'Unit', 'Price_PKR', 'price_per_kg']], use_container_width=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_grocery_data.csv",
            mime="text/csv"
        )

except FileNotFoundError:
    st.error("❌ Error: 'grocery.csv' file not found. Please make sure the file is in the same directory as this script.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("📊 **Grocery Price Analysis Dashboard** | Built with Streamlit")