import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page for optimal senior user experience
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛒",
    layout="wide",  # Wide layout reduces scrolling
    initial_sidebar_state="collapsed"  # Simpler single-page view
)

# Custom CSS for Senior-Friendly Design - Larger fonts, high contrast, clear spacing
st.markdown("""
    <style>
    /* Increase base font size for better readability */
    .stApp {
        font-size: 16px;
    }
    
    /* Large, prominent headers */
    h1 {
        font-size: 42px !important;
        color: #1f4788;
        font-weight: bold;
    }
    
    h2 {
        font-size: 32px !important;
        color: #2c5aa0;
        margin-top: 30px;
    }
    
    h3 {
        font-size: 24px !important;
        color: #3d6bb3;
    }
    
    /* Large metric displays */
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 18px !important;
    }
    
    /* High contrast buttons */
    .stButton button {
        font-size: 18px;
        padding: 15px 30px;
        background-color: #1f4788;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    
    .stButton button:hover {
        background-color: #2c5aa0;
    }
    
    /* Clear spacing between sections */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Accessible select boxes */
    .stSelectbox label {
        font-size: 18px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load datasets
transaction_data =  pd.read_csv('datasets/transaction_data.csv')
product = pd.read_csv('datasets/product.csv')
hh_demographic = pd.read_csv('datasets/hh_demographic.csv')
campaign_table = pd.read_csv('datasets/campaign_table.csv')
campaign_desc = pd.read_csv('datasets/campaign_desc.csv')

# Define the start date (Day 1)
start_date = pd.to_datetime('2023-01-01')

# Calculate the new 'DATE' column:
# 1. Access the 'DAY' column in your DataFrame.
# 2. Subtract 1 (since Day 1 has a zero-day offset).
# 3. Use pd.to_timedelta to convert the day difference into a time difference.
# 4. Add the timedelta to the start_date.

transaction_data['DATE'] = start_date + pd.to_timedelta(
    transaction_data['DAY'] - 1, 
    unit='D'
)

# Convert 'START_DAY' to 'START_DATE' in campaign table
campaign_desc['START_DATE'] = start_date + pd.to_timedelta(
    campaign_desc['START_DAY'] - 1, 
    unit='D'
)

# Convert 'END_DAY' to 'END_DATE' in campaign table
campaign_desc['END_DATE'] = start_date + pd.to_timedelta(
    campaign_desc['END_DAY'] - 1, 
    unit='D'
)

# Add product information to transactions
df = transaction_data.merge(
    product[['PRODUCT_ID', 'DEPARTMENT', 'BRAND', 'COMMODITY_DESC']], 
    left_on='PRODUCT_ID',
    right_on='PRODUCT_ID',
    how='left'
)

# Add demographic information (select key classification variables)
df = df.merge(
    hh_demographic[['household_key', 'classification_1', 'classification_2', 
                    'classification_3', 'classification_5']], 
    on='household_key', 
    how='left'
)

# Rename for clarity
df = df.rename(columns={
    'classification_1': 'DEMOGRAPHIC_GROUP',      # Group1 through Group6
    'classification_2': 'DEMOGRAPHIC_TYPE',        # X, Y, Z
    'classification_3': 'DEMOGRAPHIC_LEVEL',       # Level1 through Level12
    'classification_5': 'SHOPPING_SEGMENT'         # Group1 through Group6
})

# Add campaign participation
campaign_participation = campaign_table[['household_key', 'CAMPAIGN', 'DESCRIPTION']].copy()
campaign_participation['IN_CAMPAIGN'] = 1

# Get first campaign per household (if multiple campaigns)
campaign_participation = campaign_participation.groupby('household_key').first().reset_index()

df = df.merge(
    campaign_participation, 
    on='household_key', 
    how='left'
)

df['IN_CAMPAIGN'] = df['IN_CAMPAIGN'].fillna(0).astype(int)
df['CAMPAIGN_TYPE'] = df['DESCRIPTION'].fillna('No Campaign')

# Temporal features (critical for time-series forecasting)
df['MONTH'] = df['DATE'].dt.month
df['MONTH_NAME'] = df['DATE'].dt.strftime('%B')
df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
df['DAY_NAME'] = df['DATE'].dt.strftime('%A')
df['QUARTER'] = df['DATE'].dt.quarter
df['YEAR'] = df['DATE'].dt.year
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)

# Discount features (for price optimization ML)
df['TOTAL_DISCOUNT'] = (
    df['COUPON_MATCH_DISC'] + 
    df['COUPON_DISC'] + 
    df['RETAIL_DISC']
)
df['DISCOUNT_RATE'] = (
    df['TOTAL_DISCOUNT'] / 
    (df['SALES_VALUE'] + df['TOTAL_DISCOUNT'])
).fillna(0)

# Revenue features
df['NET_REVENUE'] = df['SALES_VALUE']  # Already net of discounts
df['UNIT_PRICE'] = df['SALES_VALUE'] / df['QUANTITY']
df['HAS_DISCOUNT'] = (df['TOTAL_DISCOUNT'] > 0).astype(int)

# Customer-Level Aggregations
# Customer lifetime value and segmentation features for ML
customer_metrics = df.groupby('household_key').agg({
    'BASKET_ID': 'nunique',          # Number of shopping trips
    'SALES_VALUE': 'sum',             # Total spent
    'QUANTITY': 'sum',                # Total items bought
    'DATE': ['min', 'max'],            # First and last purchase
    'TOTAL_DISCOUNT': 'sum',          # Total discounts received
    'STORE_ID': 'nunique'             # Number of different stores visited
}).reset_index()

customer_metrics.columns = ['household_key', 'NUM_TRIPS', 'TOTAL_SPENT', 
                            'TOTAL_ITEMS', 'FIRST_PURCHASE', 'LAST_PURCHASE',
                            'TOTAL_DISCOUNTS', 'NUM_STORES']

customer_metrics['DAYS_ACTIVE'] = (
    customer_metrics['LAST_PURCHASE'] - customer_metrics['FIRST_PURCHASE']
).dt.days + 1

customer_metrics['AVG_BASKET_VALUE'] = (
    customer_metrics['TOTAL_SPENT'] / customer_metrics['NUM_TRIPS']
)

customer_metrics['ITEMS_PER_TRIP'] = (
    customer_metrics['TOTAL_ITEMS'] / customer_metrics['NUM_TRIPS']
)

customer_metrics['DISCOUNT_RATE'] = (
    customer_metrics['TOTAL_DISCOUNTS'] / 
    (customer_metrics['TOTAL_SPENT'] + customer_metrics['TOTAL_DISCOUNTS'])
)

product_performance = df.groupby('PRODUCT_ID').agg({
    'QUANTITY': 'sum',
    'SALES_VALUE': 'sum',
    'BASKET_ID': 'nunique',
    'household_key': 'nunique',
    'TOTAL_DISCOUNT': 'sum'
}).reset_index()

product_performance.columns = ['PRODUCT_ID', 'TOTAL_QUANTITY', 'TOTAL_SALES', 
                               'NUM_BASKETS', 'NUM_CUSTOMERS', 'TOTAL_DISCOUNTS']

# Merge back product details
product_performance = product_performance.merge(
    product[['PRODUCT_ID', 'DEPARTMENT', 'BRAND', 'COMMODITY_DESC']], 
    left_on='PRODUCT_ID',
    right_on='PRODUCT_ID',
    how='left'
)

product_performance['AVG_PRICE'] = (
    product_performance['TOTAL_SALES'] / product_performance['TOTAL_QUANTITY']
)

dept_performance = df.groupby('DEPARTMENT').agg({
    'SALES_VALUE': 'sum',
    'QUANTITY': 'sum',
    'BASKET_ID': 'nunique',
    'household_key': 'nunique'
}).reset_index()

dept_performance.columns = ['DEPARTMENT', 'TOTAL_REVENUE', 'TOTAL_QUANTITY',
                            'NUM_BASKETS', 'NUM_CUSTOMERS']

campaign_metrics = df.groupby(['household_key', 'IN_CAMPAIGN']).agg({
    'SALES_VALUE': 'sum',
    'BASKET_ID': 'nunique',
    'QUANTITY': 'sum'
}).reset_index()

# HEADER SECTION
st.title("🛒 Retail Business Intelligence Dashboard")
st.markdown("""
<div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
    <h3 style='color: #1f4788; margin-top: 0;'>
        2-Year Analysis: 2,500 Frequent Shopper Households
    </h3>
    <p style='font-size: 18px; margin-bottom: 0;'>
        This dashboard analyzes purchasing patterns from our most valuable customers over 
        a 2-year period. All charts are interactive - simply hover over them to see detailed information.
    </p>
</div>
""", unsafe_allow_html=True)

# KEY METRICS SECTION
# Large numbers at top provide immediate context
st.header("📈 Business Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = df['SALES_VALUE'].sum()
    st.metric(
        label="Total Revenue",
        value=f"${total_revenue:,.0f}",
        delta="2 year period"
    )

with col2:
    avg_basket = df.groupby('BASKET_ID')['SALES_VALUE'].sum().mean()
    st.metric(
        label="Average Basket",
        value=f"${avg_basket:.2f}",
        delta="per shopping trip"
    )

with col3:
    unique_customers = df['household_key'].nunique()
    st.metric(
        label="Active Households",
        value=f"{unique_customers:,}",
        delta="frequent shoppers"
    )

with col4:
    total_items = df['QUANTITY'].sum()
    st.metric(
        label="Items Sold",
        value=f"{total_items:,.0f}",
        delta="units"
    )

# Additional context
col1, col2, col3 = st.columns(3)
with col1:
    num_transactions = df['BASKET_ID'].nunique()
    st.metric("Shopping Trips", f"{num_transactions:,}")
with col2:
    num_products = df['PRODUCT_ID'].nunique()
    st.metric("Unique Products", f"{num_products:,}")
with col3:
    avg_items_per_basket = df.groupby('BASKET_ID')['QUANTITY'].sum().mean()
    st.metric("Items per Trip", f"{avg_items_per_basket:.1f}")

# SECTION 1: SALES TRENDS (Time-series for forecasting ML)
st.header("📅 Sales Trends Over Time")
st.markdown("""
<p style='font-size: 18px; color: #555;'>
    <strong>Why this matters for predictions:</strong> Two years of historical data captures 
    seasonal patterns, trends, and shopping behavior changes over time.
</p>
""", unsafe_allow_html=True)

# Monthly sales aggregation
monthly_sales = df.groupby(df['DATE'].dt.to_period('M')).agg({
    'SALES_VALUE': 'sum',
    'BASKET_ID': 'nunique',
    'QUANTITY': 'sum'
}).reset_index()
monthly_sales['DATE'] = monthly_sales['DATE'].dt.to_timestamp()

# Create line chart
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=monthly_sales['DATE'],
    y=monthly_sales['SALES_VALUE'],
    mode='lines+markers',
    name='Monthly Revenue',
    line=dict(color='#1f4788', width=4),
    marker=dict(size=12),
    hovertemplate='<b>%{x|%B %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
))

fig_trend.update_layout(
    height=500,
    font=dict(size=16),
    title=dict(text='Monthly Revenue Trend (2 Years)', font=dict(size=24)),
    xaxis=dict(
        title=dict(
            text='Month',
        font=dict(size=18)
        ),
        tickfont=dict(size=14),
        gridcolor='#e0e0e0'
    ),
    yaxis=dict(
        title=dict(
            text='Revenue ($)',
            font=dict(size=18) 
        ),
        tickfont=dict(size=14),
        gridcolor='#e0e0e0'
    ),
    hovermode='x unified',
    plot_bgcolor='white'
)

st.plotly_chart(fig_trend, use_container_width=True)

# Year-over-year comparison
yearly_sales = df.groupby('YEAR')['SALES_VALUE'].sum().reset_index()
if len(yearly_sales) > 1:
    col1, col2 = st.columns(2)
    for idx, row in yearly_sales.iterrows():
        with col1 if idx == 0 else col2:
            st.metric(
                f"Year {int(row['YEAR'])} Revenue",
                f"${row['SALES_VALUE']:,.0f}"
            )

with st.expander("How this data helps Machine Learning", expanded=False):
    st.markdown("""
    <div style='font-size: 16px; line-height: 1.8;'>
        <p><strong>Time Series Forecasting:</strong></p>
        <ul>
            <li><strong>Pattern Recognition:</strong> 24 months of data reveals seasonal cycles and trends</li>
            <li><strong>Demand Prediction:</strong> ML models can forecast sales 3-6 months ahead with 85-90% accuracy</li>
            <li><strong>Inventory Planning:</strong> Predict which products will be in high demand when</li>
            <li><strong>Anomaly Detection:</strong> Identify unusual sales patterns that may indicate problems</li>
        </ul>
        <p><strong>Models to use:</strong> ARIMA, Prophet, LSTM Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

