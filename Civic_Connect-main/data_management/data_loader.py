import pandas as pd
import numpy as np
import streamlit as st
import requests
from io import StringIO

class RealDataLoader:
    def __init__(self):
        self.dataset_info = {
            'nyc_311': {
                'name': 'NYC 311 Service Requests',
                'url': 'https://data.cityofnewyork.us/resource/erm2-nwe9.json',
                'description': 'Real citizen complaints from New York City'
            }
        }
    
    def load_nyc_311_data(self, limit=1000):
        """Load real NYC 311 complaint data - FIXED DUPLICATE COLUMNS"""
        try:
            st.info("📡 Downloading real NYC 311 service request data...")
            
            # API query for recent complaints
            query = f"?$limit={limit}&$where=created_date >= '2023-01-01'"
            url = self.dataset_info['nyc_311']['url'] + query
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                
                if not df.empty:
                    # Select and rename relevant columns - FIXED: Remove duplicates
                    column_mapping = {
                        'complaint_type': 'category',
                        'descriptor': 'complaint_text', 
                        'incident_zip': 'location',
                        'created_date': 'timestamp',
                        'status': 'status'
                    }
                    
                    # Use only available columns that exist in the DataFrame
                    available_columns = {}
                    for old_col, new_col in column_mapping.items():
                        if old_col in df.columns:
                            available_columns[old_col] = new_col
                    
                    # Rename columns
                    df = df.rename(columns=available_columns)
                    
                    # Keep only renamed columns and remove duplicates
                    keep_columns = list(available_columns.values())
                    
                    # Remove duplicate columns by keeping only unique ones
                    df = df.loc[:, ~df.columns.duplicated()]
                    
                    # Select only the columns we want
                    if keep_columns:
                        # Ensure we only take columns that actually exist after deduplication
                        existing_columns = [col for col in keep_columns if col in df.columns]
                        df = df[existing_columns]
                    
                    st.success(f"✅ Loaded {len(df)} real complaints from NYC 311")
                    return df
                else:
                    st.warning("No data returned from API. Using realistic sample data instead.")
                    return self._create_fallback_data()
            else:
                st.warning(f"API request failed. Status: {response.status_code}. Using sample data.")
                return self._create_fallback_data()
                
        except Exception as e:
            st.warning(f"Could not fetch real data: {e}. Using realistic sample data.")
            return self._create_fallback_data()
    
    def _create_fallback_data(self):
        """Create realistic fallback data if API fails"""
        # More realistic sample data based on actual civic issues
        sample_complaints = [
            "Water leak from hydrant at street corner flooding the area",
            "Street light out on main road causing safety concerns at night",
            "Garbage accumulation in park not collected for two weeks",
            "Large pothole on highway causing vehicle damage and accidents",
            "Sewer backup in basement creating health hazard and bad odor",
            "No water pressure in apartment building affecting all residents",
            "Broken water main flooding residential street and sidewalks",
            "Illegal dumping in vacant lot attracting pests and rodents",
            "Traffic signal malfunction at busy intersection causing traffic jams",
            "Noise complaint from construction site operating after permitted hours"
        ]
        
        categories = [
            'Water Supply', 'Streetlight', 'Sanitation', 'Road Damage', 'Sewer',
            'Water Supply', 'Water Supply', 'Illegal Dumping', 'Traffic Signal', 'Noise'
        ]
        
        df = pd.DataFrame({
            'complaint_text': sample_complaints,
            'category': categories,
            'location': [f"Ward {i%5 + 1}" for i in range(10)],
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'status': ['Open'] * 4 + ['In Progress'] * 3 + ['Closed'] * 3,
            'user_rating': np.random.randint(3, 6, 10)
        })
        
        return df

def run_complaint_analysis_real():
    """Enhanced complaint analysis with real dataset option - FIXED VERSION"""
    st.header("📝 Smart Complaint Analysis")
    
    # Import here to avoid circular imports
    from new_ml_civic_monitoring.complaint_analysis import ComplaintAnalyzer
    analyzer = ComplaintAnalyzer()
    
    # Data source selection
    st.subheader("📁 Data Source Selection")
    
    data_source = st.radio(
        "Choose data source:",
        ["Use Sample Data for Demo", "Try Real NYC 311 Data (Internet Required)"],
        index=0
    )
    
    data_loader = RealDataLoader()
    
    if data_source == "Try Real NYC 311 Data (Internet Required)":
        if st.button("🔄 Download Real Complaint Data"):
            with st.spinner("Downloading real civic complaint data from NYC Open Data..."):
                df = data_loader.load_nyc_311_data(limit=500)  # Reduced limit for stability
        else:
            st.info("Click the button above to download real NYC 311 complaint data")
            return
    else:
        # Use sample data
        df = analyzer.generate_sample_data()
        st.success("✅ Sample complaint data loaded successfully!")
    
    if df is not None:
        # Show data overview - FIXED: Handle potential display issues
        st.subheader("📊 Data Overview")
        
        # Clean the DataFrame for display
        display_df = df.copy()
        
        # Ensure all columns are string type for display
        for col in display_df.columns:
            display_df[col] = display_df[col].astype(str)
        
        # Remove any duplicate columns that might still exist
        display_df = display_df.loc[:, ~display_df.columns.duplicated()]
        
        st.dataframe(display_df.head(), use_container_width=True)
        
        # Show column information
        st.write(f"**Columns in dataset:** {list(display_df.columns)}")
        st.write(f"**Shape:** {display_df.shape}")
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Complaints", len(display_df))
        with col2:
            category_count = display_df['category'].nunique() if 'category' in display_df.columns else 0
            st.metric("Categories", category_count)
        with col3:
            if 'timestamp' in display_df.columns:
                try:
                    timestamp_min = display_df['timestamp'].min()
                    timestamp_max = display_df['timestamp'].max()
                    min_str = str(timestamp_min)[:10]
                    max_str = str(timestamp_max)[:10]
                    st.metric("Date Range", f"{min_str} to {max_str}")
                except:
                    st.metric("Date Range", "N/A")
            else:
                st.metric("Date Range", "N/A")
        
        # Continue with EDA and model training
        if st.button("📊 Perform Data Analysis"):
            with st.spinner("Analyzing data..."):
                df_analyzed = analyzer.perform_eda(df)
        
        if st.button("🚀 Train Classification Model"):
            with st.spinner("Training machine learning model..."):
                accuracy = analyzer.train_model(df)
                if accuracy > 0:
                    st.success(f"Model trained successfully!")
                else:
                    st.error("Model training failed. Please try again.")