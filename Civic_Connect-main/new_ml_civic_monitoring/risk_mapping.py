import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

class RiskMapper:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.pca = PCA(n_components=2)
        self.is_trained = False
    
    def generate_sample_risk_data(self):
        """Generate sample risk assessment data"""
        locations = [f"Ward {i}" for i in range(1, 21)]
        
        data = []
        for location in locations:
            # Create risk patterns
            high_risk_areas = ['Ward 3', 'Ward 7', 'Ward 12', 'Ward 18']
            medium_risk_areas = ['Ward 5', 'Ward 9', 'Ward 11', 'Ward 15', 'Ward 19']
            
            if location in high_risk_areas:
                complaint_count = np.random.randint(50, 100)
                fault_frequency = np.random.uniform(0.3, 0.8)
                response_time = np.random.uniform(5.0, 10.0)
                user_rating = np.random.uniform(2.0, 3.5)
                risk_score = np.random.uniform(0.7, 1.0)
            elif location in medium_risk_areas:
                complaint_count = np.random.randint(20, 50)
                fault_frequency = np.random.uniform(0.1, 0.3)
                response_time = np.random.uniform(3.0, 5.0)
                user_rating = np.random.uniform(3.5, 4.0)
                risk_score = np.random.uniform(0.3, 0.7)
            else:
                complaint_count = np.random.randint(5, 20)
                fault_frequency = np.random.uniform(0.0, 0.1)
                response_time = np.random.uniform(1.0, 3.0)
                user_rating = np.random.uniform(4.0, 5.0)
                risk_score = np.random.uniform(0.0, 0.3)
            
            data.append({
                'location': location,
                'complaint_count': complaint_count,
                'fault_frequency': fault_frequency,
                'response_time': response_time,
                'user_rating': user_rating,
                'risk_score': risk_score,
                'latitude': 28.6 + np.random.uniform(-0.1, 0.1),
                'longitude': 77.2 + np.random.uniform(-0.1, 0.1),
                'population_density': np.random.randint(1000, 10000),
                'infrastructure_age': np.random.randint(1, 50)
            })
        
        return pd.DataFrame(data)
    
    def perform_risk_eda(self, df):
        """Exploratory Data Analysis for risk data"""
        st.subheader("📊 Risk Data Analysis")
        
        # Risk score distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(df, x='risk_score', 
                              title='Risk Score Distribution',
                              color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.box(df, y='risk_score', 
                         title='Risk Score Statistics')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig3 = px.imshow(corr_matrix, 
                        title='Risk Factor Correlations',
                        color_continuous_scale='RdBu_r',
                        aspect="auto")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Feature relationships
        fig4 = px.scatter(df, x='complaint_count', y='risk_score',
                         size='population_density', color='response_time',
                         hover_data=['location'],
                         title='Complaint Count vs Risk Score',
                         color_continuous_scale='Viridis')
        st.plotly_chart(fig4, use_container_width=True)
        
        return df
    
    def identify_hotspots(self, df):
        """Cluster locations based on risk factors"""
        st.subheader("🔍 Hotspot Detection using Clustering")
        
        # Prepare features for clustering
        feature_cols = ['complaint_count', 'fault_frequency', 'response_time', 
                       'user_rating', 'population_density', 'infrastructure_age']
        
        X = df[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply K-means clustering
        df['risk_cluster'] = self.kmeans.fit_predict(X_scaled)
        df['risk_score_ml'] = self.kmeans.transform(X_scaled).mean(axis=1)
        
        # Apply PCA for visualization
        X_pca = self.pca.fit_transform(X_scaled)
        df['pca1'] = X_pca[:, 0]
        df['pca2'] = X_pca[:, 1]
        
        # Visualize clusters
        fig1 = px.scatter(df, x='pca1', y='pca2', color='risk_cluster',
                         size='risk_score_ml', hover_data=['location'],
                         title='Risk Clusters Visualization (PCA)',
                         color_continuous_scale='Viridis')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Cluster analysis
        cluster_summary = df.groupby('risk_cluster').agg({
            'risk_score_ml': 'mean',
            'complaint_count': 'mean',
            'response_time': 'mean',
            'location': 'count'
        }).round(3)
        
        st.subheader("📈 Cluster Analysis")
        st.dataframe(cluster_summary, use_container_width=True)
        
        # Risk level assignment
        risk_thresholds = {
            'High Risk': df['risk_score_ml'].quantile(0.8),
            'Medium Risk': df['risk_score_ml'].quantile(0.5),
            'Low Risk': df['risk_score_ml'].min()
        }
        
        def assign_risk_level(score):
            if score >= risk_thresholds['High Risk']:
                return 'High Risk'
            elif score >= risk_thresholds['Medium Risk']:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        df['risk_level'] = df['risk_score_ml'].apply(assign_risk_level)
        
        self.is_trained = True
        return df
    
    def create_risk_map(self, df):
        """Create interactive risk heatmap"""
        st.subheader("🗺️ Interactive Risk Map")
        
        # Create base map
        center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add risk markers
        for _, row in df.iterrows():
            # Determine color based on risk level
            if row['risk_level'] == 'High Risk':
                color = 'red'
            elif row['risk_level'] == 'Medium Risk':
                color = 'orange'
            else:
                color = 'green'
            
            # Create popup content
            popup_text = f"""
            <b>{row['location']}</b><br>
            Risk Level: {row['risk_level']}<br>
            Complaints: {row['complaint_count']}<br>
            Response Time: {row['response_time']:.1f} days<br>
            User Rating: {row['user_rating']:.1f}/5.0
            """
            
            # Add marker to map
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=row['complaint_count'] / 5 + 5,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500)
        
        # Risk summary
        risk_summary = df['risk_level'].value_counts()
        fig = px.pie(values=risk_summary.values, names=risk_summary.index,
                    title='Risk Level Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_risk_insights(self, df):
        """Generate actionable insights from risk analysis"""
        st.subheader("💡 Risk Analysis Insights")
        
        high_risk_areas = df[df['risk_level'] == 'High Risk']
        medium_risk_areas = df[df['risk_level'] == 'Medium Risk']
        
        insights = []
        
        if not high_risk_areas.empty:
            worst_area = high_risk_areas.loc[high_risk_areas['risk_score_ml'].idxmax()]
            insights.append(f"🚨 **Priority Action Needed**: {worst_area['location']} has the highest risk score ({worst_area['risk_score_ml']:.2f})")
            insights.append(f"📞 **Key Issues**: {int(worst_area['complaint_count'])} complaints with {worst_area['response_time']:.1f} days average response time")
        
        if not medium_risk_areas.empty:
            insights.append(f"⚠️ **Watchlist**: {len(medium_risk_areas)} areas require monitoring")
        
        # Correlation insights
        complaint_corr = df[['risk_score_ml', 'complaint_count']].corr().iloc[0,1]
        response_corr = df[['risk_score_ml', 'response_time']].corr().iloc[0,1]
        
        insights.append(f"📊 **Risk Drivers**: Complaint count (r={complaint_corr:.2f}) and response time (r={response_corr:.2f}) strongly correlate with risk")
        
        for insight in insights:
            st.info(insight)

def run_risk_mapping():
    """Streamlit interface for risk mapping"""
    st.header("🗺️ Civic Risk Hotspot Detection")
    
    mapper = RiskMapper()
    
    # Load sample data
    st.subheader("📁 Risk Data Overview")
    df = mapper.generate_sample_risk_data()
    st.success("✅ Sample risk assessment data loaded successfully!")
    
    st.dataframe(df.head(), use_container_width=True)
    
    # EDA
    df = mapper.perform_risk_eda(df)
    
    # Hotspot detection
    if st.button("🔍 Identify Risk Hotspots"):
        with st.spinner("Analyzing risk patterns..."):
            df = mapper.identify_hotspots(df)
            st.success("✅ Risk hotspots identified!")
    
    if mapper.is_trained:
        # Display results
        st.subheader("📋 Risk Assessment Results")
        results_df = df[['location', 'risk_level', 'risk_score_ml', 'complaint_count', 'response_time']]
        st.dataframe(results_df.sort_values('risk_score_ml', ascending=False), use_container_width=True)
        
        # Create risk map
        mapper.create_risk_map(df)
        
        # Generate insights
        mapper.generate_risk_insights(df)