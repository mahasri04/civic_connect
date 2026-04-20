import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from textblob import TextBlob

class BusinessInsights:
    def __init__(self):
        self.metrics = {}
    
    def generate_comprehensive_insights(self, complaints_df, utility_df, risk_df):
        """Generate comprehensive business insights with proper error handling"""
        st.header("📈 Business Intelligence Dashboard")
        
        # Key Performance Indicators
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_complaints = len(complaints_df)
            st.metric("Total Complaints", total_complaints)
        
        with col2:
            # Handle missing response_time column
            if 'response_time' in risk_df.columns:
                avg_response_time = risk_df['response_time'].mean()
                st.metric("Avg Response Time", f"{avg_response_time:.1f} days")
            else:
                st.metric("Avg Response Time", "N/A")
        
        with col3:
            # Handle missing user_rating column
            if 'user_rating' in complaints_df.columns:
                satisfaction_rate = (complaints_df['user_rating'] >= 4).mean()
                st.metric("Satisfaction Rate", f"{satisfaction_rate:.1%}")
            else:
                st.metric("Satisfaction Rate", "N/A")
        
        with col4:
            # Handle missing risk_level column
            if 'risk_level' in risk_df.columns:
                high_risk_areas = len(risk_df[risk_df['risk_level'] == 'High Risk'])
                st.metric("High Risk Areas", high_risk_areas)
            else:
                # Alternative: calculate risk based on available columns
                if 'risk_score' in risk_df.columns:
                    high_risk_threshold = risk_df['risk_score'].quantile(0.8)
                    high_risk_areas = len(risk_df[risk_df['risk_score'] >= high_risk_threshold])
                    st.metric("High Risk Areas", high_risk_areas)
                else:
                    st.metric("High Risk Areas", "N/A")
        
        # Complaint Analysis Insights
        st.subheader("📝 Complaint Analytics")
        self._analyze_complaints(complaints_df)
        
        # Operational Efficiency
        st.subheader("⚙️ Operational Efficiency")
        self._analyze_operations(utility_df, risk_df)
        
        # Predictive Insights
        st.subheader("🔮 Predictive Insights & Recommendations")
        self._generate_recommendations(complaints_df, utility_df, risk_df)

    def _analyze_complaints(self, df):
        """Analyze complaint patterns and trends"""
        # Add your complaint analysis code here
        st.info("Complaint analysis would be displayed here")
        # You can add your actual complaint analysis code from previous implementations

    def _analyze_operations(self, utility_df, risk_df):
        """Analyze operational efficiency and resource utilization"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Utility usage patterns
            utility_df['timestamp'] = pd.to_datetime(utility_df['timestamp'])
            utility_df['day_type'] = utility_df['timestamp'].dt.dayofweek.apply(
                lambda x: 'Weekend' if x >= 5 else 'Weekday'
            )
            
            usage_by_day_type = utility_df.groupby('day_type')['water_usage'].mean()
            
            fig1 = px.pie(values=usage_by_day_type.values, names=usage_by_day_type.index,
                         title='Average Water Usage by Day Type')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Risk vs Response time correlation
            fig2 = px.scatter(risk_df, x='response_time', y='risk_score',
                            size='complaint_count', color='risk_level',
                            title='Response Time vs Risk Score',
                            hover_data=['location'])
            st.plotly_chart(fig2, use_container_width=True)
        
        # Efficiency metrics
        st.subheader("📊 Efficiency Metrics")
        
        eff_col1, eff_col2, eff_col3 = st.columns(3)
        
        with eff_col1:
            avg_water_usage = utility_df['water_usage'].mean()
            st.metric("Avg Water Usage", f"{avg_water_usage:.0f} units")
        
        with eff_col2:
            anomaly_rate = (utility_df['water_usage'] > utility_df['water_usage'].quantile(0.9)).mean()
            st.metric("Anomaly Rate", f"{anomaly_rate:.1%}")
        
        with eff_col3:
            resource_efficiency = (risk_df['user_rating'] / risk_df['complaint_count']).mean()
            st.metric("Resource Efficiency", f"{resource_efficiency:.2f}")

    def _generate_recommendations(self, complaints_df, utility_df, risk_df):
        """Generate actionable business recommendations with proper error handling"""
        
        st.subheader("🎯 Strategic Recommendations")
        
        recommendations = []
        
        # Priority areas analysis with error handling
        if 'risk_level' in risk_df.columns:
            high_risk_locations = risk_df[risk_df['risk_level'] == 'High Risk']
        elif 'risk_score' in risk_df.columns:
            high_risk_threshold = risk_df['risk_score'].quantile(0.8)
            high_risk_locations = risk_df[risk_df['risk_score'] >= high_risk_threshold]
        else:
            high_risk_locations = pd.DataFrame()  # Empty DataFrame
        
        # Slow response areas with error handling
        if 'response_time' in risk_df.columns:
            slow_response_areas = risk_df[risk_df['response_time'] > risk_df['response_time'].quantile(0.75)]
        else:
            slow_response_areas = pd.DataFrame()
        
        if not high_risk_locations.empty:
            if 'risk_score' in high_risk_locations.columns:
                top_priority = high_risk_locations.loc[high_risk_locations['risk_score'].idxmax()]
            else:
                top_priority = high_risk_locations.iloc[0]
            
            location_name = top_priority.get('location', 'Unknown Location')
            risk_score = top_priority.get('risk_score', 'N/A')
            complaint_count = top_priority.get('complaint_count', 'N/A')
            
            recommendations.append(
                f"🚨 **Immediate Action**: Allocate additional resources to {location_name} "
                f"(Risk score: {risk_score}, {complaint_count} pending complaints)"
            )
        
        # Seasonal patterns
        if 'timestamp' in complaints_df.columns:
            try:
                complaints_df['month'] = pd.to_datetime(complaints_df['timestamp']).dt.month
                monthly_complaints = complaints_df.groupby('month').size()
                
                if len(monthly_complaints) > 0:
                    peak_month = monthly_complaints.idxmax()
                    peak_count = monthly_complaints.max()
                    recommendations.append(
                        f"📅 **Seasonal Planning**: Prepare for peak demand in month {peak_month} "
                        f"(historically {peak_count} complaints)"
                    )
            except:
                pass
        
        # Resource optimization
        if 'user_rating' in risk_df.columns:
            low_rating_areas = risk_df[risk_df['user_rating'] < risk_df['user_rating'].quantile(0.25)]
            if not low_rating_areas.empty:
                recommendations.append(
                    f"⭐ **Quality Improvement**: Focus on service quality in {len(low_rating_areas)} areas "
                    f"with below-average satisfaction ratings"
                )
        
        # Preventive maintenance
        if 'water_usage' in utility_df.columns and 'location' in utility_df.columns:
            high_usage_areas = utility_df.groupby('location')['water_usage'].mean().nlargest(3)
            if not high_usage_areas.empty:
                recommendations.append(
                    f"🔧 **Preventive Maintenance**: Schedule maintenance in high-usage areas: "
                    f"{', '.join(high_usage_areas.index)}"
                )
        
        # Display recommendations
        if recommendations:
            for i, recommendation in enumerate(recommendations, 1):
                st.success(f"{i}. {recommendation}")
        else:
            st.info("No specific recommendations available. Please check if all required data columns are present.")
        
        # Cost-Benefit Analysis
        st.subheader("💰 Cost-Benefit Insights")
        
        # Calculate potential savings with error handling
        if 'response_time' in risk_df.columns and not high_risk_locations.empty:
            avg_response_time = risk_df['response_time'].mean()
            potential_savings = len(high_risk_locations) * avg_response_time * 100
        else:
            potential_savings = 0
        
        if 'user_rating' in risk_df.columns:
            efficiency_gain = (risk_df['user_rating'].max() - risk_df['user_rating'].min()) * 10
        else:
            efficiency_gain = 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated Monthly Savings", f"₹{potential_savings:,.0f}")
        with col2:
            st.metric("Efficiency Gain Potential", f"+{efficiency_gain:.1f}%")
        
        # ROI Projection
        if potential_savings > 0:
            st.info(
                f"**ROI Projection**: Implementing these recommendations could yield "
                f"~₹{potential_savings * 12:,.0f} annual savings and improve citizen "
                f"satisfaction by ~{efficiency_gain:.1f}%"
            )

def run_business_insights():
    """Streamlit interface for business insights - FIXED VERSION"""
    st.header("📊 Business Intelligence & Analytics")
    
    insights = BusinessInsights()
    
    # Create sample data directly (simpler approach)
    complaints_df = pd.DataFrame({
        'complaint_text': [
            "Water supply issue in Ward 1 affecting multiple households",
            "Street light not working on main road causing safety concerns",
            "Garbage collection delayed for 5 days in residential area",
            "Large pothole on highway damaging vehicles daily",
            "Sewage overflow in park area creating health hazard"
        ],
        'category': ['Water Supply', 'Streetlight', 'Sanitation', 'Road Damage', 'Sewer'],
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
        'user_rating': [4, 3, 2, 5, 1],
        'location': ['Ward 1', 'Ward 2', 'Ward 3', 'Ward 4', 'Ward 5']
    })
    
    utility_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=15, freq='D'),
        'water_usage': [100, 105, 98, 110, 95, 120, 115, 105, 100, 108, 112, 98, 107, 103, 109],
        'pressure_level': [50, 48, 52, 45, 55, 42, 48, 50, 53, 47, 49, 51, 46, 52, 48],
        'energy_consumption': [200, 195, 210, 190, 205, 215, 198, 202, 195, 208, 212, 192, 207, 199, 204],
        'location': ['Zone A'] * 15
    })
    
    risk_df = pd.DataFrame({
        'location': [f'Ward {i}' for i in range(1, 8)],
        'complaint_count': [15, 8, 22, 5, 12, 18, 7],
        'response_time': [3.2, 1.8, 4.5, 1.2, 2.8, 3.8, 1.5],
        'user_rating': [3.8, 4.2, 3.2, 4.5, 3.9, 3.5, 4.1],
        'risk_score': [0.8, 0.3, 0.9, 0.2, 0.6, 0.7, 0.4],
        'risk_level': ['High Risk', 'Low Risk', 'High Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Low Risk']
    })
    
    st.success("✅ Sample data loaded successfully for business insights analysis!")
    
    # Show quick data overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Complaints", len(complaints_df))
    with col2:
        st.metric("Utility Records", len(utility_df))
    with col3:
        st.metric("Risk Areas", len(risk_df))
    
    # Generate comprehensive insights
    insights.generate_comprehensive_insights(complaints_df, utility_df, risk_df)

if __name__ == "__main__":
    run_business_insights()