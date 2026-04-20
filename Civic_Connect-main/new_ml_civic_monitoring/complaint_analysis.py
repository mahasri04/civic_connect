import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from textblob import TextBlob
import re

class ComplaintAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = LogisticRegression(random_state=42)
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Clean and preprocess complaint text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def generate_sample_data(self):
        """Generate realistic sample complaint data with balanced classes"""
        sample_complaints = [
            # Water Supply
            "Water supply not working for 3 days in ward 5",
            "Water leakage near community hall causing water wastage",
            "Broken water pipeline on MG road affecting traffic",
            "Low water pressure in apartment building morning and evening",
            # Streetlight  
            "Streetlight broken near main road causing accidents",
            "Dark streets due to non-functional lights in sector 3",
            "Flickering street lights in park area creating safety issues",
            "Complete blackout in residential area due to street light failure",
            # Sanitation
            "Garbage not collected this week in sector 2",
            "Overflowing garbage bins in park area attracting insects",
            "Illegal dumping in vacant lot near school premises",
            "Bad odor from garbage collection point affecting residents",
            # Road Damage
            "Road has large potholes near market area damaging vehicles",
            "Cracked road surface causing traffic congestion during peak hours",
            "Speed breaker damaged creating hazard for two wheelers",
            "Road repair work incomplete leaving debris and dust"
        ]
        
        categories = [
            'Water Supply', 'Water Supply', 'Water Supply', 'Water Supply',
            'Streetlight', 'Streetlight', 'Streetlight', 'Streetlight', 
            'Sanitation', 'Sanitation', 'Sanitation', 'Sanitation',
            'Road Damage', 'Road Damage', 'Road Damage', 'Road Damage'
        ]
        
        df = pd.DataFrame({
            'complaint_text': sample_complaints,
            'category': categories,
            'location': [f'Ward {i%8 + 1}' for i in range(16)],
            'timestamp': pd.date_range('2024-01-01', periods=16, freq='D'),
            'user_rating': np.random.randint(3, 6, 16)
        })
        
        return df
    
    def perform_eda(self, df):
        """Automated Exploratory Data Analysis"""
        st.subheader("📊 Exploratory Data Analysis")
        
        df_eda = df.copy()
        df_eda['complaint_text'] = df_eda['complaint_text'].fillna('Unknown complaint')
        df_eda['text_length'] = df_eda['complaint_text'].apply(lambda x: len(str(x)))
        df_eda['word_count'] = df_eda['complaint_text'].apply(lambda x: len(str(x).split()))
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(df_eda, x='text_length', 
                              title='Complaint Text Length Distribution', 
                              color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.box(df_eda, y='text_length', 
                         title='Text Length Statistics')
            st.plotly_chart(fig2, use_container_width=True)
        
        if 'category' in df_eda.columns:
            df_eda['category'] = df_eda['category'].fillna('Other')
            category_counts = df_eda['category'].value_counts()
            fig3 = px.bar(x=category_counts.index, y=category_counts.values,
                         title='Complaint Category Distribution',
                         labels={'x': 'Category', 'y': 'Count'},
                         color=category_counts.values, 
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig3, use_container_width=True)
            st.write("**Class Distribution:**")
            st.dataframe(category_counts, use_container_width=True)
        
        # Sentiment analysis
        df_eda['sentiment'] = df_eda['complaint_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        
        fig4 = px.histogram(df_eda, x='sentiment', 
                          title='Sentiment Analysis of Complaints', 
                          color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig4, use_container_width=True)
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_sentiment = df_eda['sentiment'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
        with col2:
            avg_text_length = df_eda['text_length'].mean()
            st.metric("Avg Text Length", f"{avg_text_length:.1f} chars")
        with col3:
            unique_categories = df_eda['category'].nunique()
            st.metric("Unique Categories", unique_categories)
        
        return df_eda
    
    def train_model(self, df):
        """Train the classification model"""
        st.subheader("🤖 Model Training & Evaluation")
        
        # Create a working copy
        df_train = df.copy()
        
        # Handle missing values
        df_train['complaint_text'] = df_train['complaint_text'].fillna('')
        df_train['category'] = df_train['category'].fillna('Other')
        
        # Preprocess data
        df_train['cleaned_text'] = df_train['complaint_text'].apply(self.preprocess_text)
        
        # Check class distribution
        class_distribution = df_train['category'].value_counts()
        st.write("**Class Distribution:**")
        st.write(class_distribution)
        
        # Check if we have enough data and balanced classes
        if len(df_train) < 10:
            st.error("Not enough data to train model. Need at least 10 samples.")
            return 0.0
        
        if df_train['category'].nunique() < 2:
            st.error("Need at least 2 different categories to train classification model.")
            return 0.0
        
        try:
            # Split data with stratification
            X = df_train['cleaned_text']
            y = df_train['category']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Check if split was successful
            if len(X_train) == 0 or len(X_test) == 0:
                st.error("Data split failed. Not enough samples.")
                return 0.0
            
            # Vectorize text
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.classifier.fit(X_train_vec, y_train)
            
            # Evaluate
            train_score = self.classifier.score(X_train_vec, y_train)
            test_score = self.classifier.score(X_test_vec, y_test)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_score:.2%}")
            with col2:
                st.metric("Test Accuracy", f"{test_score:.2%}")
            
            # Show classification report
            y_pred = self.classifier.predict(X_test_vec)
            
            st.subheader("📋 Classification Report")
            st.text(classification_report(y_test, y_pred, zero_division=0))
            
            # Show confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            if len(cm) > 0:
                cm_df = pd.DataFrame(cm, 
                                   index=self.classifier.classes_, 
                                   columns=self.classifier.classes_)
                
                st.subheader("🎯 Confusion Matrix")
                fig = px.imshow(cm_df, 
                               title='Confusion Matrix',
                               color_continuous_scale='Blues',
                               aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            
            self.is_trained = True
            st.success("✅ Model trained successfully!")
            
            return test_score
            
        except Exception as e:
            st.error(f"Model training failed: {e}")
            return 0.0


def run_complaint_analysis():
    """Streamlit interface for complaint analysis without prediction feature"""
    st.header("📝 Smart Complaint Analysis")
    
    # Initialize analyzer
    analyzer = ComplaintAnalyzer()
    
    # Load sample data
    df = analyzer.generate_sample_data()
    st.success("✅ Balanced sample data loaded successfully!")
    
    # Show data overview
    st.subheader("📁 Data Overview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Show basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Complaints", len(df))
    with col2:
        st.metric("Categories", df['category'].nunique())
    with col3:
        st.metric("Date Range", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
    
    # EDA Section
    st.markdown("---")
    if st.button("📊 Perform Exploratory Data Analysis"):
        with st.spinner("Analyzing data patterns..."):
            df = analyzer.perform_eda(df)
    
    # Model Training Section
    st.markdown("---")
    st.subheader("🤖 Machine Learning Model")
    
    if st.button("🚀 Train Classification Model"):
        with st.spinner("Training machine learning model..."):
            accuracy = analyzer.train_model(df)
            if accuracy > 0:
                st.success(f"✅ Model training completed with {accuracy:.2%} accuracy!")
    
    # Information section about what the model does
    st.markdown("---")
    st.subheader("💡 About This Analysis")
    
    st.info("""
    **What this module does:**
    - 📊 **Exploratory Data Analysis**: Analyzes complaint patterns, text length, sentiment, and category distribution
    - 🤖 **Machine Learning**: Trains a text classification model to categorize civic complaints
    - 📈 **Performance Metrics**: Evaluates model accuracy, precision, recall, and shows confusion matrix
    
    **Model Output:**
    - Classifies complaints into categories: Water Supply, Streetlight, Sanitation, Road Damage
    - Provides detailed performance metrics and visualizations
    - Helps understand patterns in civic complaints for better resource allocation
    """)


# Run the Streamlit app
if __name__ == "__main__":
    run_complaint_analysis()