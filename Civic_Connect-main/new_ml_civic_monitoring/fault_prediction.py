import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# ======================================================================
#                     FAULT PREDICTION CLASS
# ======================================================================

class FaultPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(n_estimators=120, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.08, random_state=42)
        self.is_trained = False

    # --------------------- SAMPLE DATA GENERATOR -----------------------

    def generate_sample_utility_data(self):
        dates = pd.date_range('2024-01-01', '2024-06-01', freq='D')
        data = []

        for i, date in enumerate(dates):
            seasonal = 12 * np.sin(2 * np.pi * i / 365)
            trend = 0.06 * i
            noise = np.random.normal(0, 2)

            water_usage = 100 + seasonal + trend + noise
            pressure_level = 52 + 4 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 2)
            energy_consumption = 210 + 15 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 4)

            if i in [45, 46, 47, 120, 121]:  # simulated faults
                water_usage *= 1.45
                pressure_level *= 0.4

            data.append({
                'timestamp': date,
                'water_usage': max(water_usage, 0),
                'pressure_level': max(pressure_level, 0),
                'energy_consumption': max(energy_consumption, 0),
                'location': f"Zone {i % 5 + 1}"
            })

        return pd.DataFrame(data)

    # ------------------------ FEATURE ENGINEERING ----------------------

    def create_features(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear

        df['usage_roll_mean_7'] = df['water_usage'].rolling(window=7).mean()
        df['usage_roll_std_7'] = df['water_usage'].rolling(window=7).std()
        df['pressure_roll_mean_7'] = df['pressure_level'].rolling(window=7).mean()

        df['usage_lag_1'] = df['water_usage'].shift(1)
        df['usage_lag_7'] = df['water_usage'].shift(7)

        return df.dropna()

    # ------------------------ EXPLORATORY ANALYSIS ---------------------

    def perform_eda(self, df):
        st.subheader("📊 Utility Data Overview")

        fig1 = px.line(df, x='timestamp', y='water_usage',
                       title='Water Usage Over Time')
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig2 = px.histogram(df, x='water_usage',
                                title='Distribution of Water Usage')
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = px.box(df, y='water_usage',
                          title='Water Usage Box Plot')
            st.plotly_chart(fig3, use_container_width=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()

        fig4 = px.imshow(
            corr,
            title='Feature Correlation Heatmap',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig4, use_container_width=True)

        return df

    # ------------------------ ANOMALY DETECTION ------------------------

    def detect_anomalies(self, df):
        st.subheader("🚨 Anomaly Detection")

        features = ['water_usage', 'pressure_level', 'energy_consumption']
        X = df[features].fillna(method='ffill')

        X_scaled = self.scaler.fit_transform(X)

        anomalies = self.anomaly_detector.fit_predict(X_scaled)

        df['anomaly'] = anomalies == -1
        df['anomaly_score'] = self.anomaly_detector.decision_function(X_scaled)

        fig = px.scatter(
            df,
            x='timestamp',
            y='water_usage',
            color='anomaly',
            color_discrete_map={False: 'blue', True: 'red'},
            title="Detected Anomalies in Water Usage"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Total Anomalies Detected", df['anomaly'].sum())

        if df['anomaly'].sum() > 0:
            st.write("⚠️ **Anomalous Records:**")
            st.dataframe(df[df['anomaly']], use_container_width=True)

        return df

    # ------------------------ MODEL TRAINING ---------------------------

    def train_prediction_model(self, df):
        st.subheader("🤖 Fault Prediction Model Training")

        df_en = self.create_features(df)

        features = [
            'hour', 'day_of_week', 'month', 'day_of_year',
            'usage_roll_mean_7', 'usage_roll_std_7',
            'pressure_roll_mean_7', 'usage_lag_1', 'usage_lag_7'
        ]

        X = df_en[features]
        y = df_en['water_usage']

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.rf_model.fit(X_train, y_train)

        y_pred = self.rf_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")

        res_df = pd.DataFrame({
            'Timestamp': df_en['timestamp'].iloc[split:],
            'Actual': y_test.values,
            'Predicted': y_pred
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_df['Timestamp'], y=res_df['Actual'],
                                 name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=res_df['Timestamp'], y=res_df['Predicted'],
                                 name='Predicted', line=dict(color='red')))
        fig.update_layout(title="Actual vs Predicted Water Usage")
        st.plotly_chart(fig, use_container_width=True)

        self.is_trained = True
        return mae, rmse

# ======================================================================
#                       STREAMLIT UI FUNCTION
# ======================================================================

def run_fault_prediction():
    st.header("🔮 Smart Fault Prediction System")

    predictor = FaultPredictor()

    st.subheader("📂 Loading Utility Data...")
    df = predictor.generate_sample_utility_data()
    st.success("Sample data loaded successfully!")

    st.dataframe(df.head(), use_container_width=True)

    df = predictor.perform_eda(df)
    df = predictor.detect_anomalies(df)

    if st.button("🚀 Train Fault Prediction Model"):
        with st.spinner("Training..."):
            mae, rmse = predictor.train_prediction_model(df)
        st.success(f"Model Trained Successfully! MAE={mae:.2f}, RMSE={rmse:.2f}")

    st.subheader("⏳ Future Forecast (Coming Soon)")
    st.info("Future forecasting will be enabled when real-time data is added.")
