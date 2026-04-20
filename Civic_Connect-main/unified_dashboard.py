import streamlit as st
import sys
import os

# Add new ML module folder to Python path
sys.path.append('new_ml_civic_monitoring')

# Import new ML modules
from new_ml_civic_monitoring.complaint_analysis import run_complaint_analysis
from new_ml_civic_monitoring.fault_prediction import run_fault_prediction
from new_ml_civic_monitoring.risk_mapping import run_risk_mapping
from new_ml_civic_monitoring.business_insights import run_business_insights


def run_yolo_integration():
    """Run the actual YOLO image prediction with proper integration"""
    try:
        from existing_yolo_modules.yolo_integration import run_yolo_prediction
        run_yolo_prediction()

    except ImportError as e:
        st.error(f"YOLO module import error: {e}")

        st.header("📸 Image Issue Detection")
        st.warning("YOLO integration not available. Using basic fallback.")

        uploaded_file = st.file_uploader("Upload image for analysis", 
                                         type=['jpg', 'jpeg', 'png'])

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.info("YOLO model would run here. Ensure your model file exists inside existing_yolo_modules.")

    except Exception as e:
        st.error(f"Error running YOLO: {e}")
        st.info("Check if best.pt model is available in existing_yolo_modules folder.")


def show_dashboard_overview():
    """Show main dashboard with all module overview"""
    st.header("📊 System Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Existing Modules", "1", "YOLO Image Detection")
    with col2:
        st.metric("New ML Modules", "4", "Complaints, Faults, Risk, Insights")
    with col3:
        st.metric("Integration", "Complete", "Unified Dashboard")

    st.markdown("""
    ### 🚀 Getting Started

    1. **Explore Existing Features** – Use the Image Issue Detection module powered by YOLO  
    2. **Try New ML Modules** – Smart Complaint Analytics, Fault Prediction, Risk Mapping & Insights  
    3. **Upload Real Data** – Use your own CSV data for civic analytics  
    4. **Generate Automated Insights** – Get predictions, risk scores and dashboards instantly
    """)


def main():
    st.set_page_config(
        page_title="Civic AI Monitoring System",
        page_icon="🏙️",
        layout="wide"
    )

    st.title("🏙️ Smart ML-Driven Civic Issue & Resource Monitoring System")

    st.markdown("""
    **Clean & Green Technology – Software Solution**

    *Unified platform combining YOLO image detection with data-driven ML analytics*
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Module",
        [
            "🏠 Dashboard Overview",
            "📸 Image Issue Detection (Existing YOLO)",
            "📝 Complaint Analysis (New ML)",
            "🔮 Fault Prediction (New ML)",
            "🗺️ Risk Mapping (New ML)",
            "📊 Business Insights (New ML)"
        ]
    )

    # Main content routing
    if app_mode == "🏠 Dashboard Overview":
        show_dashboard_overview()

    elif app_mode == "📸 Image Issue Detection (Existing YOLO)":
        run_yolo_integration()

    elif app_mode == "📝 Complaint Analysis (New ML)":
        try:
            from data_management.data_loader import run_complaint_analysis_real
            run_complaint_analysis_real()
        except ImportError:
            run_complaint_analysis()

    elif app_mode == "🔮 Fault Prediction (New ML)":
        run_fault_prediction()  # ✅ FIXED

    elif app_mode == "🗺️ Risk Mapping (New ML)":
        run_risk_mapping()

    elif app_mode == "📊 Business Insights (New ML)":
        run_business_insights()


if __name__ == "__main__":
    main()
