import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Ad Spend Optimizer",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Ad Spend Optimization & Revenue Predictor")
st.write(
    "Predict your campaign revenue and see how different spending levels affect your returns. "
)
st.divider()

@st.cache_resource
def load_models():
    models = {}
    model_paths = {
        "Linear Regression": "model/linear_regression_model.pkl",
        "Random Forest": "model/ad_spend_optimizer.pkl",
        "Decision Tree": "model/decision_tree_model.pkl",
        "XGBoost": "model/xgboost_model.pkl"
    }
    
    for name, path in model_paths.items():
        try:
            models[name] = joblib.load(path)
        except Exception:
            continue
    
    return models

models = load_models()

if not models:
    st.error("No trained models found. Make sure your .pkl files are in the 'model/' folder.")
    st.stop()

# Sidebar for inputs
st.sidebar.header("Campaign Parameters")

acquisition_cost = st.sidebar.number_input("Acquisition Cost ($)", min_value=0.0, value=10000.0)
roi = st.sidebar.number_input("Expected ROI Multiplier", min_value=0.0, value=5.0)
impressions = st.sidebar.number_input("Impressions", min_value=0, value=50000)
clicks = st.sidebar.number_input("Clicks", min_value=0, value=2500)
engagement_score = st.sidebar.number_input("Engagement Score", min_value=0.0, value=80.0, max_value=100.0)
total_spend = st.sidebar.number_input("Total Ad Spend ($)", min_value=0.0, value=10000.0)

st.sidebar.divider()
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# Model performance metrics from training
model_metrics = {
    "Linear Regression": {"r2": 0.78, "mae": 980},
    "Random Forest": {"r2": 0.89, "mae": 750},
    "Decision Tree": {"r2": 0.84, "mae": 920},
    "XGBoost": {"r2": 0.88, "mae": 791}
}

if st.button("Predict Revenue", type="primary"):
    input_data = pd.DataFrame([{
        "Acquisition_Cost": acquisition_cost,
        "ROI": roi,
        "Impressions": impressions,
        "Clicks": clicks,
        "Engagement_Score": engagement_score,
        "total_spend": total_spend
    }])

    # Handle feature alignment
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model.feature_names_in_]

    predicted_revenue = model.predict(input_data)[0]
    calculated_roi = (predicted_revenue - total_spend) / total_spend * 100 if total_spend > 0 else 0

    # Generate optimization curve
    spend_multipliers = np.linspace(0.8, 1.5, 10)
    projected_revenues = []

    for multiplier in spend_multipliers:
        test_data = input_data.copy()
        test_data["total_spend"] = test_data["total_spend"] * multiplier
        projected_revenues.append(model.predict(test_data)[0])

    # Calculate revenue increase with 20% more spend
    baseline_revenue = projected_revenues[3]
    increased_revenue = projected_revenues[5]
    revenue_growth = ((increased_revenue - baseline_revenue) / baseline_revenue) * 100

    # Display results
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Revenue", f"${predicted_revenue:,.0f}")
    col2.metric("Revenue Growth (+20% spend)", f"{revenue_growth:.1f}%")
    col3.metric("Return on Investment", f"{calculated_roi:.1f}%")

    st.divider()

    # Show optimization chart
    st.subheader(f"Spend vs Revenue Analysis")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spend_multipliers * 100, projected_revenues, marker='o', linewidth=2, color='#1f77b4')
    ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Current Spend')
    ax.set_xlabel("Spend Level (% of current budget)")
    ax.set_ylabel("Predicted Revenue ($)")
    ax.set_title(f"Revenue Projection Across Different Spend Levels")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Model info
    st.subheader("Model Information")
    metrics = model_metrics.get(model_name, {"r2": "N/A", "mae": "N/A"})
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Model:** {model_name}")
        st.write(f"**Base Spend:** ${total_spend:,.0f}")
        st.write(f"**Predicted Revenue:** ${predicted_revenue:,.0f}")
    with col_b:
        st.write(f"**R² Score:** {metrics['r2']}")
        st.write(f"**Mean Absolute Error:** ${metrics['mae']:,.0f}")
        st.write(f"**Projected ROI:** {calculated_roi:.1f}%")

st.divider()

