import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Online Retail Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)


# ============================================================
# 2. Application Header
# ============================================================

st.title("🛍️ Online Retail Customer Segmentation")

st.markdown(
    """
    ## RFM + K-Means

    This application predicts a customer's segment based on:

    - **Recency** → How recently the customer purchased
    - **Frequency** → How frequently the customer purchased
    - **Monetary** → How much the customer spent
    """
)


# ============================================================
# 3. Model Path
# ============================================================

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# The model is located in the same directory as app.py
MODEL_PATH = os.path.join(
    BASE_DIR,
    "Online-retail.joblib"
)


# ============================================================
# 4. Load Saved Model
# ============================================================

@st.cache_resource
def load_model():

    # Check whether the model file exists
    if not os.path.isfile(MODEL_PATH):

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Load the saved joblib file
    return joblib.load(MODEL_PATH)


# ============================================================
# 5. Load Model Safely
# ============================================================

try:

    artifacts = load_model()

except Exception as e:

    st.error(
        "❌ Failed to load the model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 6. Extract Saved Model Components
# ============================================================

model = artifacts["model"]

scaler = artifacts["scaler"]

features = artifacts["features"]

transformation = artifacts.get(
    "transformation",
    "log1p"
)

segment_mapping = artifacts.get(
    "segment_mapping",
    {}
)

reference_date = artifacts.get(
    "reference_date",
    None
)


# ============================================================
# 7. Sidebar - Model Information
# ============================================================

st.sidebar.header(
    "⚙️ Model Information"
)

st.sidebar.write(
    "**Algorithm:** K-Means"
)

st.sidebar.write(
    f"**Number of Clusters:** {model.n_clusters}"
)

st.sidebar.write(
    f"**Features:** {', '.join(features)}"
)

st.sidebar.write(
    f"**Transformation:** {transformation}"
)

if reference_date is not None:

    st.sidebar.write(
        f"**Reference Date:** {reference_date}"
    )


# ============================================================
# 8. Customer Input Section
# ============================================================

st.header(
    "👤 Customer RFM Information"
)

st.write(
    "Enter the customer's RFM values "
    "to predict the customer segment."
)


# Create three columns
col1, col2, col3 = st.columns(3)


# ============================================================
# 9. Recency Input
# ============================================================

with col1:

    recency = st.number_input(
        "📅 Recency",
        min_value=0.0,
        value=30.0,
        step=1.0,
        help=(
            "Number of days since the customer's "
            "last purchase."
        )
    )


# ============================================================
# 10. Frequency Input
# ============================================================

with col2:

    frequency = st.number_input(
        "🛒 Frequency",
        min_value=0.0,
        value=5.0,
        step=1.0,
        help=(
            "Number of unique invoices/purchases "
            "made by the customer."
        )
    )


# ============================================================
# 11. Monetary Input
# ============================================================

with col3:

    monetary = st.number_input(
        "💰 Monetary",
        min_value=0.0,
        value=500.0,
        step=10.0,
        help=(
            "Total amount spent by the customer."
        )
    )


# ============================================================
# 12. Prediction Button
# ============================================================

predict_button = st.button(
    "🚀 Predict Customer Segment",
    use_container_width=True
)


# ============================================================
# 13. Prediction Process
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Step 1: Create Input DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                recency,
                frequency,
                monetary
            ]],
            columns=features
        )


        # ----------------------------------------------------
        # Step 2: Apply Log Transformation
        # ----------------------------------------------------

        if transformation == "log1p":

            input_transformed = np.log1p(
                input_data
            )

        else:

            input_transformed = input_data.copy()


        # ----------------------------------------------------
        # Step 3: Apply Saved StandardScaler
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_transformed
        )


        # ----------------------------------------------------
        # Step 4: Predict Cluster
        # ----------------------------------------------------

        cluster = model.predict(
            input_scaled
        )[0]


        # Convert NumPy integer to Python integer
        cluster = int(cluster)


        # ----------------------------------------------------
        # Step 5: Convert Cluster to Business Segment
        # ----------------------------------------------------

        segment = segment_mapping.get(
            cluster,
            segment_mapping.get(
                str(cluster),
                f"Cluster {cluster}"
            )
        )


        # ====================================================
        # 14. Prediction Result
        # ====================================================

        st.success(
            "✅ Prediction completed successfully!"
        )


        result_col1, result_col2 = st.columns(2)


        # ----------------------------------------------------
        # Cluster Result
        # ----------------------------------------------------

        with result_col1:

            st.metric(
                label="Cluster",
                value=str(cluster)
            )


        # ----------------------------------------------------
        # Segment Result
        # ----------------------------------------------------

        with result_col2:

            st.metric(
                label="Customer Segment",
                value=segment
            )


        # ====================================================
        # 15. Customer RFM Profile
        # ====================================================

        st.subheader(
            "📋 Customer RFM Profile"
        )


        rfm_summary = pd.DataFrame(
            {
                "Metric": [
                    "Recency",
                    "Frequency",
                    "Monetary"
                ],

                "Value": [
                    recency,
                    frequency,
                    monetary
                ]
            }
        )


        st.dataframe(
            rfm_summary,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # 16. Business Recommendation
        # ====================================================

        st.subheader(
            "💡 Business Recommendation"
        )


        # ----------------------------------------------------
        # High Value Customers
        # ----------------------------------------------------

        if segment == "High Value Customers":

            st.info(
                """
                ⭐ **High Value Customer**

                Recommended actions:

                - Provide VIP benefits
                - Offer personalized promotions
                - Encourage repeat purchases
                - Maintain strong customer engagement
                """
            )


        # ----------------------------------------------------
        # Loyal Customers
        # ----------------------------------------------------

        elif segment == "Loyal Customers":

            st.info(
                """
                ❤️ **Loyal Customer**

                Recommended actions:

                - Reward customer loyalty
                - Offer loyalty program benefits
                - Provide personalized offers
                - Encourage increased purchase frequency
                """
            )


        # ----------------------------------------------------
        # At Risk Customers
        # ----------------------------------------------------

        elif segment == "At Risk Customers":

            st.warning(
                """
                ⚠️ **At Risk Customer**

                Recommended actions:

                - Launch re-engagement campaigns
                - Provide targeted discounts
                - Recommend relevant products
                - Encourage the customer to return
                """
            )


        # ----------------------------------------------------
        # Other Segments
        # ----------------------------------------------------

        else:

            st.info(
                f"""
                The customer belongs to:

                **{segment}**

                Additional business analysis can be
                performed using the customer's RFM profile.
                """
            )


    # ========================================================
    # 17. Prediction Error Handling
    # ========================================================

    except Exception as e:

        st.error(
            "❌ An error occurred during prediction."
        )

        st.exception(e)


# ============================================================
# 18. Footer
# ============================================================

st.divider()

st.caption(
    "Online Retail Customer Segmentation | "
    "RFM + K-Means | "
    "Made by Eng Ali Zaki"
)
