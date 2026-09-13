import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Online Retail Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)


# ============================================================
# 2. Model Path
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Online-retail.joblib"
)


# ============================================================
# 3. Load Model
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:

    artifacts = load_model()

except Exception as e:

    st.error("❌ Failed to load the model.")
    st.exception(e)
    st.stop()


# ============================================================
# 4. Extract Model Components
# ============================================================

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]

transformation = artifacts.get(
    "transformation",
    "log1p"
)

reference_date = artifacts.get(
    "reference_date",
    None
)

cluster_mapping = artifacts.get(
    "cluster_mapping",
    {}
)

silhouette_score_value = artifacts.get(
    "silhouette_score",
    None
)


# ============================================================
# 5. Header
# ============================================================

st.title("🛍️ Online Retail Customer Segmentation")

st.markdown(
    """
    ### RFM + Customer Behavioral Segmentation

    This application uses **K-Means clustering** to identify
    customer groups based on purchasing behavior.
    """
)


# ============================================================
# 6. Model Information
# ============================================================

with st.expander("ℹ️ Model Information"):

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Algorithm",
            "K-Means"
        )

    with col2:

        st.metric(
            "Clusters",
            model.n_clusters
        )

    with col3:

        st.metric(
            "Features",
            len(features)
        )

    with col4:

        if silhouette_score_value is not None:

            st.metric(
                "Silhouette",
                f"{silhouette_score_value:.3f}"
            )

    st.write("### Features used by the model")

    st.write(", ".join(features))

    if reference_date is not None:

        st.write(
            f"Reference Date: **{reference_date}**"
        )


# ============================================================
# 7. Customer Input
# ============================================================

st.header("👤 Customer Profile")

st.write(
    "Enter the customer's purchasing behavior."
)


# ------------------------------------------------------------
# RFM Features
# ------------------------------------------------------------

st.subheader("📊 RFM Features")

col1, col2, col3 = st.columns(3)

with col1:

    recency = st.number_input(
        "Recency",
        min_value=0.0,
        value=30.0,
        step=1.0,
        help="Days since the customer's last purchase."
    )

with col2:

    frequency = st.number_input(
        "Frequency",
        min_value=1.0,
        value=5.0,
        step=1.0,
        help="Number of unique invoices."
    )

with col3:

    monetary = st.number_input(
        "Monetary",
        min_value=0.0,
        value=500.0,
        step=10.0,
        help="Total amount spent."
    )


# ------------------------------------------------------------
# Behavioral Features
# ------------------------------------------------------------

st.subheader("🛒 Behavioral Features")

col1, col2, col3 = st.columns(3)

with col1:

    total_quantity = st.number_input(
        "Total Quantity",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

with col2:

    unique_products = st.number_input(
        "Unique Products",
        min_value=1.0,
        value=10.0,
        step=1.0
    )

with col3:

    average_order_value = st.number_input(
        "Average Order Value",
        min_value=0.0,
        value=100.0,
        step=5.0
    )


col1, col2, col3 = st.columns(3)

with col1:

    avg_quantity_order = st.number_input(
        "Average Quantity / Order",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

with col2:

    avg_unit_price = st.number_input(
        "Average Unit Price",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

with col3:

    lifetime_days = st.number_input(
        "Customer Lifetime (Days)",
        min_value=0.0,
        value=180.0,
        step=1.0
    )


col1, col2, col3 = st.columns(3)

with col1:

    avg_days_between = st.number_input(
        "Average Days Between Orders",
        min_value=0.0,
        value=30.0,
        step=1.0
    )

with col2:

    purchase_rate = st.number_input(
        "Purchase Rate",
        min_value=0.0,
        value=0.05,
        step=0.01
    )

with col3:

    cancellation_rate = st.number_input(
        "Cancellation Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01
    )


weekend_ratio = st.number_input(
    "Weekend Purchase Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01
)


# ============================================================
# 8. Prediction Button
# ============================================================

st.divider()

predict = st.button(
    "🚀 Predict Customer Segment",
    type="primary",
    use_container_width=True
)


# ============================================================
# 9. Prediction
# ============================================================

if predict:

    try:

        # ----------------------------------------------------
        # Create Input DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                recency,
                frequency,
                monetary,
                total_quantity,
                unique_products,
                average_order_value,
                avg_quantity_order,
                avg_unit_price,
                lifetime_days,
                avg_days_between,
                purchase_rate,
                cancellation_rate,
                weekend_ratio
            ]],
            columns=features
        )


        # ----------------------------------------------------
        # Apply Same Transformation Used During Training
        # ----------------------------------------------------

        if transformation == "log1p":

            input_transformed = np.log1p(
                input_data
            )

        else:

            input_transformed = input_data.copy()


        # ----------------------------------------------------
        # Apply Saved Scaler
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_transformed
        )


        # ----------------------------------------------------
        # Predict Cluster
        # ----------------------------------------------------

        cluster = int(
            model.predict(input_scaled)[0]
        )


        # ----------------------------------------------------
        # Get Business Segment
        # ----------------------------------------------------

        segment = cluster_mapping.get(
            cluster,
            cluster_mapping.get(
                str(cluster),
                f"Cluster {cluster}"
            )
        )


        # ====================================================
        # 10. Prediction Result
        # ====================================================

        st.success(
            "✅ Customer segmentation completed!"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Customer Cluster",
                f"Cluster {cluster}"
            )

        with result_col2:

            st.metric(
                "Business Segment",
                segment
            )


        # ====================================================
        # 11. Customer Profile
        # ====================================================

        st.subheader("📋 Customer RFM & Behavioral Profile")

        display_data = input_data.T.reset_index()

        display_data.columns = [
            "Feature",
            "Value"
        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # 12. Business Recommendation
        # ====================================================

        st.subheader("💡 Business Recommendation")

        recommendations = {

            "High Value": """
            ⭐ **High Value Customer**

            - Provide VIP treatment.
            - Offer personalized promotions.
            - Encourage repeat purchases.
            - Protect this customer from churn.
            """,

            "Loyal": """
            ❤️ **Loyal Customer**

            - Reward loyalty.
            - Offer loyalty-program benefits.
            - Recommend complementary products.
            - Increase purchase frequency.
            """,

            "At Risk": """
            ⚠️ **At Risk Customer**

            - Launch a re-engagement campaign.
            - Provide targeted offers.
            - Recommend relevant products.
            - Monitor future purchasing activity.
            """,

            "Regular": """
            👤 **Regular Customer**

            - Encourage higher purchase frequency.
            - Use personalized recommendations.
            - Test promotional offers.
            - Move the customer toward the Loyal segment.
            """
        }

        recommendation = recommendations.get(
            segment,
            f"""
            **{segment}**

            Analyze the customer's RFM and behavioral
            characteristics to determine the best action.
            """
        )

        st.info(recommendation)


    # ========================================================
    # Error Handling
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# 13. Footer
# ============================================================

st.divider()

st.caption(
    "Made by Eng.Ali Ahmed Zaki"
)
