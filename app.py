import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Ad Intelligence Platform", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { background-color: #0E1117; color: #FAFAFA; }
div[data-testid="metric-container"] {
    background-color: #161A23; border-radius: 14px; padding: 16px;
}
.stButton>button {
    background-color: #2563EB; color: white;
    border-radius: 10px; height: 3em;
}


</style>
""", unsafe_allow_html=True)

df = pd.read_csv("data/ads_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

age_model = joblib.load("models/age_model.pkl")
device_model = joblib.load("models/device_model.pkl")
location_model = joblib.load("models/location_model.pkl")
cpc_model = joblib.load("models/cpc_model.pkl")
view_model = joblib.load("models/viewtime_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.title(" Ad Performance Intelligence System")

tab1, tab2 = st.tabs(["Home Dashboard", "Strategy & Prediction"])

# =============== HOME TAB ======
with tab1:
    st.subheader(" Global Filters")

    f1, f2, f3 = st.columns(3)
    location_sel = f1.multiselect("Location", df["location"].unique())
    device_sel = f2.multiselect("Device", df["device_type"].unique())
    topic_sel = f3.multiselect("Ad Topic", df["ad_topic"].unique())

    apply = st.button("Apply Filters")

    if apply:
        filtered = df.copy()
        if location_sel:
            filtered = filtered[filtered["location"].isin(location_sel)]
        if device_sel:
            filtered = filtered[filtered["device_type"].isin(device_sel)]
        if topic_sel:
            filtered = filtered[filtered["ad_topic"].isin(topic_sel)]
    else:
        filtered = df.copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg CTR", round(filtered["click_through_rate"].mean(), 4))
    c2.metric("Avg Conversion", round(filtered["conversion_rate"].mean(), 4))
    c3.metric("Avg CPC", round(filtered["cost_per_click"].mean(), 2))
    c4.metric("Avg View Time", round(filtered["view_time"].mean(), 1))

    st.markdown("---")

    # CTR by Device – Bubble (INSIGHTFUL)
    device_ctr = filtered.groupby("device_type").agg(
        ctr=("click_through_rate", "mean"),
        samples=("user_id", "count")
    ).reset_index()

    st.plotly_chart(
        px.scatter(
            device_ctr, x="device_type", y="ctr",
            size="samples", color="device_type",
            title="CTR by Device (Bubble = Sample Size)"
        ),
        use_container_width=True
    )

    #  CTR vs Conversion by Topic – Grouped Bar
    topic_perf = filtered.groupby("ad_topic", as_index=False).agg(
        CTR=("click_through_rate", "mean"),
        Conversion=("conversion_rate", "mean")
    )

    st.plotly_chart(
        px.bar(
            topic_perf, x="ad_topic", y=["CTR", "Conversion"],
            barmode="group",
            title="CTR vs Conversion by Ad Topic"
        ),
        use_container_width=True
    )

    colA, colB = st.columns(2)

    colA.plotly_chart(
        px.pie(filtered, names="content_type",
               title="Content Type Distribution"),
        use_container_width=True
    )

    colB.plotly_chart(
        px.pie(filtered, names="engagement_level",
               title="Engagement Distribution"),
        use_container_width=True
    )

    st.plotly_chart(
        px.choropleth(
            filtered.groupby("location", as_index=False)["click_through_rate"].mean(),
            locations="location",
            locationmode="country names",
            color="click_through_rate",
            title="CTR by Location"
        ),
        use_container_width=True
    )

# ========================== STRATEGY TAB ==========================
with tab2:
    st.subheader(" Ad Strategy Recommendation")

    p1, p2, p3, p4 = st.columns(4)
    content_type = p1.selectbox("Content Type", df["content_type"].unique())
    ad_topic = p2.selectbox("Ad Topic", df["ad_topic"].unique())
    gender = p3.selectbox("Gender", df["gender"].unique())
    audience = p4.selectbox("Target Audience", df["ad_target_audience"].unique())

    if st.button("Generate Strategy"):
        sample = pd.DataFrame([{
            "device_type": encoders["device_type"].transform([df["device_type"].mode()[0]])[0],
            "location": encoders["location"].transform([df["location"].mode()[0]])[0],
            "gender": encoders["gender"].transform([gender])[0],
            "content_type": encoders["content_type"].transform([content_type])[0],
            "ad_topic": encoders["ad_topic"].transform([ad_topic])[0],
            "ad_target_audience": encoders["ad_target_audience"].transform([audience])[0],
            "hour": 12,
            "weekday": 3,
            "click_through_rate": filtered["click_through_rate"].mean(),
            "conversion_rate": filtered["conversion_rate"].mean()
        }])

        age_probs = age_model.predict_proba(sample)[0]
        age_idx = np.argsort(age_probs)[-3:][::-1]
        ages = encoders["age_group"].inverse_transform(age_idx)

        loc_probs = location_model.predict_proba(sample)[0]
        loc_idx = np.argsort(loc_probs)[-3:][::-1]
        locations = encoders["location"].inverse_transform(loc_idx)

        device = encoders["device_type"].inverse_transform(
            device_model.predict(sample)
        )[0]

        cpc = cpc_model.predict(sample)[0]
        view = view_model.predict(sample)[0]

        #  Strategy Summary Cards
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(" Best Device", device)
        s2.metric(" Predicted CPC", f"${round(cpc,2)}")
        s3.metric(" View Time", f"{round(view,1)} sec")
        s4.metric(" Top Location", locations[0])

        st.markdown("###  Strategy Performance Comparison")

        avg_cpc = df["cost_per_click"].mean()
        avg_view = df["view_time"].mean()

        d1, d2 = st.columns(2)

        d1.metric(
            " CPC Improvement",
            f"${round(cpc,2)}",
            delta=f"{round(avg_cpc - cpc,2)} better"
        )

        d2.metric(
            " View Time Gain",
            f"{round(view,1)} sec",
            delta=f"+{round(view - avg_view,1)} sec"
        )

        # Result Table
        result = pd.DataFrame({
            "Age Group": ages,
            "Recommended Location": locations,
            "Confidence (%)": (age_probs[age_idx]*100).round(1)
        })
        st.dataframe(result)

        #  Confidence Bars
        st.markdown("### Confidence Levels")
        for i, age in enumerate(ages):
            st.progress(float(age_probs[age_idx[i]]))
            st.caption(f"{age}: {round(age_probs[age_idx[i]]*100,1)}%")

        
        st.markdown("###  Performance Scores")

        g1, g2 = st.columns(2)

        cpc_score = max(0, min(100, (avg_cpc / cpc) * 50))
        view_score = max(0, min(100, (view / avg_view) * 50))

        g1.plotly_chart(
            px.pie(
                names=["Score", "Remaining"],
                values=[cpc_score, 100 - cpc_score],
                hole=0.7,
                title="Cost Efficiency Score"
            ),
            use_container_width=True
        )

        g2.plotly_chart(
            px.pie(
                names=["Score", "Remaining"],
                values=[view_score, 100 - view_score],
                hole=0.7,
                title="Engagement Score"
            ),
            use_container_width=True
        )


        #  Auto Insights
        st.info(
            f"""
             **{content_type} ads** in **{ad_topic}** category perform best for  
            **{', '.join(ages)}** users.

             Top regions: **{locations[0]}**, **{locations[1]}**  

             Expected CPC is **{round((df['cost_per_click'].mean()-cpc)/df['cost_per_click'].mean()*100,1)}% better**
            than average.

             Strategy prioritizes **high engagement with cost efficiency**.
            """
        )

        st.download_button(
            " Download Strategy",
            result.to_csv(index=False).encode("utf-8"),
            "ad_strategy.csv",
            "text/csv"
        )
