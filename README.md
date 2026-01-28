📊 Ad Performance Intelligence System

An end-to-end data-driven ad analytics and strategy recommendation platform built using Streamlit, Machine Learning, and Interactive Visual Analytics.
The system enables marketers and analysts to analyze ad performance, identify audience behavior patterns, and generate optimized advertising strategies using predictive models.

                                     🚀 Features
🔍 Interactive Analytics Dashboard
Global filtering by location, device, and ad topic
KPI tracking:
Click-Through Rate (CTR)
Conversion Rate
Cost Per Click (CPC)
View Time
Dynamic visualizations:
CTR by device (bubble chart)
CTR vs Conversion by ad topic
Engagement & content distribution (pie charts)
Geographical CTR heatmap

                         🧠 AI-Driven Strategy Recommendation

Predicts:
Best performing device
Top age groups
High-confidence locations
Expected CPC
Estimated view time
Strategy insights generated using trained ML models
Performance benchmarking against historical averages
Confidence-based recommendations for decision support

                          📈 Performance Scoring

Cost Efficiency Score
Engagement Score
Visual confidence indicators for predictions

                             🏗️ Tech Stack
Category	Tools
Frontend	Streamlit
Data Processing	Pandas, NumPy
Visualization	Plotly
Machine Learning	Scikit-learn
Model Persistence	Joblib
Styling	Custom CSS
📂 Project Structure
ad-performance-intelligence/
│
├── app.py
├── data/
│   └── ads_data.csv
│
├── models/
│   ├── age_model.pkl
│   ├── device_model.pkl
│   ├── location_model.pkl
│   ├── cpc_model.pkl
│   ├── viewtime_model.pkl
│   └── encoders.pkl
│
├── requirements.txt
└── README.md
              Installation & Setup
1️⃣ Clone Repository
**git clone https://github.com/your-username/ad-performance-intelligence.git
cd ad-performance-intelligence
**
2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Application
streamlit run app.py

📊 Machine Learning Models

Classification Models

Age Group Prediction

Device Recommendation

Location Optimization

Regression Models

Cost Per Click (CPC)

View Time Estimation

Models are trained offline and loaded using joblib for real-time inference.

📥 Outputs

Interactive dashboards

Strategy recommendation cards

Downloadable CSV strategy report

Confidence-based prediction tables

🎯 Use Cases

Marketing analytics & optimization

Digital ad campaign planning

Audience targeting strategy

Cost efficiency improvement

Engagement maximization

🧠 Key Highlights

End-to-end ML-powered decision support system

Real-time visual analytics with business KPIs

Explainable strategy recommendations

Production-ready Streamlit interface

👤 Author

Udit Katiyar
Data Analyst | Machine Learning Enthusiast
