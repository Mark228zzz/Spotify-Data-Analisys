# 🎧 Spotify Data Analysis

This project analyzes Spotify user behavior to:
- Understand what drives Premium subscription
- Predict who is likely to subscribe
- Segment users using K-Means clustering
- Visualize segments using t-SNE
- Generate clear business personas and strategies

## 🧠 Tools & Techniques
- Python (pandas, sklearn, seaborn, matplotlib)
- K-Means Clustering + t-SNE
- Streamlit dashboard for interaction
- Logistic Regression, Decision Trees
- SMOTE for class imbalance
- The dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/meeraajayakumar/spotify-user-behavior-dataset)

## 📊 Key Insights
- **82%** of users use Free plan
- Most of users are **20** to **35** years old
- Melody lovers listening at night are more likely to subscribe
- **20-35** years old users the most willing to subscribe
- **50%** of **12-20** years old users will to subscribe
- The most preffed Premium Plan is `Individual`
- Segment 3 ("Heavy Hybrids") is high-value and highly active

## 📁 Files
- `notebook/spotify_analysis.ipynb`: Full EDA and ML
- `app/spotify_predictor_app.py`: Interactive Streamlit tool
- `data/Spotify_data.xlsx`: Dataset

## 💡 Business Use
This project simulates real work at Spotify or similar services:
- Data-driven personalization
- Premium targeting strategies
- Clear visual storytelling

## 🔥 Try it
1. **Clone this repo:**
```bash
git clone https://github.com/Mark228zzz/Spotify-Data-Analisys
```

2. **Setup the environment:**
```bash
python3 -m venv NAME_YOUR_ENV

source NAME_YOUR_ENV/bin/activate # For Linux/MacOS

env\Scripts\activate.bat # For Windows
```

3. **Install all requirements:**
```bash
pip install -r requirements.txt
```

4. **Run Streamlit app:**

```bash
streamlit run app/spotify_predictor_app.py
```

## 👤 Author & Links

- [LinkedIn](https://www.linkedin.com/in/mark-mazur123/)
- [Medium](https://medium.com/@mark.mazur)
- [GitHub](https://github.com/Mark228zzz)

## 📄 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
