import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title='Spotify Premium Predictor', layout='centered')
st.title('🎧 Spotify Premium Willingness Predictor')

# Load data
df = pd.read_excel('./data/spotify_data.xlsx')
st.subheader('📄 Data Preview')
st.dataframe(df)

# Define features and target
target = 'premium_sub_willingness'
features = [col for col in df.columns if col != target]

# Encode categorical features
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

# Split data
X = df_encoded[features]
y = df_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_type = st.selectbox('Choose model type:', ['Decision Tree', 'Logistic Regression', 'Random Forest'], help='Select the model.')

# --- Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None

# Train model button
if st.button('Train Model'):
    if model_type == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=5)
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=50)
    else:
        st.error('Invalid model type selected.')
        model = None

    if model:
        model.fit(X_train, y_train)
        st.session_state.model = model
        st.session_state.model_trained = True
        st.success('✅ Model trained and stored in session!')

# If model already trained, show metrics and allow prediction
if st.session_state.model_trained:
    model = st.session_state.model
    predictions = model.predict(X_test)
    st.subheader('📈 Predictions on Test Set:')
    st.markdown(f'**Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%**')

    # --- Predict for new user
    st.subheader('🔮 Predict for a New User')
    new_user = {}
    for feature in features:
        value = st.selectbox(f'Select value for {feature}:', options=df[feature].unique(), key=feature, help=f'Select a value for the feature `{feature}` from the dataset.')
        new_user[feature] = value

    if st.button('Predict New User'):
        new_user_df = pd.DataFrame([new_user])
        new_user_encoded = new_user_df.copy()

        # Encode categorical
        for col in new_user_encoded.columns:
            if new_user_encoded[col].dtype == 'object':
                new_user_encoded[col] = LabelEncoder().fit_transform(new_user_encoded[col].astype(str))

        prediction = model.predict(new_user_encoded)
        st.markdown(f'### 🎯 Prediction: **{"Yes" if prediction[0] == 1 else "No"}**')
