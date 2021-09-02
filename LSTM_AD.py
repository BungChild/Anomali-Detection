"""The App."""

import pandas as pd
import numpy as np
import streamlit as st
from tensorflow import keras
import pickle
import plotly.graph_objects as go
import base64

# Create sliding window
def sliding_window(data, window_size):
    sub_seq, next_values = [], []
    for i in range(len(data)-window_size):
        sub_seq.append(data[i:i+window_size])
        next_values.append(data[i+window_size])
    X = np.stack(sub_seq)
    y = np.array(next_values)
    return X,y

window_size = 30

# Load the model from the file
model = keras.models.load_model('anomaly_detection')

# load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

threshold = 527.8798828125

st.write("""
# LSTM Anomaly Detection App for Web Traffic Data
""")

st.write("""
### Data format and must be greater than 30 timestamps
| timestamp  | value   |
| -----------|:-------:|
| 1          | 10      |
| 2          | 7       |
| 3          | 17      |
""")

uploaded_file = st.file_uploader("Choose a file", type='csv')

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['scaled'] = scaler.transform(df[['value']])

        X, y = sliding_window(df[['scaled']].values, window_size)

        predict = scaler.inverse_transform(model.predict(X))
        y = scaler.inverse_transform(y)

        abs_error = np.abs(y - predict)

        st.sidebar.header('Customize Threshold')
        form = st.sidebar.form("my_form")
        threshold = form.slider('Threshold',
                                min_value=abs_error.min(),
                                max_value=abs_error.max(),
                                value=threshold)
        form.form_submit_button("Apply")

        test_anomaly = pd.DataFrame()
        test_anomaly['timestamp'] = df['timestamp'][window_size:]
        test_anomaly['value'] = df['value'][window_size:]
        test_anomaly['abs_error'] = abs_error
        test_anomaly['anomaly_hat'] = 0
        test_anomaly.loc[test_anomaly['abs_error'] >= threshold, 'anomaly_hat'] = 1

        anomalies = test_anomaly.loc[test_anomaly['anomaly_hat'] == 1]

        st.write("Visualize Detected Anomalies from Data")  
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_anomaly['timestamp'], y=test_anomaly['value'], name='value'))
        fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['value'], mode='markers', name='Anomaly'))
        st.plotly_chart(fig)

        st.write("Anomalies Data")
        st.write(anomalies)

        csv = anomalies.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  
        link= f'<a href="data:file/csv;base64,{b64}" download="anomalies.csv">Download</a>'
        st.markdown(link, unsafe_allow_html=True)
    except:
        st.write("# Error!!!")