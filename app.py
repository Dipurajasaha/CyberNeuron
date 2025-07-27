import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('API_Tester.html')

CHECKPOINT_DIR = 'Model'
MODEL_FILE = os.path.join(CHECKPOINT_DIR, 'best_model.keras')

try:
    model = load_model(MODEL_FILE)
    print("Model loaded successfully.")
except (ImportError, IOError) as e:
    print(f"Error loading model: {e}")
    exit()

dummy_data = {
    'Destination Port': [0], 'Flow Duration': [0], 'Total Fwd Packets': [0],
    'Total Backward Packets': [0], 'Total Length of Fwd Packets': [0],
    'Total Length of Bwd Packets': [0], 'Fwd Packet Length Max': [0],
    'Fwd Packet Length Min': [0], 'Fwd Packet Length Mean': [0],
    'Fwd Packet Length Std': [0], 'Bwd Packet Length Max': [0],
    'Bwd Packet Length Min': [0], 'Bwd Packet Length Mean': [0],
    'Bwd Packet Length Std': [0], 'Flow Bytes/s': [0], 'Flow Packets/s': [0],
    'Flow IAT Mean': [0], 'Flow IAT Std': [0], 'Flow IAT Max': [0], 'Flow IAT Min': [0],
    'Fwd IAT Total': [0], 'Fwd IAT Mean': [0], 'Fwd IAT Std': [0], 'Fwd IAT Max': [0],
    'Fwd IAT Min': [0], 'Bwd IAT Total': [0], 'Bwd IAT Mean': [0], 'Bwd IAT Std': [0],
    'Bwd IAT Max': [0], 'Bwd IAT Min': [0], 'Fwd PSH Flags': [0], 'Bwd PSH Flags': [0],
    'Fwd URG Flags': [0], 'Bwd URG Flags': [0], 'Fwd Header Length': [0],
    'Bwd Header Length': [0], 'Fwd Packets/s': [0], 'Bwd Packets/s': [0],
    'Min Packet Length': [0], 'Max Packet Length': [0], 'Packet Length Mean': [0],
    'Packet Length Std': [0], 'Packet Length Variance': [0], 'FIN Flag Count': [0],
    'SYN Flag Count': [0], 'RST Flag Count': [0], 'PSH Flag Count': [0], 'ACK Flag Count': [0],
    'URG Flag Count': [0], 'CWE Flag Count': [0], 'ECE Flag Count': [0], 'Down/Up Ratio': [0],
    'Average Packet Size': [0], 'Avg Fwd Segment Size': [0], 'Avg Bwd Segment Size': [0],
    'Fwd Header Length.1': [0], 'Fwd Avg Bytes/Bulk': [0], 'Fwd Avg Packets/Bulk': [0],
    'Fwd Avg Bulk Rate': [0], 'Bwd Avg Bytes/Bulk': [0], 'Bwd Avg Packets/Bulk': [0],
    'Bwd Avg Bulk Rate': [0], 'Subflow Fwd Packets': [0], 'Subflow Fwd Bytes': [0],
    'Subflow Bwd Packets': [0], 'Subflow Bwd Bytes': [0], 'Init_Win_bytes_forward': [0],
    'Init_Win_bytes_backward': [0], 'act_data_pkt_fwd': [0], 'min_seg_size_forward': [0],
    'Active Mean': [0], 'Active Std': [0], 'Active Max': [0], 'Active Min': [0], 'Idle Mean': [0],
    'Idle Std': [0], 'Idle Max': [0], 'Idle Min': [0]
}

dummy_df = pd.DataFrame(dummy_data)
feature_names = list(dummy_df.columns)

ss = StandardScaler()
ss.fit(dummy_df)

dummy_labels = ['BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration', 
                'Web Attack � Brute Force', 'Web Attack � XSS', 
                'Web Attack � Sql Injection', 'FTP-Patator', 'SSH-Patator', 
                'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 
                'DoS GoldenEye', 'Heartbleed']
le = LabelEncoder()
le.fit(dummy_labels)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame(data, index=[0])        
        input_df = input_df[feature_names]
        scaled_data = ss.transform(input_df)
        reshaped_data = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))
        prediction = model.predict(reshaped_data)       
        predicted_index = np.argmax(prediction, axis=1)      
        predicted_label = le.inverse_transform(predicted_index)
        return jsonify({'prediction': predicted_label[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
