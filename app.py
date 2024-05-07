import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import tensorflow as tf



app=Flask(__name__)
model = tf.keras.models.load_model('cnnmultimodel.h5', compile=False)
optimizer = tf.keras.optimizers.Adam()  # Example: Use Adam optimizer with default settings
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
stdscaler=pickle.load(open('standardize.pkl','rb'))
encoder=pickle.load(open('encoder.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    new_data=stdd(data)
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

def stdd(data):
    column_names = ['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'tcprtt',
       'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'response_body_len', 'ct_state_ttl', 'ct_src_dport_ltm',
       'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
       'ct_src_ltm','ct_srv_dst', 'is_sm_ips_ports', 'proto', 'service', 'state'
       ]
    df = pd.DataFrame([data], columns=column_names)
    nums = df.drop(columns=['proto', 'service', 'state'])
    nums= nums.astype(float)
    scaler_columns=['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
        'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'tcprtt',
        'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
        'response_body_len', 'ct_state_ttl', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
        'ct_src_ltm','ct_srv_dst',  'is_sm_ips_ports']
    nums= nums[scaler_columns].copy()
    nums.columns = scaler_columns
    new_cat=pd.DataFrame(encoder.transform([[df['proto'][0], df['service'][0], df['state'][0]
    ]]),columns=['proto', 'service', 'state'])
    new_cat=new_cat.astype(float)
    new_nums=stdscaler.transform(nums)
    new_columns = ['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'tcprtt',
       'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'response_body_len', 'ct_state_ttl', 'ct_src_dport_ltm',
       'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
       'ct_src_ltm','ct_srv_dst',  'is_sm_ips_ports', 'proto', 'service', 'state']
    new_data=pd.concat([new_cat,pd.DataFrame(new_nums,columns=scaler_columns)],axis=1)
    new_data = new_data[new_columns].copy()
    return new_data


def csvstdd(data):
    column_names = ['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'tcprtt',
       'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'response_body_len', 'ct_state_ttl', 'ct_src_dport_ltm',
       'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
       'ct_src_ltm','ct_srv_dst', 'is_sm_ips_ports', 'proto', 'service', 'state'
       ]
    df = pd.DataFrame(data, columns=column_names)
    nums = df.drop(columns=['proto', 'service', 'state'])
    nums= nums.astype(float)
    scaler_columns=['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
        'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'tcprtt',
        'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
        'response_body_len', 'ct_state_ttl', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
        'ct_src_ltm','ct_srv_dst',  'is_sm_ips_ports']
    nums= nums[scaler_columns].copy()
    nums.columns = scaler_columns
    # new_cat=pd.DataFrame(encoder.transform([[df['proto'][0], df['service'][0], df['state'][0]
    # ]]),columns=['proto', 'service', 'state'])
    cat_columns = ['proto', 'service', 'state']
    new_cat = pd.DataFrame(encoder.transform(df[cat_columns]), columns=cat_columns)
    new_cat=new_cat.astype(float)
    new_nums=stdscaler.transform(nums)
    new_columns = ['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'tcprtt',
       'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'response_body_len', 'ct_state_ttl', 'ct_src_dport_ltm',
       'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
       'ct_src_ltm','ct_srv_dst',  'is_sm_ips_ports', 'proto', 'service', 'state']
    new_data=pd.concat([new_cat,pd.DataFrame(new_nums,columns=scaler_columns)],axis=1)
    new_data = new_data[new_columns].copy()
    return new_data


@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    final_input=stdd(data)
    print(final_input)
    output=model.predict(final_input)[0]
    ans=0
    for i in range(len(output)):
        if(output[i]==1):
            ans=i
    if (ans == 6):
        return render_template("home.html",prediction_text="Safe Not an Intrusion : Normal")    
    class_lables=['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic',
       'Normal', 'Reconnaissance', 'Shellcode', 'Worms']
    return render_template("home.html",prediction_text="Alert! It is an Intrusion, Attack type is {}".format(class_lables[ans]))

@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/predict')
def predicthtml():
    return render_template('predict.html')

@app.route('/csv', methods=['GET', 'POST'])
def csv_predict():
    if request.method == 'POST':
        csv_file = request.files['csv_file']
        if csv_file:
            # Save the uploaded CSV file to a temporary location
            csv_path = 'uploads/' + csv_file.filename
            csv_file.save(csv_path)
            dff = pd.read_csv(csv_path)
            df = csvstdd(dff)
            predictions = predict_from_csv(df)
            predictions_array = np.array(predictions)
            predicted_class_indices = np.argmax(predictions_array, axis=1)
            class_labels = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic',
            'Normal', 'Reconnaissance', 'Shellcode', 'Worms']
            predicted_labels = [class_labels[idx] for idx in predicted_class_indices]
            prediction_results = list(zip(range(1, len(df) + 1), dff.values.tolist(), predicted_labels))
            attribute_names = ['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
            'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'tcprtt',
            'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
            'response_body_len', 'ct_state_ttl', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
            'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'proto', 'service', 'state']

            return render_template('csv.html', attribute_names=attribute_names, prediction_results=prediction_results)

    return render_template('csv.html')


def predict_from_csv(df):
    print(df)
    predictions = model.predict(df)  # Assuming your model can directly predict on the DataFrame
    return predictions.tolist()

@app.route('/format')
def format():
    return render_template('format.html')


if __name__=="__main__":
    app.run(debug=True)
