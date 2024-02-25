import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd



app=Flask(__name__)
model=pickle.load(open('rf.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
encoderr=pickle.load(open('encoder.pkl','rb'))



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
       'ct_src_ltm', 'is_sm_ips_ports', 'proto', 'service', 'state',
       ]
    df = pd.DataFrame([data], columns=column_names)
    nums = df.drop(columns=['proto', 'service', 'state'])
    nums= nums.astype(float)
    scaler_columns=['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
        'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'tcprtt',
        'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
        'response_body_len', 'ct_state_ttl', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
        'ct_src_ltm', 'is_sm_ips_ports']
    nums= nums[scaler_columns].copy()
    nums.columns = scaler_columns
    new_cat=pd.DataFrame(encoderr.transform([[df['proto'][0], df['service'][0], df['state'][0]]]),columns=['proto', 'service', 'state'])
    new_cat=new_cat.astype(float)
    new_nums=scaler.transform(nums)
    new_columns = ['dur', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'tcprtt',
       'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'response_body_len', 'ct_state_ttl', 'ct_src_dport_ltm',
       'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd',
       'ct_src_ltm', 'is_sm_ips_ports', 'proto', 'service', 'state']
    new_data=pd.concat([new_cat,pd.DataFrame(new_nums,columns=scaler_columns)],axis=1)
    new_data = new_data[new_columns].copy()
    return new_data


@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    final_input=stdd(data)
    print(final_input)
    output=model.predict(final_input)[0]
    if(output==1):
        return render_template("home.html",prediction_text="Alert!!!! It is and intrusion")
    else:
        return render_template("home.html",prediction_text="Normal Network")



if __name__=="__main__":
    app.run()