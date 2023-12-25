from flask import Flask, render_template,request,send_from_directory
import json
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.animation
# from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime,timedelta
import os
import upstox_client
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
app = Flask(__name__)

test_predict=0
app.debug = True

keys={
 'axis': 'NSE_EQ|INE238A01034',
 'bob': 'NSE_EQ|INE028A01039',
 'hdfc': 'NSE_EQ|INE040A01034',
 'icici': 'NSE_EQ|INE090A01021',
 'indusind': 'NSE_EQ|INE095A01012',
 'kotak': 'NSE_EQ|INE237A01028',
 'pnb': 'NSE_EQ|INE160A01022',
 'sbi': 'NSE_EQ|INE062A01020'}

def get_hist_data(instrument_key,from_date,to_date,days='day'):
    api_instance = upstox_client.HistoryApi()
    api_response = api_instance.get_historical_candle_data1(instrument_key,days,to_date,from_date, '2.0')
    return api_response.data.candles

def dataset(data,window_size=1):
    dataX, dataY = [], []
    for i in range(len(data)-window_size-1):
        a = data[i:(i+window_size), 0] 
        dataX.append(a)
        dataY.append(data[i + window_size, 0])
    return np.array(dataX), np.array(dataY)

@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')

@app.route('/bank/<bank_type>', methods=['GET'])
def bank(bank_type):
    print(bank_type)
    global test_predict
    df=pd.read_csv(f"{app.root_path}/{bank_type}.csv")
    df=df.loc[::-1]
    df['Date']=df['Date'].apply(lambda x :datetime.strptime(x, '%Y-%m-%d').date())
    # df.index=df.pop("Date")

    df1=scaler.fit_transform(np.array(df['close']).reshape(-1,1))
    train_size=int(len(df1)*0.7)
    valid_point=int(len(df1)*0.8)
    test_size=len(df1)-valid_point
    train_data,valid_data,test_data=df1[:train_size],df1[train_size:valid_point],df1[valid_point:]
    window_size = 30
    x_train, y_train = dataset(train_data, window_size)
    x_valid,y_valid=dataset(valid_data, window_size)
    x_test, y_test = dataset(test_data,window_size)

    x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
    x_valid =x_valid.reshape(x_valid.shape[0],x_valid.shape[1] , 1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)
    # Load the model
    file=open(f"{app.root_path}/model/hdfc_model",'rb')  
    model=pickle.load(file)
    file.close()
    #prediction
    train_predict=model.predict(x_train)
    valid_predict=model.predict(x_valid)
    test_predict=model.predict(x_test)
    #Scaling the values back.
    train_predict=scaler.inverse_transform(train_predict)
    valid_predict=scaler.inverse_transform(valid_predict)
    test_predict=scaler.inverse_transform(test_predict)

    predict=np.zeros(train_size-len(train_predict))
    predict=np.append(predict,train_predict)
    predict=np.append(predict,np.zeros(valid_point-train_size-len(valid_predict)))
    predict=np.append(predict,valid_predict)
    predict=np.append(predict,np.zeros(len(df)-valid_point-len(test_predict)))
    predict=np.append(predict,test_predict)

    df['predicted']=predict
    df=df.reset_index(drop=True)
    df.to_csv(f"{app.root_path}/temp/{bank_type}.csv")

    trace1 = go.Scatter(x=df["Date"], y=df["close"], mode='lines', name='Actual')
    trace2 = go.Scatter(x=df[len(df)-len(test_predict):]["Date"], y=df[len(df)-len(test_predict):]["predicted"], mode='lines', name='Predicted')
    layout = go.Layout(title='Stock Price Prediction', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    # fig = px.line(df, x="Date", y=["predicted"], title="Stock Price Prediction")
    fig.update_layout(width=1000, height=600)
    try:
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(str(e))
        # Handle cases where FuncAnimation is not available

    return render_template("index.html",bank=bank_type.upper(),graphJSON=graphJSON,type='none')
    

@app.route('/form/<bank_type>', methods=['POST'])
def form(bank_type):
    print(bank_type)
    date = request.form['date']
    print(date)
    df=pd.read_csv(f"{app.root_path}/temp/{bank_type}.csv")

    trace1 = go.Scatter(x=df["Date"], y=df["close"], mode='lines', name='Actual')
    trace2 = go.Scatter(x=df[len(df)-len(test_predict):]["Date"], y=df[len(df)-len(test_predict):]["predicted"], mode='lines', name='Predicted')
    layout = go.Layout(title='Stock Price Prediction', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    # fig = px.line(df, x="Date", y=["predicted"], title="Stock Price Prediction")
    fig.update_layout(width=1000, height=600)
    try:
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(str(e))
    date=datetime.strptime(date, '%Y-%m-%d').date()
    if(date<=datetime.strptime("2020-11-20", '%Y-%m-%d').date()):
        temp=df[df['Date']==str(date)]
        if len(temp)==1:
            index=temp.index[0]
        else:
            index=0
            print("enter valid Date")
        sending=[]
        for i,row in df[index-5:index+5].iterrows():
            schema={
                "Date":row['Date'],
                "actual":row['close'],
                "predicted":row['predicted']
            }
            sending.append(schema)
    else:
        print("Getting data from upstox")
        data=get_hist_data(keys[bank_type.lower()],str(date-timedelta(days=60)),str(date))
        data.reverse()
        data=pd.DataFrame(data,columns=['Date', 'open', 'high','low','close','volume','OI'])
        temp=scaler.fit_transform(np.array(data[len(data)-30:]['close']).reshape(-1,1))
        temp=temp.reshape(temp.shape[1],30,1)
        
        file=open(f"{app.root_path}/model/hdfc_model",'rb')  
        model=pickle.load(file)
        file.close()
    #prediction
        pre=model.predict(temp)
        pre=scaler.inverse_transform(pre)
        row=data[len(data)-1:]
        sending=[{
                "Date":str(row['Date'].values[0]),
                "actual":str(row['close'].values[0]),
                "predicted":pre[0][0]
        }]

    return render_template('index.html',type='block',data=sending,date=date,bank=bank_type.upper(),graphJSON=graphJSON)

# # Fetches the favicon:
# @app.route('/favicon.ico', methods=['GET'])
# def favicion():
#     return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run()
