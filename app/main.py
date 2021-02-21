from flask import Flask, render_template,request, jsonify
import pandas as pd
import numpy as np
import os
import sys
#sys.path.append('..')
import model

app = Flask(__name__)

#View for homepage
@app.route('/', methods=['GET','POST'])
def homepage(): 
    return render_template('stockapp/app.html')


#View that will confirm that the stock exists and returns the short stock name
@app.route('/getStock',methods=['GET','POST'])
def getStock():
    content={}
    if(request.method=="GET"):
        print(request.args)
        stockName=str(request.args.get('stockName'))
        content['ogStock']=stockName
        #Check to see if this stockname is legit
        shortStockName=model.confirmStock(stockName)
        if(pd.notnull(shortStockName)):
            content['stockName']=shortStockName
        else:
            content['stockName']=False
    return jsonify(content)

#View that will train the model when given a stock name
@app.route('/trainModel', methods=['GET','POST'])
def trainModel():
    #Get data from the Get request...
    # Then train model for the specific stock...
    # Then return data with json....
    content={}
    if(request.method=="GET"):
        #print(request.args)
        stockName=request.args.get('stockName')
        print(stockName)
        modelType =request.args.get('modelType')
        print(modelType)
        returnData=model.main(stockName,modelType)
        #print(returnData) 
        #content=returnData
        plotlyDiv=model.plotThis(returnData['train']['x'],returnData['train']['y'],
            returnData['test']['x'],returnData['test']['y'],
            returnData['actual']['x'],returnData['actual']['y'],
            modelType=modelType
        )       
        content['plotDiv']=plotlyDiv
    return jsonify(content)


#View that will take in stock name from a get Request and return the data in JSON for TensorflowJS to deal with
@app.route('/getStockData',methods=['GET','POST'])
def getStockData():
    content={}
    if(request.method=="GET"):
        print(request.args)
        stockName=request.args.get('stockName')
        #Check to see if this stockname is legit
        tstockData=model.getStockData(stockName)
        #print(tstockData)
        if(pd.isnull(tstockData[0])):
            tstockData[0]=False
        content['stockName']=tstockData[0]
        content['stockData']=tstockData[1] 
        content['ymin']=tstockData[2]
        content['ymax']=tstockData[3]
    return jsonify(content)







if __name__=="__main__":
    if sys.argv[1:]:
        app.run()
    else:
        app.run(host='0.0.0.0',port=80)
