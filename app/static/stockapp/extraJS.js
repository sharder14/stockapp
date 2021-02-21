//Hide some divs

$("#resultDiv").hide();
$("#loadingDiv").hide();
$("#loadingWheel").hide();



function loadStock(e){
    e.preventDefault();
    $('#loadingWheel').show();
    console.log("Add this!!!!");
    /*GET Name of stock to predict...
    1. USE AJAX call to send name of stock to server
    2. In python confirm if stock name is valid..
    3. If valid return stock name and loading wheel for training
    4. After training return data to JS to populate Plotly plot
    */

    var stockName= $('#stockInput').val();
    console.log('This is the stock hahaha '+ stockName);
    var modelType= $('#modelType :selected').text();
    console.log('This is the model type '+ modelType);
    
    $.ajax({
    url:window.location.href+ "getStock",
    type: "GET",
    data: {
        'stockName': $('#stockInput').val(),
        'modelType': $('#modelType').find(":selected").text()
    },
    success: function(response){
        console.log("Ayo server did it's thang");
        console.log('Original stock short name submitted was: '+response.ogStock);
        console.log('Did try block work?...'+response.testthis);
        var stockShortName=response.stockName;
        //alert(response.test);
        if(stockShortName==false){
            alert('Not a valid stock enter in a short stock name');
            $('#loadingWheel').hide();
        }
        else{
            $('#loadingDiv').show();
            $('#stockShortName').text('Stock Short Name: '+stockShortName);
            //Do get request to get the data for the plot
            $.ajax({
                url: window.location.href+"trainModel",
                type: "GET",
                data: {
                    'stockName': $('#stockInput').val(),
                    'modelType': $('#modelType :selected').text()
                },
                success: function(response){
                    $('#loadingWheel').hide();
                    $('#modelTraining').text('Model Trained');
                    //console.log(response.plotDiv)
                    $("#plot").html(response.plotDiv);
                    $("#resultDiv").show();

                }   
            })



        }
    }   
    });

}


function loadStock2(e){

    e.preventDefault();
    $('#loadingWheel').show();
    console.log("Add this!!!!");
    /*GET Name of stock to predict...
    1. USE AJAX call to send name of stock to server
    2. In python confirm if stock name is valid..
    3. If valid return stock name and stock data...
    */

    var stockName= $('#stockInput').val();
    console.log('This is the stock hahaha '+ stockName);
    
    $.ajax({
    url: window.location.href+"getStockData",
    type: "GET",
    data: {
        'stockName': $('#stockInput').val()
    },
    success: function(response){
        console.log("Ayo server did it's thang");
        //alert(response.stockName);
        var stockShortName=response.stockName;
        var stockData=response.stockData;
        var ymin=response.ymin;
        var ymax=response.ymax;
        //console.log(stockData);
        if(stockShortName==false){
            $('#loadingWheel').hide();
            alert('Not a valid stock enter in a short stock name');
        }
        else{
            //Now deal with stockData
            var tData=JSON.parse(stockData);
            //console.log(tData['yData']);
            //console.log(tData['yData'][0]);
            $('#loadingWheel').hide();
            //Now pass xData and yData to TensorflowJS model for training...
            //var xData=tData.xData;
            var xData=Object.values(tData.xData);
            //var yData=tData.yData;
            var yData=Object.values(tData.yData);
            console.log('ymin='+ymin);
            //console.log(yData);
            
            let callback = function(epoch,log) {
                console.log('Epoch: '+epoch+' ____ Loss: '+log.loss);
            };



            //Train model with window of 20, 10 epochs, .01 learing rate, 1 layers 
            var result = trainModel(xData,yData,20,2,.01,1,callback).then(function(result){
                //Now get training predictions and plot them 
                var dates=Object.values(tData.Date);
                var close=Object.values(tData.Close);
                //Plot dates and yData
                plotThis(dates,close);
                $("#resultDiv").show();
                //Now get prediction for every training point
                var trainingPreds= makePredictions(xData,result['model']);
                console.log(trainingPreds);
                //Now for each prediction we need to get it back to normal Close format...
                var trainingPreds2=[];
                for(i=0;i<trainingPreds.length;i++){
                    const tempPred=(((trainingPreds[i]+1)*(ymax-ymin))/2)+ymin;
                    trainingPreds2.push(tempPred);
                }
                //Add the training Prediction to the plot
                Plotly.addTraces('plot',{x:dates,y:trainingPreds2, name:'Training Prediction'});
            });
            console.log(result);
        }
    }   
    });


}


async function trainModel(X, Y, window_size, n_epochs, learning_rate, n_layers, callback){
    $('#loadingWheel').show();
    const input_layer_shape  = window_size;
    const input_layer_neurons = 100;
    const rnn_input_layer_features = 10;
    const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;
  
    const rnn_input_shape  = [rnn_input_layer_features, rnn_input_layer_timesteps];
    const rnn_output_neurons = 20;
  
    const rnn_batch_size = window_size;
  
    const output_layer_shape = rnn_output_neurons;
    const output_layer_neurons = 1;
  
    const model = tf.sequential();
    const xs = tf.tensor2d(X,[X.length,X[0].length]);
    console.log(xs);
    const ys = tf.tensor2d(Y,[Y.length,1]);
    console.log(ys);


    model.add(tf.layers.dense({units: input_layer_neurons, inputShape: [input_layer_shape]}));
    model.add(tf.layers.reshape({targetShape: rnn_input_shape}));
  
    let lstm_cells = [];
    for (let index = 0; index < n_layers; index++) {
         lstm_cells.push(tf.layers.lstmCell({units: rnn_output_neurons}));
    }
  
    model.add(tf.layers.rnn({
      cell: lstm_cells,
      inputShape: rnn_input_shape,
      returnSequences: false
    }));
  
    model.add(tf.layers.dense({units: output_layer_neurons, inputShape: [output_layer_shape]}));
  
    model.compile({
      optimizer: tf.train.adam(learning_rate),
      loss: 'meanSquaredError'
    });
  
    const hist = await model.fit(xs, ys,
      { batchSize: rnn_batch_size, epochs: n_epochs, callbacks: {
        onEpochEnd: async (epoch, log) => {
          callback(epoch, log);
        }
      }
    });
    console.log(model);
    $('#loadingWheel').hide();
    return { model: model, stats: hist };

}

  

function makePredictions(X, model){
    const predictedResults = model.predict(tf.tensor2d(X, [X.length, X[0].length]));
    return Array.from(predictedResults.dataSync());
}





var trace1 = {
    x: [1, 2, 3, 4],
    y: [10, 15, 13, 17],
    mode: 'lines',
    name:'Train Prediction'
};

var trace2 = {
    x: [2, 3, 4, 5],
    y: [16, 5, 11, 9],
    mode: 'lines',
    name:'Test Prediction'
};

var trace3 = {
    x: [1, 2, 3, 4],
    y: [12, 9, 15, 12],
    mode: 'lines',
    name: 'Actual Value'

};

var data = [ trace1, trace2, trace3 ];

var annotations2=[{xref:'paper', yref:'paper', x:0.0, y:1.05,
                              xanchor:'left', yanchor:'bottom',
                              text:'Results (LSTM)',
                              font:{family:'Rockwell',
                                        size:26,
                                        color:'white'},
                              showarrow:false}];


var layout = {
    xaxis:{
        showline:true,
        showgrid:true,
        showticklabels:false,
        gridcolor:'white',
        gridwidth:2,
        linecolor:'white',
        linewidth:2
    },
    yaxis:{
        title_text:'Close (USD)',
        titlefont:{
        family:'Rockwell',
        size:12,
        color:'white',
        },
        showline:true,
        showgrid:true,
        showticklabels:true,
        gridcolor:'white',
        gridwidth:2,
        linecolor:'white',
        linewidth:2,
        ticks:'outside',
        tickfont:{
        family:'Rockwell',
        size:12,
        color:'white',
        },
    },
        showlegend:true,
        annotations: annotations2,
        plot_bgcolor:'black',
        paper_bgcolor:'black',
};
var config={responsive:true}

//Plotly.newPlot('plot', data, layout, config);


function plotThis(X,Y){
    //alert('Plotting!');
    var trace1 = {
        x: X,
        y: Y,
        mode: 'lines',
        name:'Actual'
    };
    
    
    
    var data = [ trace1];
    

    
 
    
    var config={responsive:true};
    
    Plotly.newPlot('plot', data);

}



function plotResults(trainX,trainY,testX,testY,actualX,actualY){
    var trace1 = {
        x: trainX,
        y: trainY,
        mode: 'lines',
        name:'Train Prediction'
    };
    
    var trace2 = {
        x: testX,
        y: testY,
        mode: 'lines',
        name:'Test Prediction'
    };
    
    var trace3 = {
        x: actualX,
        y: actualY,
        mode: 'lines',
        name: 'Actual Value'
    
    };
    
    var data = [ trace1, trace2, trace3 ];
    
    var annotations2=[{xref:'paper', yref:'paper', x:0.0, y:1.05,
                                  xanchor:'left', yanchor:'bottom',
                                  text:'Results (LSTM)',
                                  font:{family:'Rockwell',
                                            size:26,
                                            color:'white'},
                                  showarrow:false}];
    
    
    var layout = {
        xaxis:{
            showline:true,
            showgrid:true,
            showticklabels:false,
            gridcolor:'white',
            gridwidth:2,
            linecolor:'white',
            linewidth:2
        },
        yaxis:{
            title_text:'Close (USD)',
            titlefont:{
            family:'Rockwell',
            size:12,
            color:'white',
            },
            showline:true,
            showgrid:true,
            showticklabels:true,
            gridcolor:'white',
            gridwidth:2,
            linecolor:'white',
            linewidth:2,
            ticks:'outside',
            tickfont:{
            family:'Rockwell',
            size:12,
            color:'white',
            },
        },
            showlegend:true,
            annotations: annotations2,
            plot_bgcolor:'black',
            paper_bgcolor:'black',
    };
    var config={responsive:true}
    
    Plotly.newPlot('plot', data, layout, config);

}
