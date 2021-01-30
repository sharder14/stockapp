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
    url: window.location.origin+ "/getStock",
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
                url: window.location.origin+"/trainModel",
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
