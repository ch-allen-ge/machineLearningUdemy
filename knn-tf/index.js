require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

function knn(features, labels, predictionPoint, k){
    const { mean, variance } = tf.moments(features, 0);
    
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

    return features
        .sub(mean) // standardization
        .div(variance.pow(0.5)) 
        .sub(scaledPrediction) // distance formula
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1) //rest of knn stuff
        .concat(labels, 1)
        .unstack()
        .sort((tensorA, tensorB) => tensorA.arraySync()[0] > tensorB.arraySync()[0] ? 1 : -1)
        .slice(0, k)
        .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k;
}

let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'sqft_living'],
    labelColumns:['price']
});

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0];
    console.log('Error ', err * 100);
});