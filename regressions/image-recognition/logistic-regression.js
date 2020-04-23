const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
    constructor(features, labels, options) {
        this.features =this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.costHistory = [];
        this.bHistory = [];

        this.options = Object.assign({
            learningRate: 0.1, 
            iterations: 1000,
            decisionBoundary: 0.5
        }, options);
        
        this.weights = tf.zeros([this.features.shape[1],this.labels.shape[1]]);
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).softmax();
        const differences = currentGuesses.sub(labels);

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0]);

        return this.weights.sub(slopes.mul(this.options.learningRate));
    }

    train() {
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

        for (let i=0; i<this.options.iterations; i++) {
            for (let j=0; j<batchQuantity; j++) {
                const startIndex = j * this.options.batchSize;
                const { batchSize } = this.options;

                this.weights = tf.tidy(() => {
                    const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1])
                    const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

                    return this.gradientDescent(featureSlice, labelSlice);
                });
            }
            this.bHistory.push(this.weights.arraySync()[0][0]);
            this.recordCost();
            this.updateLearningRate();
        }
    }

    predict(observations) {
        return this.processFeatures(observations).matMul(this.weights).softmax().argMax(1);
    }

    test(testFeatures, testLabels) {
       const predictions = this.predict(testFeatures);
       testLabels = tf.tensor(testLabels).argMax(1);

       const incorrect = predictions.notEqual(testLabels).sum().arraySync();

       return (predictions.shape[0] - incorrect) / predictions.shape[0];
    }

    processFeatures(features) {
        features = tf.tensor(features);
    
        if (this.mean && this.variance) {
            features = this.calculateStandardization(features, this.mean, this.variance)
        } else {
            features = this.standardize(features);
        }

        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);

        const filler = variance.cast('bool').logicalNot().cast('float32'); //hack to turn all 0's in beginning of array to 1, since we cant divide by 0

        this.mean = mean;
        this.variance = variance.add(filler);;

        return this.calculateStandardization(features, this.mean, this.variance)
    }

    calculateStandardization(features, mean, variance) {
        return features.sub(mean).div(variance.pow(0.5));
    }

    recordCost() {
        const cost = tf.tidy(() => {
            //cross entropy used here, not MSE
            //the number tells us "how wrong we are"
            const guesses = this.features.matMul(this.weights).sigmoid();

            const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());
            const termTwo = this.labels
                                    .mul(-1)
                                    .add(1)
                                    .transpose().
                                    matMul(
                                        guesses
                                            .mul(-1)
                                            .add(1)
                                            .add(1e-7) //add a constant to avoid log(0)
                                            .log()
                                    );
            
            return termOne.add(termTwo).div(this.features.shape[0]).mul(-1).arraySync()[0][0];
        });

        this.costHistory.unshift(cost);
    }

    updateLearningRate() {
        if (this.costHistory.length < 2) {
            return;
        }

        if (this.costHistory[0] > this.costHistory[1]) {
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
    }

    // gradientDescent() {
    //     const currentGuessesForMPG = this.features.map(row => {
    //         return this.m * row[0] + this.b;
    //     });

    //     const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    //         return guess - this.labels[i][0];
    //     })) * 2 / this.features.length;

    //     const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    //         return -1 * this.features[i][0] * (this.labels[i][0] - guess)
    //     })) * 2 / this.features.length;

    //     this.m = this.m - mSlope * this.options.learningRate;
    //     this.b = this.b - bSlope * this.options.learningRate;
    // }
}

module.exports = LogisticRegression;