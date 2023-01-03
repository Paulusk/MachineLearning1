const tf = require('tensorflow');

// Generate some synthetic data for training
function generateData(numExamples) {
  const xs = tf.randomUniform([numExamples, 2]);
  const ys = xs.sum(1).greaterEqual(1).cast('float32');
  return {xs, ys};
}

// Define the model
const model = tf.sequential();
model.add(tf.layers.dense({units: 4, inputShape: [2], activation: 'relu'}));
model.add(tf.layers.dense({units: 4, activation: 'relu'}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

// Compile the model
model.compile({loss: 'binaryCrossentropy', optimizer: 'adam'});

// Train the model
const trainingData = generateData(2000);
const validationData = generateData(100);

const trainLogs = [];
model.fit(trainingData.xs, trainingData.ys, {
  epochs: 10,
  validationData: [validationData.xs, validationData.ys],
  callbacks: {
    onEpochEnd: (epoch, log) => {
      trainLogs.push(log);
      console.log(`Epoch ${epoch}: loss = ${log.loss}`);
    }
  }
}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([[1, 1]])).print();
  // Output: [[1]]
});
