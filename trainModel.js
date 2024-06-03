const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const imageDir = 'dataset';
const batchSize = 32;
const imgHeight = 224;
const imgWidth = 224;
const epochs = 10;

// Load and preprocess images
async function loadImages(dir, label) {
  const images = [];
  const labels = [];
  const files = fs.readdirSync(dir);

  for (const file of files) {
    const filePath = path.join(dir, file);
    const imageBuffer = fs.readFileSync(filePath);
    const imageTensor = tf.node.decodeImage(imageBuffer)
      .resizeBilinear([imgHeight, imgWidth])
      .toFloat()
      .div(tf.scalar(255.0));
    images.push(imageTensor);
    labels.push(label);
  }

  return { images, labels };
}

async function loadData() {
  const touchingFaceData = await loadImages(path.join(imageDir, 'touching_face'), 1);
  const notTouchingFaceData = await loadImages(path.join(imageDir, 'not_touching_face'), 0);

  const allImages = touchingFaceData.images.concat(notTouchingFaceData.images);
  const allLabels = touchingFaceData.labels.concat(notTouchingFaceData.labels);

  const imagesTensor = tf.stack(allImages);
  const labelsTensor = tf.tensor1d(allLabels, 'int32');

  return { images: imagesTensor, labels: labelsTensor };
}

async function buildAndTrainModel() {
  const { images, labels } = await loadData();

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [imgHeight, imgWidth, 3],
    filters: 32,
    kernelSize: 3,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  await model.fit(images, labels, {
    epochs: epochs,
    batchSize: batchSize,
    validationSplit: 0.2
  });

  await model.save('file://./model');
  console.log('Model training complete and saved!');
}

buildAndTrainModel();