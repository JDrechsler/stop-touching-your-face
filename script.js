// @ts-nocheck

let model;
const webcamElement = document.getElementById('webcam');
const imgHeight = 224;
const imgWidth = 224;

// Load the trained model
async function loadModel() {
  try {
    model = await tf.loadLayersModel('model/model.json');
    console.log('Model loaded');
  } catch (error) {
    console.error('Failed to load model:', error);
  }
}

// Initialize the webcam and start prediction
async function initializeWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: imgWidth, height: imgHeight }
    });
    webcamElement.srcObject = stream;
    webcamElement.addEventListener('loadeddata', detectFaceTouch);
    console.log('Webcam started');
  } catch (error) {
    console.error('Failed to start webcam:', error);
  }
}

// Detect face touch and show alert
async function detectFaceTouch() {
  if (model) {
    // Ensure frame has been loaded
    if (webcamElement.readyState === HTMLMediaElement.HAVE_ENOUGH_DATA) {
      const img = tf.browser.fromPixels(webcamElement);
      const resizedImg = tf.image.resizeBilinear(img, [imgHeight, imgWidth]);
      const normalizedImg = resizedImg.div(255.0).expandDims();
      const prediction = model.predict(normalizedImg);
      const score = prediction.dataSync()[0];

      console.log('Score:', score)
      if (score > 0.5) {
        alert('Face Touch Detected!');
      }

      // Clean up tensors
      img.dispose();
      resizedImg.dispose();
      normalizedImg.dispose();
      prediction.dispose();
    }
  }

  // Continuously call detectFaceTouch
  requestAnimationFrame(detectFaceTouch);
}

// Load the model and initialize the webcam
loadModel().then(() => {
  initializeWebcam();
});