// @ts-nocheck

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

const handDistanceThreshold = 0.13; // You can adjust this value based on your needs
const intervalTimeOut = 500;

const videoHeight = "360px";
const videoWidth = "480px";

const constraints = {
  video: true
};

let predictWebcamIntervalId = undefined;
let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let disableWebcamButton
let lastVideoTime = -1;

// Activate the webcam stream.
navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
  video.srcObject = stream;
});

// Function to calculate the distance between two landmarks
function calculateLandmarkDistance(landmark1, landmark2) {
  const dx = landmark2.x - landmark1.x;
  const dy = landmark2.y - landmark1.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// Function to check hand proximity to face (using only 19 for index and 9 for mouth)
async function checkHandProximity(results) {
  if (results.landmarks.length === 0) {
    stopSpeech();
    return;
  }
  const handLandmarks = results.landmarks[0]; // Assuming the first pose is the relevant one
  const indexFinger = handLandmarks[19]; // Assuming index finger is at index 19
  const leftMouthCorner = handLandmarks[9]; // Left mouth is in landmark 0 (first pose)

  const distance = calculateLandmarkDistance(indexFinger, leftMouthCorner);
  // console.log("Distance: ", distance)
  if (distance < handDistanceThreshold) {
    console.log("Hand is close to your face!", distance);
    speakNo();
  } else {
    stopSpeech();
  }
}
// Before we can use PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createPoseLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numPoses: 2
  });
  demosSection.classList.remove("invisible");
};

createPoseLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById(
  "output_canvas"
);
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("enableWebcamButton");
  disableWebcamButton = document.getElementById("disableWebcamButton");
  enableWebcamButton.addEventListener("click", startPredictions);
  disableWebcamButton.addEventListener("click", stopPredictions);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}


function startPredictions() {
  if (!poseLandmarker) {
    console.log("Wait! poseLandmaker not loaded yet.");
    return;
  }

  setWebcamInterval();
}

function stopPredictions() {
  clearWebcamInterval();
  stopSpeech();
}

async function predictWebcam() {
  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      // console.log({ result });
      checkHandProximity(result);
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      for (const landmark of result.landmarks) {
        drawingUtils.drawLandmarks(landmark, {
          radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
        });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
      }
      canvasCtx.restore();
    });
  }
}

function clearWebcamInterval() {
  clearInterval(predictWebcamIntervalId)
  predictWebcamIntervalId = undefined;
}

function setWebcamInterval() {
  if (predictWebcamIntervalId === undefined) {
    predictWebcamIntervalId = setInterval(predictWebcam, intervalTimeOut);
  }
}

function speakNo() {
  const utterance = new SpeechSynthesisUtterance("Stop that!!! You don't want to look terrible do you? Stop touching your face. This is really really bad for you. Stop doing it. Just stop it!");
  utterance.lang = 'en-US';
  utterance.pitch = 0.8; // You can adjust the pitch
  utterance.rate = 1; // You can adjust the rate
  utterance.volume = 1; // You can adjust the volume
  // Speak the utterance
  window.speechSynthesis.speak(utterance);
}

function stopSpeech() {
  if (window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
  }
}