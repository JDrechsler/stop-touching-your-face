import { DrawingUtils, FaceLandmarker, FaceLandmarkerResult, FilesetResolver, HandLandmarker, HandLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";

const vision = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
);

const handDistanceThreshold = 40; // You can adjust this value based on your needs
const intervalTimeOut = 250;

const videoHeight = "360px";
const videoWidth = "480px";

let predictWebcamIntervalId: Timer | undefined = undefined;
let handLandmarker: HandLandmarker | undefined = undefined;
let faceLandmarker: FaceLandmarker | undefined = undefined;
let enableWebcamButton;
let disableWebcamButton
let lastVideoTime = -1;
let textToSpeechTextArea = document.getElementById("textToSpeechTextArea")! as HTMLTextAreaElement;
const demosSection: HTMLElement = document.getElementById("demos")!;
const videoElement: HTMLVideoElement = document.getElementById("webcam") as HTMLVideoElement;
const canvasElement = document.getElementById(
  "output_canvas"
)! as HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d")!;
const drawingUtils = new DrawingUtils(canvasCtx)

const createHandLandmarker = async () => {
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
};

async function createFaceLandmarker() {
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU"
    },
    outputFaceBlendshapes: true,
    runningMode: "VIDEO",
    numFaces: 1
  });
}

function startPredictions() {
  if (!handLandmarker || !faceLandmarker) {
    console.log("Wait! landmakers not loaded yet.");
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
  videoElement.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  videoElement.style.width = videoWidth;

  let startTimeMs = performance.now();
  if (lastVideoTime !== videoElement.currentTime) {
    lastVideoTime = videoElement.currentTime;
    const handMarkerResults = handLandmarker!.detectForVideo(videoElement, startTimeMs);
    const faceMarkerResults = faceLandmarker!.detectForVideo(videoElement, startTimeMs);

    if (!handMarkerResults.landmarks || !faceMarkerResults.faceLandmarks) { return; }
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    for (const landmarks of handMarkerResults.landmarks) {
      drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 5
      });
      drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 });
    }

    for (const landmarks of faceMarkerResults.faceLandmarks) {
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
        color: "red", lineWidth: 2
      });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
        color: "white", lineWidth: 1
      });
    }

    // Check if hand is close to face
    if (isOverlapping(handMarkerResults, faceMarkerResults)) {
      speakNo();
    } else {
      stopSpeech();
    }

    canvasCtx.restore();
  }
}

function isOverlapping(handLandmarks: HandLandmarkerResult, faceLandmarks: FaceLandmarkerResult) {
  if (handLandmarks.landmarks.length === 0 || faceLandmarks.faceLandmarks.length === 0) {
    return false;
  }
  const canvasWidth = canvasElement.width;
  const canvasHeight = canvasElement.height;

  const faceNormalizedLandmarks = faceLandmarks.faceLandmarks[0];
  const handLeftNormalizedLandmarks = handLandmarks.landmarks[0];
  const handRightNormalizedLandmarks = handLandmarks.landmarks[1];

  let handLeftLandmarksInScreenCoordinates;
  let handRightLandmarksInScreenCoordinates;


  if (!faceNormalizedLandmarks) return false;
  if (!handLeftNormalizedLandmarks && !handRightNormalizedLandmarks) return false;

  const faceLandmarksInScreenCoordinates = faceNormalizedLandmarks.map((landmark: NormalizedLandmark) => {
    return {
      x: landmark.x * canvasWidth,
      y: landmark.y * canvasHeight
    };
  });

  if (handLeftNormalizedLandmarks) {
    handLeftLandmarksInScreenCoordinates = handLeftNormalizedLandmarks.map((landmark: NormalizedLandmark) => {
      return {
        x: landmark.x * canvasWidth,
        y: landmark.y * canvasHeight
      };
    });
  }

  if (handRightNormalizedLandmarks) {
    handRightLandmarksInScreenCoordinates = handRightNormalizedLandmarks.map((landmark: NormalizedLandmark) => {
      return {
        x: landmark.x * canvasWidth,
        y: landmark.y * canvasHeight
      };
    });
  }

  for (const facePoint of faceLandmarksInScreenCoordinates) {
    if (handLeftLandmarksInScreenCoordinates) {
      for (const handPoint of handLeftLandmarksInScreenCoordinates) {
        const dx = handPoint.x - facePoint.x;
        const dy = handPoint.y - facePoint.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance <= handDistanceThreshold) {
          // console.log({ 'distance face handLeft': distance })
          return true;
        }
      }
    }

    if (handRightLandmarksInScreenCoordinates) {
      for (const handPoint of handRightLandmarksInScreenCoordinates) {
        const dx = handPoint.x - facePoint.x;
        const dy = handPoint.y - facePoint.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance <= handDistanceThreshold) {
          // console.log({ 'distance face handRight': distance })
          return true;
        }
      }
    }
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
  const textToSpeech = textToSpeechTextArea.value;
  const utterance = new SpeechSynthesisUtterance(textToSpeech);
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

// Activate the webcam stream.
navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
  videoElement.srcObject = stream;
});

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("enableWebcamButton")!;
  disableWebcamButton = document.getElementById("disableWebcamButton")!;
  enableWebcamButton.addEventListener("click", startPredictions);
  disableWebcamButton.addEventListener("click", stopPredictions);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

await createHandLandmarker();
await createFaceLandmarker();
demosSection.classList.remove("invisible");
startPredictions();