import {
  HandLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

let handLandmarker: HandLandmarker | null = null;

export async function loadHandLandmarker() {
  if (handLandmarker) return handLandmarker;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/models/hand_landmarker.task", // <-- LOCAL MODEL
    },
    runningMode: "VIDEO",
    numHands: 1,
  });

  return handLandmarker;
}
