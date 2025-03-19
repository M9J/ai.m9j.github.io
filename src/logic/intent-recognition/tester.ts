import * as use from "@tensorflow-models/universal-sentence-encoder";
import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";

// Path to the model (update according to where it's stored)
const savedModelPath = `./dist/models/intent-recognition/0.2/model.json`;
const savedlabelsPath = `./dist/models/intent-recognition/0.2/labels.json`;

async function loadModel() {
  try {
    // Load the model
    const model = await tf.loadLayersModel(`file://${savedModelPath}`);
    console.log("Model loaded successfully!");
    return model;
  } catch (err) {
    console.error("Error loading the model:", err);
  }
}

async function loadLabels() {
  try {
    // Load the model
    const labelsData = fs.readFileSync(savedlabelsPath, "utf8");
    console.log("Labels loaded successfully!");
    return JSON.parse(labelsData);
  } catch (err) {
    console.error("Error loading labels:", err);
  }
}

const model = await loadModel();
const labels = await loadLabels();
const inputs = [
  "hello",
  "clear",
  "assist",
  "help",
  "refresh",
  "which version",
  "demo",
  "change look",
];
console.log("LABELS = ", labels);
if (model && labels) {
  // Initialize the Universal Sentence Encoder
  console.log("Loading USE...");
  const embedder = await use.load();
  console.log("USE loaded.");
  for (let input of inputs) {
    const inputEmbedding = await embedder.embed([input]);
    const prediction = model.predict(tf.tensor2d(inputEmbedding.arraySync())) as tf.Tensor;
    // prediction.print();
    const predictedIndex = prediction.argMax(-1).dataSync()[0];
    console.log(`INPUT = ${input}, PREDICTION = ${labels[predictedIndex]}`);
  }
}
