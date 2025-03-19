import * as use from "@tensorflow-models/universal-sentence-encoder";
import * as tf from "@tensorflow/tfjs-node";
import DATA_VALIDATION from "../../dataset/intent-recognition/validation.js"; // Import validation dataset
import * as fs from "fs";

// Define types for validation data structure
interface ValidationData {
  text: string;
  label: number[];
}

const modelLoadPath = "./dist/models/intent-recognition/0.2"; // Path to the trained model

// Check if the model directory exists
if (!fs.existsSync(modelLoadPath)) {
  console.error(`Model directory not found: ${modelLoadPath}`);
  process.exit(1);
}

// Load the saved model
console.log("Loading trained model...");
const model = await tf.loadLayersModel(`file://${modelLoadPath}/model.json`);
console.log("Model loaded successfully.");

// Load the labels
const labelsPath = `${modelLoadPath}/labels.json`;
if (!fs.existsSync(labelsPath)) {
  console.error("Labels file not found.");
  process.exit(1);
}
const labels: string[] = JSON.parse(fs.readFileSync(labelsPath, "utf-8"));

// Initialize the Universal Sentence Encoder
console.log("Loading USE...");
const embedder = await use.load();
console.log("USE loaded.");

// Prepare validation data
const validationEmbeddings: number[][] = [];
const validationLabels: number[][] = [];
console.log("Generating embeddings for validation data...");
for (const data of DATA_VALIDATION as ValidationData[]) {
  const embedding = await embedder.embed([data.text]);
  validationEmbeddings.push(embedding.arraySync()[0]);
  validationLabels.push(data.label);
  console.log(`Processed: ${data.text}`);
}
console.log("Validation embeddings generation completed.");

// Convert validation data to tensors
console.log("Converting validation data to tensors...");
const validationXs = tf.tensor2d(validationEmbeddings); // Input embeddings
const validationYs = tf.tensor2d(validationLabels); // One-hot encoded labels
console.log("Tensor conversion completed.");

// Recompile the model
console.log("Recompiling the model...");
model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
console.log("Model compiled successfully.");

// Evaluate the model on validation data
console.log("Evaluating the model on validation data...");
const results = model.evaluate(validationXs, validationYs) as tf.Scalar[];
const validationLoss = results[0].dataSync()[0];
const validationAccuracy = results[1].dataSync()[0];
console.log(`Validation Loss: ${validationLoss.toFixed(4)}`);
console.log(`Validation Accuracy: ${validationAccuracy.toFixed(4)}`);

