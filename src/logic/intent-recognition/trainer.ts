import * as use from "@tensorflow-models/universal-sentence-encoder";
import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";
import LABELS from "../../dataset/intent-recognition/labels.js";
import DATA_TRAINING from "../../dataset/intent-recognition/training.js";
import { checkAndCreateDirectories } from "../../util/dir.js";

const modelSavePath = "./dist/models/intent-recognition/0.2/";
const EPOCHS = 100;

console.log("=== AI.M9J.GITHUB.IO - Intent Recognition ===");
await checkAndCreateDirectories(modelSavePath);

const intents = LABELS;

// Initialize the Universal Sentence Encoder
console.log("Loading USE...");
const embedder = await use.load();
console.log("USE loaded.");

// Prepare training data
const trainingData = DATA_TRAINING;

// Generate embeddings for training data
const trainingEmbeddings = [];
const trainingLabels = [];
console.log("Generating embeddings...");
for (const data of trainingData) {
  const embedding = await embedder.embed([data.text]);
  trainingEmbeddings.push(embedding.arraySync()[0]);
  trainingLabels.push(data.label);
  console.log(`Processed: ${data.text}`);
}
console.log("Embeddings generation completed.");

// Convert to tensors
console.log("Converting data to tensors...");
const xs = tf.tensor2d(trainingEmbeddings); // Input embeddings
const ys = tf.tensor2d(trainingLabels); // One-hot encoded labels
console.log("Tensor conversion completed.");

// Define the model
const embeddingSize = trainingEmbeddings[0].length; // Infer embedding size dynamically
const model = tf.sequential();
model.add(tf.layers.dense({ units: 128, activation: "relu", inputShape: [embeddingSize] }));
model.add(tf.layers.dense({ units: intents.length, activation: "softmax" }));
console.log("Compiling model...");
model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
console.log("Model compiled.");

// Train the model and show progress
await model.fit(xs, ys, { epochs: EPOCHS });

await model.save(`file://${modelSavePath}`);
console.log(`\nModel saved`);

const jsonData = JSON.stringify(LABELS);
try {
  fs.writeFileSync(`${modelSavePath}/labels.json`, jsonData);
  console.log("Labels has been written successfully to labels.json");
} catch (err) {
  console.error("Error writing file:", err);
}

// Predict an intent
const testPrompt = "hello";
const inputEmbedding = await embedder.embed([testPrompt]);
const prediction = model.predict(tf.tensor2d(inputEmbedding.arraySync())) as tf.Tensor;
const predictedIndex = prediction.argMax(-1).dataSync()[0];
const intentLabels = LABELS;
console.log(`\nPredicted intent: ${testPrompt} = ${intentLabels[predictedIndex]}`);
