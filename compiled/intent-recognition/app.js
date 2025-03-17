import * as use from "@tensorflow-models/universal-sentence-encoder";
import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";
import DATA_INTENTS from "../../dataset/intent-recognition/intents.json" assert { type: "json" };
import DATA_TRAINING from "../../dataset/intent-recognition/trainingData.json" assert { type: "json" };
const modelSavePath = "./dist/models/intent-recognition/0.1/";
// Check if the directory exists
if (!fs.existsSync(modelSavePath)) {
    fs.mkdirSync(modelSavePath, { recursive: true }); // Create the directory
    console.log(`Model save directory created: ${modelSavePath}`);
}
else
    console.log(`Model directory already exist.`);
console.log("ai.m9j.github.io - Intent Recognition");
const intents = DATA_INTENTS;
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
model.add(tf.layers.dense({ units: Object.keys(intents).length, activation: "softmax" }));
console.log("Compiling model...");
model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });
console.log("Model compiled.");
// Train the model and show progress
await model.fit(xs, ys, {
    epochs: 100, // Total number of iterations over the dataset
    callbacks: {
        // Called at the end of every batch
        onBatchEnd: async (batch, logs) => {
            if (logs) {
                console.log(`Batch ${batch + 1}: Loss = ${logs.loss.toFixed(4)}, Accuracy = ${logs.acc ? logs.acc.toFixed(4) : "N/A"}`);
            }
        },
        // Called at the end of every epoch
        onEpochEnd: async (epoch, logs) => {
            if (logs) {
                console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}, Accuracy = ${logs.acc ? logs.acc.toFixed(4) : "N/A"}`);
            }
        },
    },
});
// Predict an intent
const inputEmbedding = await embedder.embed(["hello"]);
const prediction = model.predict(tf.tensor2d(inputEmbedding.arraySync()));
const predictedIndex = prediction.argMax(-1).dataSync()[0];
await model.save(`file://${modelSavePath}`);
console.log(`\nModel saved`);
// Map predicted index back to intent
const intentLabels = Object.keys(intents); // ['greeting', 'goodbye', 'thanks']
console.log(`\nPredicted intent: ${intentLabels[predictedIndex]}`);
