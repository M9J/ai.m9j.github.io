import * as tf from "@tensorflow/tfjs-node";
import * as path from "path";
import * as use from "@tensorflow-models/universal-sentence-encoder";

// Path to the model (update according to where it's stored)
const savedModelsPath = `./dist/models/intent-recognition/0.1/model.json`;
const modelPath = path.join(savedModelsPath);

async function loadModel() {
  try {
    // Load the model
    const model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log("Model loaded successfully!");
    return model;
    // // Example: Use the model for prediction
    // const input = tf.tensor2d([[5.1, 3.5, 1.4, 0.2]]); // Example input
    // const prediction = model.predict(input) as tf.Tensor;
    // prediction.print(); // Print the prediction results
  } catch (err) {
    console.error("Error loading the model:", err);
  }
}

const model = await loadModel();
if (model) {
  // Initialize the Universal Sentence Encoder
  console.log("Loading USE...");
  const embedder = await use.load();
  console.log("USE loaded.");

  const inputEmbedding = await embedder.embed(["hello"]);

  const prediction = model.predict(tf.tensor2d(inputEmbedding.arraySync())) as tf.Tensor;
  prediction.print();
  //   const predictedIndex = prediction.argMax(-1).dataSync()[0];
  // Map predicted index back to intent
  //   const intentLabels = Object.keys(intents); // ['greeting', 'goodbye', 'thanks']
  //   console.log(`\nPredicted intent: ${intentLabels[predictedIndex]}`);
}
