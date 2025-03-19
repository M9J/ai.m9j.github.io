import * as tf from "@tensorflow/tfjs-node";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import fs from "fs";
import v8 from "v8";
import { checkAndCreateDirectories } from "../util/dir.js";

const savePath = "./cache/toolset/use-model";
await checkAndCreateDirectories(savePath);

export async function saveUSEModel() {
  console.log("Initializing TensorFlow backend...");
  tf.setBackend("tensorflow"); // Explicitly set the TensorFlow backend
  await tf.ready(); // Ensure TensorFlow backend is ready
  console.log("TensorFlow backend initialized.");
  console.log("Loading GraphModel...");
  const embedder = await use.load();
  //   console.log("Extracting GraphModel...");
  //   const model = await embedder.loadModel();
  //   await model.save(`file://${savePath}`);
  // console.log(`USE GraphModel saved at ${savePath}`);
  const snapshot = v8.serialize(embedder);
  fs.writeFileSync(`${savePath}/embedder.dat`, snapshot);
  console.log("Embedder saved as raw content.");
}

export async function loadUSEModelLocally() {
  console.log("Loading the Universal Sentence Encoder locally...");
  //   const model = await tf.loadGraphModel(`file://${savePath}/model.json`);
  // Deserialize the embedder
  const savedSnapshot = fs.readFileSync("embedder.dat");
  const loadedEmbedder = v8.deserialize(savedSnapshot);
  console.log("Embedder loaded:", loadedEmbedder);
  const model = loadedEmbedder;
  console.log("Model loaded successfully from local storage.");
  return model;
}
