import fs from "fs";

export async function checkAndCreateDirectories(directoryPath: string): Promise<boolean> {
  if (!fs.existsSync(directoryPath)) {
    fs.mkdirSync(directoryPath, { recursive: true });
    console.log(`Directory structure created: ${directoryPath}`);
    return true;
  } else {
    console.log(`Directory structure already exist.`);
    return false;
  }
}
