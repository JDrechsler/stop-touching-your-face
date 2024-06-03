// @ts-nocheck

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const inputVideoFile = './dataset/not_touching_face.mov';
const outputDir = './dataset/not_touching_face';
const imgHeight = 224;
const imgWidth = 224;
const frameRate = 24;

// Ensure output directories exist
const createDirectory = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`Created directory ${dir}`);
  }
};

createDirectory(outputDir);

// Extract and resize frames using ffmpeg
const extractFramesCommand = `ffmpeg -i ${inputVideoFile} -vf "fps=${frameRate},scale=${imgWidth}:${imgHeight}" ${outputDir}/frame_%04d.jpg`;
execSync(extractFramesCommand);

console.log('Frames extracted and resized');