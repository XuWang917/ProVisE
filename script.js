const capabilityKeys = ["Perception", "Understanding", "Reasoning", "Interaction"];

const capabilityTaskKeys = {
  Perception: ["counting", "depth", "orientation", "size"],
  Understanding: ["relationship", "perspective", "mental_modeling", "grounding"],
  Reasoning: ["multihop", "prediction", "feasibility"],
  Interaction: ["affordance", "navigation", "trajectory"],
};

if ("scrollRestoration" in history) {
  history.scrollRestoration = "manual";
}

if (!window.location.hash) {
  window.scrollTo(0, 0);
}

const taskKeys = capabilityKeys.flatMap((capability) => capabilityTaskKeys[capability]);
const capabilityStartTasks = new Set(["relationship", "multihop", "affordance"]);

const taskLabels = {
  counting: "Counting",
  depth: "Depth",
  orientation: "Orientation",
  size: "Object Size",
  relationship: "Relationship",
  perspective: "Perspective",
  mental_modeling: "Mental Modeling",
  grounding: "Spatial Grounding",
  multihop: "Multi-hop",
  prediction: "Prediction",
  feasibility: "Geometric Feasibility",
  affordance: "Affordance",
  navigation: "Navigation",
  trajectory: "Trajectory",
};

const expansionTaskKeys = [
  "size",
  "grounding",
  "feasibility",
];
const coreTaskKeys = taskKeys.filter((key) => !expansionTaskKeys.includes(key));
const coreCapabilityTaskKeys = Object.fromEntries(capabilityKeys.map((capability) => [
  capability,
  capabilityTaskKeys[capability].filter((key) => !expansionTaskKeys.includes(key)),
]));

const expansionResults = window.SPATIALGEN_EXPANSION_RESULTS || {
  systems: {},
  coreSystems: {},
};

const paperModels = [
  {
    model: "Nano Banana 2",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Main evaluation",
    overall: 43.58,
    capabilities: { Perception: 73.89, Understanding: 31.43, Reasoning: 18.81, Interaction: 41.95 },
    tasks: { counting: 72.5, depth: 86.67, orientation: 62.5, relationship: 37.14, perspective: 25.71, mental_modeling: 31.43, multihop: 16.67, prediction: 20.95, affordance: 42.5, navigation: 40.0, trajectory: 43.33 },
  },
  {
    model: "LLaVA-OV-1.5-8B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Main evaluation",
    overall: 38.09,
    capabilities: { Perception: 41.11, Understanding: 33.33, Reasoning: 32.82, Interaction: 43.33 },
    tasks: { counting: 52.5, depth: 63.33, orientation: 7.5, relationship: 37.14, perspective: 31.43, mental_modeling: 31.43, multihop: 33.33, prediction: 32.3, affordance: 26.67, navigation: 30.0, trajectory: 73.33 },
  },
  {
    model: "GPT-5 Image Mini",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Main evaluation",
    overall: 36.92,
    capabilities: { Perception: 67.78, Understanding: 30.48, Reasoning: 20.33, Interaction: 23.57 },
    tasks: { counting: 52.5, depth: 73.33, orientation: 77.5, relationship: 34.29, perspective: 28.57, mental_modeling: 28.57, multihop: 10.0, prediction: 30.66, affordance: 14.05, navigation: 36.67, trajectory: 20.0 },
  },
  {
    model: "InternVL3.5-8B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Main evaluation",
    overall: 33.11,
    capabilities: { Perception: 38.33, Understanding: 20.0, Reasoning: 66.53, Interaction: 18.73 },
    tasks: { counting: 35.0, depth: 70.0, orientation: 10.0, relationship: 11.43, perspective: 20.0, mental_modeling: 28.57, multihop: 56.67, prediction: 76.4, affordance: 22.86, navigation: 26.67, trajectory: 6.67 },
  },
  {
    model: "Seedream 4.5",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Main evaluation",
    overall: 33.72,
    capabilities: { Perception: 53.33, Understanding: 28.57, Reasoning: 23.04, Interaction: 26.37 },
    tasks: { counting: 30.0, depth: 90.0, orientation: 40.0, relationship: 40.0, perspective: 20.0, mental_modeling: 25.71, multihop: 10.0, prediction: 36.08, affordance: 9.12, navigation: 36.67, trajectory: 33.33 },
  },
  {
    model: "Qwen3-VL-8B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Main evaluation",
    overall: 32.04,
    capabilities: { Perception: 52.22, Understanding: 20.95, Reasoning: 36.22, Interaction: 20.16 },
    tasks: { counting: 52.5, depth: 66.67, orientation: 37.5, relationship: 14.29, perspective: 25.71, mental_modeling: 22.86, multihop: 33.33, prediction: 39.1, affordance: 30.48, navigation: 16.67, trajectory: 13.33 },
  },
  {
    model: "GLM-4.6V-Flash",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Main evaluation",
    overall: 31.73,
    capabilities: { Perception: 43.61, Understanding: 38.1, Reasoning: 11.77, Interaction: 26.78 },
    tasks: { counting: 25.0, depth: 73.33, orientation: 32.5, relationship: 48.57, perspective: 25.71, mental_modeling: 40.0, multihop: 3.33, prediction: 20.2, affordance: 33.69, navigation: 16.67, trajectory: 30.0 },
  },
  {
    model: "JoyAI-Image",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Main evaluation",
    overall: 29.76,
    capabilities: { Perception: 38.61, Understanding: 27.62, Reasoning: 12.1, Interaction: 34.84 },
    tasks: { counting: 57.5, depth: 53.33, orientation: 5.0, relationship: 40.0, perspective: 17.14, mental_modeling: 25.71, multihop: 10.0, prediction: 14.2, affordance: 34.51, navigation: 33.33, trajectory: 36.67 },
  },
  {
    model: "FLUX.2 [klein] 4B",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Main evaluation",
    overall: 25.63,
    capabilities: { Perception: 33.06, Understanding: 28.57, Reasoning: 12.6, Interaction: 23.93 },
    tasks: { counting: 20.0, depth: 46.67, orientation: 32.5, relationship: 40.0, perspective: 14.29, mental_modeling: 31.43, multihop: 6.67, prediction: 18.53, affordance: 11.79, navigation: 36.67, trajectory: 23.33 },
  },
];

const humanReference = {
  model: "Human",
  protocol: "Reference",
  protocolKey: "human",
  source: "Reference",
  overall: 82.13,
  capabilities: { Perception: 95.0, Understanding: 80.0, Reasoning: 71.17, Interaction: 78.72 },
  tasks: {
    counting: 92.5,
    depth: 100.0,
    orientation: 92.5,
    relationship: 74.28571428571429,
    perspective: 91.42857142857143,
    mental_modeling: 74.28571428571429,
    multihop: 63.33333333333333,
    prediction: 79.0,
    affordance: 89.48025379152829,
    navigation: 83.33333333333334,
    trajectory: 63.33333333333333,
  },
};

const specialistModels = [
  {
    model: "SpaceR",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 41.83,
    capabilities: { Perception: 56.36, Understanding: 31.43, Reasoning: 49.27, Interaction: 32.19 },
    tasks: { counting: 60.0, depth: 36.67, orientation: 67.5, relationship: 51.43, perspective: 20.0, mental_modeling: 22.86, multihop: 43.33, prediction: 56.4, affordance: 18.81, navigation: 26.67, trajectory: 53.33 },
  },
  {
    model: "Cambrian-S-7B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 41.29,
    capabilities: { Perception: 62.73, Understanding: 24.76, Reasoning: 35.86, Interaction: 37.89 },
    tasks: { counting: 77.5, depth: 46.67, orientation: 60.0, relationship: 22.86, perspective: 22.86, mental_modeling: 28.57, multihop: 16.67, prediction: 58.9, affordance: 20.0, navigation: 30.0, trajectory: 66.67 },
  },
  {
    model: "GEM-2B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 35.82,
    capabilities: { Perception: 50.91, Understanding: 37.14, Reasoning: 28.59, Interaction: 21.05 },
    tasks: { counting: 62.5, depth: 40.0, orientation: 47.5, relationship: 45.71, perspective: 22.86, mental_modeling: 42.86, multihop: 33.33, prediction: 22.9, affordance: 2.86, navigation: 16.67, trajectory: 46.67 },
  },
  {
    model: "SpatialThinker-3B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 35.64,
    capabilities: { Perception: 47.27, Understanding: 30.48, Reasoning: 32.86, Interaction: 29.5 },
    tasks: { counting: 47.5, depth: 36.67, orientation: 55.0, relationship: 28.57, perspective: 20.0, mental_modeling: 42.86, multihop: 36.67, prediction: 28.3, affordance: 5.77, navigation: 43.33, trajectory: 43.33 },
  },
  {
    model: "SpaceOm",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 34.2,
    capabilities: { Perception: 49.09, Understanding: 26.67, Reasoning: 32.86, Interaction: 26.05 },
    tasks: { counting: 50.0, depth: 43.33, orientation: 52.5, relationship: 28.57, perspective: 25.71, mental_modeling: 25.71, multihop: 33.33, prediction: 32.3, affordance: 5.0, navigation: 40.0, trajectory: 36.67 },
  },
  {
    model: "VLM-3R-7B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 30.18,
    capabilities: { Perception: 41.82, Understanding: 22.86, Reasoning: 24.32, Interaction: 28.18 },
    tasks: { counting: 62.5, depth: 53.33, orientation: 12.5, relationship: 17.14, perspective: 25.71, mental_modeling: 25.71, multihop: 33.33, prediction: 13.5, affordance: 13.64, navigation: 33.33, trajectory: 40.0 },
  },
  {
    model: "SpatialRGPT-8B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 25.85,
    capabilities: { Perception: 28.18, Understanding: 30.48, Reasoning: 17.64, Interaction: 22.81 },
    tasks: { counting: 60.0, depth: 16.67, orientation: 5.0, relationship: 22.86, perspective: 37.14, mental_modeling: 31.43, multihop: 23.33, prediction: 10.8, affordance: 7.62, navigation: 16.67, trajectory: 46.67 },
  },
  {
    model: "Spatial-MLLM",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 23.94,
    capabilities: { Perception: 16.36, Understanding: 26.67, Reasoning: 24.91, Interaction: 29.12 },
    tasks: { counting: 32.5, depth: 10.0, orientation: 5.0, relationship: 20.0, perspective: 25.71, mental_modeling: 34.29, multihop: 36.67, prediction: 10.8, affordance: 16.19, navigation: 33.33, trajectory: 40.0 },
  },
  {
    model: "SpatialBot-3B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 23.25,
    capabilities: { Perception: 24.55, Understanding: 33.33, Reasoning: 23.09, Interaction: 10.7 },
    tasks: { counting: 55.0, depth: 10.0, orientation: 5.0, relationship: 28.57, perspective: 31.43, mental_modeling: 40.0, multihop: 33.33, prediction: 10.8, affordance: 9.05, navigation: 6.67, trajectory: 16.67 },
  },
];

const extraTextModels = [
  {
    model: "GPT-5.6 Sol",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary API Evaluation",
    overall: 80.98095238095239,
    capabilities: { Perception: 85.23809523809524, Understanding: 70.71428571428571, Reasoning: 85.35555555555555, Interaction: 84.61904761904763 },
    tasks: { counting: 82.5, depth: 86.66666666666667, orientation: 77.5, relationship: 40.0, perspective: 74.28571428571429, mental_modeling: 80.0, multihop: 86.66666666666667, prediction: 89.4, affordance: 80.52380952380953, navigation: 96.66666666666667, trajectory: 76.66666666666667 },
  },
  {
    model: "Claude Fable 5",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary API Evaluation",
    provisional: true,
    completedSamples: 331,
    totalSamples: 470,
    overall: 78.5687117480221,
    capabilities: { Perception: 87.77777777777779, Understanding: 63.492063492063494, Reasoning: 81.1, Interaction: 82.7487684729064 },
    tasks: { counting: 91.66666666666667, depth: 86.66666666666667, orientation: 85.0, relationship: 53.333333333333336, perspective: 74.28571428571429, mental_modeling: 62.857142857142854, multihop: 90.0, prediction: 72.2, affordance: 71.80952380952381, navigation: 93.10344827586206, trajectory: 83.33333333333333 },
  },
  {
    model: "Kimi K3",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary API Evaluation",
    overall: 63.36496598639456,
    capabilities: { Perception: 81.16071428571428, Understanding: 50.0, Reasoning: 53.4888888888889, Interaction: 67.33333333333333 },
    tasks: { counting: 95.0, depth: 80.0, orientation: 72.5, relationship: 48.57142857142857, perspective: 34.285714285714285, mental_modeling: 40.0, multihop: 26.666666666666668, prediction: 53.80000000000001, affordance: 75.33333333333333, navigation: 46.666666666666664, trajectory: 80.0 },
  },
  {
    model: "Claude Opus 4.8",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary API Evaluation",
    overall: 57.785908649173955,
    capabilities: { Perception: 81.54761904761904, Understanding: 45.71428571428571, Reasoning: 49.42380952380953, Interaction: 50.561224489795926 },
    tasks: { counting: 82.5, depth: 83.33333333333333, orientation: 77.5, relationship: 51.42857142857143, perspective: 45.714285714285715, mental_modeling: 37.142857142857146, multihop: 30.0, prediction: 49.70000000000002, affordance: 51.683673469387756, navigation: 33.333333333333336, trajectory: 66.66666666666667 },
  },
  {
    model: "BAGEL-7B-MoT",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Local Evaluation",
    overall: 48.23982683982684,
    capabilities: { Perception: 71.94444444444444, Understanding: 37.14285714285714, Reasoning: 34.78333333333333, Interaction: 44.60317460317461 },
    tasks: { counting: 77.5, depth: 73.33333333333333, orientation: 65.0, relationship: 40.0, perspective: 28.57142857142857, mental_modeling: 42.857142857142854, multihop: 26.666666666666668, prediction: 42.9, affordance: 30.476190476190485, navigation: 26.666666666666668, trajectory: 76.66666666666667 },
  },
  {
    model: "GPT-5.4",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 56.24,
    capabilities: { Perception: 73.06, Understanding: 36.19, Reasoning: 49.25, Interaction: 64.14 },
    tasks: { counting: 75.0, depth: 76.67, orientation: 67.5, relationship: 31.43, perspective: 31.43, mental_modeling: 45.71, multihop: 50.0, prediction: 48.5, affordance: 42.43, navigation: 83.33, trajectory: 66.67 },
  },
  {
    model: "Cosmos3-Nano",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 45.16,
    capabilities: { Perception: 71.94, Understanding: 28.57, Reasoning: 54.23, Interaction: 28.91 },
    tasks: { counting: 55.0, depth: 83.33, orientation: 77.5, relationship: 14.29, perspective: 37.14, mental_modeling: 34.29, multihop: 46.67, prediction: 61.8, affordance: 0.06, navigation: 23.33, trajectory: 63.33 },
  },
  {
    model: "SenseNova-Vision-7B-MoT",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 40.48,
    capabilities: { Perception: 46.36, Understanding: 31.43, Reasoning: 38.95, Interaction: 44.56 },
    tasks: { counting: 60.0, depth: 33.33, orientation: 42.5, relationship: 37.14, perspective: 34.29, mental_modeling: 22.86, multihop: 30.0, prediction: 49.7, affordance: 63.81, navigation: 26.67, trajectory: 40.0 },
  },
  {
    model: "RynnBrain-8B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 33.84,
    capabilities: { Perception: 60.0, Understanding: 31.43, Reasoning: 25.67, Interaction: 15.56 },
    tasks: { counting: 67.5, depth: 80.0, orientation: 32.5, relationship: 31.43, perspective: 22.86, mental_modeling: 40.0, multihop: 43.33, prediction: 8.0, affordance: 0.0, navigation: 26.67, trajectory: 20.0 },
  },
  {
    model: "RoboBrain2.5-8B-NV",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 28.55,
    capabilities: { Perception: 48.33, Understanding: 16.19, Reasoning: 40.22, Interaction: 13.33 },
    tasks: { counting: 55.0, depth: 80.0, orientation: 10.0, relationship: 17.14, perspective: 20.0, mental_modeling: 11.43, multihop: 33.33, prediction: 47.1, affordance: 0.0, navigation: 13.33, trajectory: 26.67 },
  },
  {
    model: "Janus-Pro-7B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 27.34,
    capabilities: { Perception: 27.5, Understanding: 31.43, Reasoning: 20.77, Interaction: 27.46 },
    tasks: { counting: 67.5, depth: 10.0, orientation: 5.0, relationship: 34.29, perspective: 17.14, mental_modeling: 42.86, multihop: 13.33, prediction: 28.2, affordance: 5.71, navigation: 16.67, trajectory: 60.0 },
  },
  {
    model: "Janus-1.3B",
    protocol: "Text Answering",
    protocolKey: "text",
    source: "Supplementary Evaluation",
    overall: 26.22,
    capabilities: { Perception: 37.22, Understanding: 36.19, Reasoning: 17.43, Interaction: 11.11 },
    tasks: { counting: 55.0, depth: 56.67, orientation: 0.0, relationship: 48.57, perspective: 22.86, mental_modeling: 37.14, multihop: 6.67, prediction: 28.2, affordance: 0.0, navigation: 16.67, trajectory: 16.67 },
  },
];

const extraVisualModels = [
  {
    model: "GPT-5 Image 2",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Supplementary Evaluation",
    overall: 48.13,
    capabilities: { Perception: 62.78, Understanding: 33.33, Reasoning: 47.04, Interaction: 49.02 },
    tasks: { counting: 72.5, depth: 63.33, orientation: 52.5, relationship: 34.29, perspective: 40.0, mental_modeling: 25.71, multihop: 40.0, prediction: 54.07, affordance: 60.4, navigation: 40.0, trajectory: 46.67 },
  },
  {
    model: "SenseNova-Vision-7B-MoT",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Supplementary Evaluation",
    overall: 33.34,
    capabilities: { Perception: 39.44, Understanding: 33.33, Reasoning: 18.67, Interaction: 37.04 },
    tasks: { counting: 22.5, depth: 83.33, orientation: 12.5, relationship: 54.29, perspective: 8.57, mental_modeling: 37.14, multihop: 33.33, prediction: 4.0, affordance: 51.13, navigation: 20.0, trajectory: 40.0 },
  },
  {
    model: "FLUX.2 [klein] 9B",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Supplementary Evaluation",
    overall: 31.54,
    capabilities: { Perception: 42.5, Understanding: 31.43, Reasoning: 19.92, Interaction: 28.43 },
    tasks: { counting: 27.5, depth: 80.0, orientation: 20.0, relationship: 40.0, perspective: 8.57, mental_modeling: 45.71, multihop: 23.33, prediction: 16.5, affordance: 28.61, navigation: 36.67, trajectory: 20.0 },
  },
  {
    model: "BAGEL-7B-MoT",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Supplementary Evaluation",
    provisional: true,
    overall: 22.763018464622075,
    capabilities: { Perception: 22.223333333333333, Understanding: 30.476190476190478, Reasoning: 16.666666666666664, Interaction: 19.65376611631269 },
    tasks: { counting: 10.0, depth: 36.67, orientation: 20.0, relationship: 40.0, perspective: 14.285714285714285, mental_modeling: 37.142857142857146, multihop: 33.33333333333333, prediction: 0.0, affordance: 5.627965015604739, navigation: 33.33333333333333, trajectory: 20.0 },
  },
  {
    model: "Qwen-Image-Edit-2511",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Supplementary Evaluation",
    overall: 22.877385467509832,
    capabilities: { Perception: 22.776666666666667, Understanding: 26.666666666666668, Reasoning: 25.959523809523812, Interaction: 17.13406417452018 },
    tasks: { counting: 10.0, depth: 43.33, orientation: 15.0, relationship: 31.428571428571427, perspective: 14.285714285714285, mental_modeling: 34.285714285714285, multihop: 23.333333333333332, prediction: 28.585714285714293, affordance: 4.735525856893873, navigation: 33.33333333333333, trajectory: 13.333333333333334 },
  },
  {
    model: "OmniGen-v1",
    protocol: "Visual Answering",
    protocolKey: "visual",
    source: "Supplementary Evaluation",
    overall: 29.507002871143392,
    capabilities: { Perception: 36.943333333333335, Understanding: 31.428571428571427, Reasoning: 25.247619047619047, Interaction: 22.988693067208317 },
    tasks: { counting: 7.5, depth: 93.33, orientation: 10.0, relationship: 45.714285714285715, perspective: 5.714285714285714, mental_modeling: 42.857142857142854, multihop: 33.33333333333333, prediction: 17.161904761904765, affordance: 5.63274586829162, navigation: 40.0, trajectory: 23.333333333333332 },
  },
];


const examples = [
  {
    "id": "counting",
    "label": "Counting",
    "short": "Count target objects",
    "capability": "Perception",
    "protocol": "Instance Marking",
    "source": "CountBench",
    "question": "How many chairs are there in the image?",
    "groundTruth": "8",
    "prediction": "8",
    "correct": true,
    "imageModel": "GPT-5-image-mini",
    "input": "assets/examples/counting_input.png",
    "generated": "assets/examples/counting_generated.png",
    "parsed": "assets/examples/counting_parsed.png"
  },
  {
    "id": "depth",
    "label": "Depth",
    "short": "Compare relative depth",
    "capability": "Perception",
    "protocol": "Relative Depth",
    "source": "BLINK",
    "question": "Which point is closer to the camera?",
    "groundTruth": "(B)",
    "prediction": "(B)",
    "correct": true,
    "imageModel": "GPT-5-image-mini",
    "input": "assets/examples/depth_input.png",
    "generated": "assets/examples/depth_generated.png",
    "parsed": "assets/examples/depth_parsed.png"
  },
  {
    "id": "orientation",
    "label": "Orientation",
    "short": "Infer facing direction",
    "capability": "Perception",
    "protocol": "Direction Grid",
    "source": "EgoOrientBench",
    "question": "From the perspective of the camera, which orientation is the cat in the photo facing? A.front B.front right C.right D.back right E.back F.back left G.left H.front left Answer with the option's letter and word from the given choices directly.",
    "groundTruth": "left",
    "prediction": "left",
    "correct": true,
    "imageModel": "GPT-5-image-mini",
    "input": "assets/examples/orientation_input.png",
    "generated": "assets/examples/orientation_generated.png",
    "parsed": "assets/examples/orientation_parsed.png"
  },
  {
    "id": "size",
    "label": "Object Size",
    "short": "Compare object size",
    "capability": "Perception",
    "protocol": "Grounded Measurement",
    "source": "SPHERE",
    "question": "Which man is taller? A. The man walking at the front. B. The man walking behind.",
    "groundTruth": "A. front",
    "prediction": "A. front",
    "correct": true,
    "imageModel": "GPT-5 Image 2",
    "input": "assets/examples/size_input.jpg?v=20260715-final-v4",
    "generated": "assets/examples/size_generated.png?v=20260715-final-v4"
  },
  {
    "id": "relationship",
    "label": "Relationship",
    "short": "Verify spatial relation",
    "capability": "Understanding",
    "protocol": "Binary Color Presence",
    "source": "VSR",
    "question": "Is the following statement true or false: The bed is under the suitcase.",
    "groundTruth": "true",
    "prediction": "true",
    "correct": true,
    "imageModel": "GPT-5-image-mini",
    "input": "assets/examples/relationship_input.png",
    "generated": "assets/examples/relationship_generated.png",
    "parsed": "assets/examples/relationship_parsed.png"
  },
  {
    "id": "perspective",
    "label": "Perspective",
    "short": "Reason from the cat's viewpoint",
    "capability": "Understanding",
    "protocol": "Direction Grid",
    "source": "ViewSpatial-Bench",
    "question": "From the perspective of this cat, where is the book located?",
    "groundTruth": "B. back",
    "prediction": "B. back",
    "correct": true,
    "imageModel": "GPT-5-image-mini",
    "input": "assets/examples/perspective_input.png",
    "generated": "assets/examples/perspective_generated.png",
    "parsed": "assets/examples/perspective_parsed.png"
  },
  {
    "id": "mental_modeling",
    "label": "Mental Modeling",
    "short": "Model unseen scene",
    "capability": "Understanding",
    "protocol": "Label Code",
    "source": "MindCube",
    "question": "Standing at the viewpoint shown in Image 3 and facing the same direction, what is behind you?",
    "groundTruth": "B",
    "prediction": "B",
    "correct": true,
    "imageModel": "GPT-5-image-mini",
    "input": "assets/examples/mental_modeling_input.png",
    "inputSet": [
      { "label": "Image 1 · Front", "src": "assets/examples/mental_modeling_front.png" },
      { "label": "Image 2 · Left", "src": "assets/examples/mental_modeling_left.png" },
      { "label": "Image 3 · Back", "src": "assets/examples/mental_modeling_back.png" },
      { "label": "Image 4 · Right", "src": "assets/examples/mental_modeling_right.png" }
    ],
    "generated": "assets/examples/mental_modeling_generated.png",
    "parsed": "assets/examples/mental_modeling_parsed.png"
  },
  {
    "id": "grounding",
    "label": "Spatial Grounding",
    "short": "Locate a referred object",
    "capability": "Understanding",
    "protocol": "Cyan Point Marker",
    "source": "RefCOCOg",
    "question": "Locate the large vase between the two smaller vases.",
    "predictionLabel": "Parsed point",
    "groundTruthLabel": "Valid region",
    "groundTruth": "p ∈ M_GT",
    "groundTruthDetail": "RefCOCOg provides a target segmentation mask M_GT rather than one unique point. Any normalized point inside M_GT is valid.",
    "prediction": "(0.484, 0.709)",
    "predictionDetail": "The cyan-dot centroid is parsed as p_pred = (0.484, 0.709) in normalized image coordinates.",
    "evaluationCaption": "p_pred (0.484, 0.709) lies inside M_GT",
    "correct": true,
    "imageModel": "GPT-5 Image 2",
    "input": "assets/examples/grounding_input.jpg?v=20260715-final-v4",
    "generated": "assets/examples/grounding_generated.png?v=20260715-final-v4"
  },
  {
    "id": "multihop",
    "label": "Multi-hop",
    "short": "Apply multi-step transform",
    "capability": "Reasoning",
    "protocol": "Label Code",
    "source": "VisWorld-Eval",
    "question": "The scene is viewed from an oblique front perspective. 'Front' refers to objects closer to the camera, while 'left' and 'right' correspond to the respective sides of the frame. Starting with the initial arrangement, perform the following operation: 1. Place an orange cylinder to the right of the blue cylinder. After the operation, what is the position of the orange cuboid relative to the object that is now closest to the right of the blue cylinder? A. left-front B. left-back C. right-front D. right",
    "groundTruth": "A",
    "prediction": "A",
    "correct": true,
    "imageModel": "GPT-5-image-mini",
    "input": "assets/examples/multihop_input.png",
    "generated": "assets/examples/multihop_generated.png",
    "parsed": "assets/examples/multihop_parsed.png"
  },
  {
    "id": "prediction",
    "label": "Prediction",
    "short": "Predict physical outcome",
    "capability": "Reasoning",
    "protocol": "Label Code",
    "source": "VisWorld-Eval",
    "question": "Given the red ball and its initial direction, estimate which numbered hole it will enter first after ideal wall reflections.",
    "groundTruth": "4",
    "prediction": "4",
    "correct": true,
    "imageModel": "GPT-5-image-mini",
    "input": "assets/examples/prediction_input.png",
    "generated": "assets/examples/prediction_generated.png",
    "outputLabel": "Reflected Trajectory",
    "parsed": "assets/examples/prediction_parsed.png"
  },
  {
    "id": "feasibility",
    "label": "Geometric Feasibility",
    "short": "Evaluate seating clearance",
    "capability": "Reasoning",
    "protocol": "Counterfactual Fit Rendering",
    "source": "SPHERE-VLM",
    "question": "Can the standing man sit between the two people on the sofa? A. No. B. Yes.",
    "groundTruth": "A. no",
    "groundTruthDetail": "The official SPHERE-VLM answer is no.",
    "prediction": "A. no",
    "predictionDetail": "The inserted seated person overlaps both occupied sofa regions, indicating collision rather than a clear fit.",
    "evaluationCaption": "A. no · the rendered placement visibly overlaps both neighbors",
    "correct": true,
    "imageModel": "GPT-5 Image 2",
    "input": "assets/examples/feasibility_sofa_input.jpg?v=20260715-final-agentic-v4",
    "generated": "assets/examples/feasibility_generated.png?v=20260715-final-agentic-v4"
  },
  {
    "id": "affordance",
    "label": "Affordance",
    "short": "Ground action region",
    "capability": "Interaction",
    "protocol": "Region Mask",
    "source": "RoboAfford-Eval",
    "question": "Spot the right sofa in the image.",
    "groundTruth": "M_GT · mask",
    "groundTruthDetail": "Reference mask M_GT: object_reference_img_82_anno_1.png (640×480).",
    "prediction": "M_pred · mask",
    "predictionDetail": "Generated mask M_pred: foreground where grayscale pixel > 127 (640×480).",
    "evaluationCaption": "Precision 0.986",
    "correct": true,
    "imageModel": "GPT-5 Image 2",
    "input": "assets/examples/affordance_input.png",
    "generated": "assets/examples/affordance_generated.png",
    "parsed": "assets/examples/affordance_parsed.png"
  },
  {
    "id": "navigation",
    "label": "Navigation",
    "short": "Choose future view",
    "capability": "Interaction",
    "protocol": "State Similarity",
    "source": "PhysBench",
    "question": "After moving in the arrow's direction, which candidate image is the next view?",
    "groundTruth": "C",
    "prediction": "C",
    "correct": true,
    "imageModel": "GPT-5-image-mini",
    "input": "assets/examples/navigation_input.png",
    "inputSetLayout": "navigation",
    "inputSet": [
      { "label": "Current", "src": "assets/examples/navigation_current.jpg" },
      { "label": "A", "src": "assets/examples/navigation_option_a.jpg" },
      { "label": "B", "src": "assets/examples/navigation_option_b.jpg" },
      { "label": "C", "src": "assets/examples/navigation_option_c.jpg" },
      { "label": "D", "src": "assets/examples/navigation_option_d.jpg" }
    ],
    "generated": "assets/examples/navigation_generated.png",
    "outputLabel": "Selected View",
    "parsed": "assets/examples/navigation_parsed.png"
  },
  {
    "id": "trajectory",
    "label": "Trajectory",
    "short": "Move toward the blue spoon",
    "capability": "Interaction",
    "protocol": "Trajectory Drawing",
    "source": "ShareRobot-Bench",
    "question": "You are a robot using the joint control. The task is \"move towards the blue spoon\".",
    "groundTruth": "(.233,.325) → (.319,.275) → (.388,.346)",
    "groundTruthDetail": "Normalized GT trajectory: [(0.232813, 0.325000), (0.318750, 0.275000), (0.387500, 0.345833)]",
    "prediction": "(.017,.517) → … → (.102,.565) → … → (.122,.531)",
    "predictionDetail": "Parsed normalized trajectory: [(0.017188, 0.516667), (0.027366, 0.518750), (0.037544, 0.516667), (0.048370, 0.516667), (0.056250, 0.517648), (0.060937, 0.529493), (0.067661, 0.540214), (0.073106, 0.551641), (0.082134, 0.557428), (0.091619, 0.561742), (0.101562, 0.564583), (0.111093, 0.568750), (0.121272, 0.570833), (0.131450, 0.572917), (0.140981, 0.577083), (0.149180, 0.573907), (0.141525, 0.563700), (0.133870, 0.553494), (0.126215, 0.543287), (0.121875, 0.531250)]",
    "evaluationCaption": "DFD 0.324 < 0.400 success threshold",
    "correct": true,
    "imageModel": "GPT-5 Image 2",
    "input": "assets/examples/trajectory_1005_frame_0.png"
  }
];

function expansionSystemKey(model) {
  return `${model.model}::${model.protocolKey}`;
}

function meanScore(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function integrateExpansionScores(model) {
  if (model.provisional) {
    return {
      ...model,
      coreOverall: null,
      coreCapabilities: Object.fromEntries(capabilityKeys.map((capability) => [capability, null])),
    };
  }

  const systemKey = expansionSystemKey(model);
  const coreEntry = expansionResults.coreSystems?.[systemKey];
  const entry = expansionResults.systems[systemKey];
  const tasks = { ...model.tasks };

  coreTaskKeys.forEach((key) => {
    const value = coreEntry?.tasks?.[key];
    if (Number.isFinite(value)) tasks[key] = value;
  });
  const coreTaskScores = coreTaskKeys.map((key) => tasks[key]).filter(Number.isFinite);
  const needsAuditedCore = model.protocolKey === "text";
  const hasCompleteCore = coreTaskScores.length === coreTaskKeys.length
    && (!needsAuditedCore || coreEntry?.status === "complete");
  const coreCapabilities = Object.fromEntries(capabilityKeys.map((capability) => {
    if (!hasCompleteCore) return [capability, null];
    const scores = coreCapabilityTaskKeys[capability].map((key) => tasks[key]);
    return [capability, meanScore(scores)];
  }));
  const coreOverall = hasCompleteCore ? meanScore(coreTaskScores) : null;

  expansionTaskKeys.forEach((key) => {
    const value = entry?.tasks?.[key];
    if (Number.isFinite(value)) tasks[key] = value;
  });

  const taskScores = taskKeys.map((key) => tasks[key]).filter(Number.isFinite);
  const hasCompleteBenchmark = taskScores.length === taskKeys.length
    && hasCompleteCore
    && entry?.status === "complete";
  const capabilities = Object.fromEntries(capabilityKeys.map((capability) => {
    if (!hasCompleteBenchmark) return [capability, null];
    const scores = capabilityTaskKeys[capability].map((key) => tasks[key]);
    return [capability, meanScore(scores)];
  }));
  const overall = hasCompleteBenchmark ? meanScore(taskScores) : null;

  return {
    ...model,
    coreOverall,
    coreCapabilities,
    tasks,
    capabilities,
    overall,
  };
}

const baseModels = [
  humanReference,
  ...paperModels,
  ...extraTextModels,
  ...extraVisualModels,
  ...specialistModels,
];
const integratedModels = baseModels.map(integrateExpansionScores);

const filters = {
  all: integratedModels,
  text: integratedModels.filter((model) => model.protocolKey === "text"),
  visual: integratedModels.filter((model) => model.protocolKey === "visual"),
};

const filterLabels = {
  all: "models + human reference",
  text: "text-answering models",
  visual: "visual-answering models",
};



let activeFilter = "all";
let activeExampleLevel = "Perception";
let activeExampleId = "counting";
let activeExampleStep = "input";
let exampleAutoplayTimers = [];
let predictionRevealTimer = null;
let examplesInView = false;
let pendingExampleAutoplay = true;
let overlaySyncFrame = null;
let examplePacingFrame = null;
let exampleResizeObserver = null;
let stepperTransitionTimer = null;
let atlasPreviewTimer = null;
let atlasPreviewToken = 0;
let atlasPreviewRotationTimer = null;
let atlasPreviewRotationIndex = 0;
let atlasPreviewInView = false;
let atlasPreviewInteraction = false;

const EXAMPLE_MOTION_RATE = 0.82;
const EXAMPLE_TIMING_SCALE = 1.28;
const ATLAS_PREVIEW_ROTATION_MS = 2800;

const pipelineSteps = [
  { id: "input", label: "Input", icon: "image" },
  { id: "visual", label: "Visual Output", icon: "spark" },
  { id: "parsing", label: "Parsing", icon: "parse" },
  { id: "evaluation", label: "Evaluation", icon: "score" },
];

const exampleLevelOrder = ["Perception", "Understanding", "Reasoning", "Interaction"];

const exampleLevelText = {
  Perception: "Direct evidence",
  Understanding: "View and relation",
  Reasoning: "Multi-step change",
  Interaction: "Embodied action",
};

const benchmarkCapabilityMeta = {
  Perception: { samples: 145, summary: "Direct visual evidence" },
  Understanding: { samples: 140, summary: "Views, relations, and scene structure" },
  Reasoning: { samples: 90, summary: "Spatial change and geometric feasibility" },
  Interaction: { samples: 95, summary: "Embodied states, regions, and paths" },
};

const benchmarkTaskMeta = {
  counting: { samples: 40, answerSpace: "Discrete count", answerForm: "Mark and count", parsedOutput: "Count", metric: "Exact match", order: 1 },
  depth: { samples: 30, answerSpace: "Pairwise choice", answerForm: "Depth map", parsedOutput: "A/B choice", metric: "Accuracy", order: 2 },
  orientation: { samples: 40, answerSpace: "Direction choice", answerForm: "Direction grid", parsedOutput: "Direction", metric: "Accuracy", order: 3 },
  size: { samples: 35, answerSpace: "Pairwise choice", answerForm: "Grounded comparison", parsedOutput: "Choice", metric: "Accuracy", order: 4 },
  grounding: { samples: 35, answerSpace: "Image point", answerForm: "Point marker", parsedOutput: "Point", metric: "Point-in-mask", order: 1 },
  relationship: { samples: 35, answerSpace: "Binary relation", answerForm: "Binary marking", parsedOutput: "Boolean", metric: "Accuracy", order: 2 },
  perspective: { samples: 35, answerSpace: "Direction choice", answerForm: "Direction grid", parsedOutput: "Choice", metric: "Accuracy", order: 3 },
  mental_modeling: { samples: 35, answerSpace: "Discrete choice", answerForm: "Label code", parsedOutput: "Choice", metric: "Accuracy", order: 4 },
  prediction: { samples: 25, answerSpace: "Outcome slot", answerForm: "Reflected path", parsedOutput: "Slot", metric: "Partial score", order: 1 },
  multihop: { samples: 30, answerSpace: "Relative position", answerForm: "Label code", parsedOutput: "Choice", metric: "Accuracy", order: 2 },
  feasibility: { samples: 35, answerSpace: "Binary feasibility", answerForm: "Fit rendering", parsedOutput: "Choice", metric: "Accuracy", order: 3 },
  affordance: { samples: 35, answerSpace: "Pixel region", answerForm: "Region mask", parsedOutput: "Region", metric: "Precision", order: 1 },
  navigation: { samples: 30, answerSpace: "Next-view choice", answerForm: "State matching", parsedOutput: "State", metric: "Accuracy", order: 2 },
  trajectory: { samples: 30, answerSpace: "Continuous path", answerForm: "Path drawing", parsedOutput: "Path", metric: "DFD", order: 3 },
};

const overlayConfigs = {
  counting: {
    caption: "Instance marking",
    type: "stars",
    useInputBase: true,
    baseSize: [917, 821],
    // Pixel centers on counting_input.png, aligned to each chair seat.
    pointsPx: [[126, 242], [378, 238], [589, 238], [805, 238], [126, 648], [378, 645], [589, 645], [805, 645]],
    stars: [
      { size: 17.2, rotate: -3, delay: 110, enterX: -18, enterY: 10 },
      { size: 16.1, rotate: 2, delay: 430, enterX: 12, enterY: -14 },
      { size: 16.0, rotate: -2, delay: 245, enterX: -10, enterY: -16 },
      { size: 16.2, rotate: 3, delay: 620, enterX: 16, enterY: 8 },
      { size: 17.0, rotate: 2, delay: 330, enterX: -16, enterY: -10 },
      { size: 16.0, rotate: -3, delay: 760, enterX: 11, enterY: 14 },
      { size: 16.1, rotate: 3, delay: 520, enterX: -12, enterY: 13 },
      { size: 16.0, rotate: -2, delay: 880, enterX: 17, enterY: -9 },
    ],
  },
  depth: {
    caption: "Depth-map transition",
    type: "depth",
    useInputBase: true,
    depthSrc: "assets/examples/depth_generated.png",
    baseSize: [536, 411],
    points: [
      { xPx: 238, yPx: 357, label: "A", mean: "176.1", loupeX: 38, loupeY: 68 },
      { xPx: 300, yPx: 358, label: "B", mean: "184.0", active: true, loupeX: 62, loupeY: 68 },
    ],
    decision: "B is lighter in the depth map",
  },
  orientation: {
    caption: "Direction grid",
    type: "directionGrid",
    selected: 3,
    labels: ["Back left", "Back", "Back right", "Left", "Center", "Right", "Front left", "Front", "Front right"],
    arrows: ["↖", "↑", "↗", "←", "•", "→", "↙", "↓", "↘"],
  },
  size: {
    caption: "Grounded measurement",
    parseCaption: "Measurement evidence parser",
    type: "agenticVisual",
    outputSrc: "assets/examples/size_generated.png?v=20260715-final-v4",
    parserTitle: "Grounded height comparison",
    readoutLabel: "Taller extent",
    readout: "Front man",
    metric: "Choice exact match",
    cardPosition: "bottom",
  },
  relationship: {
    caption: "Green / blue relationship masks",
    parseCaption: "HSV color-presence decoder",
    type: "binaryColor",
    outputSrc: "assets/examples/relationship_generated.png",
    threshold: "0.5%",
  },
  perspective: {
    caption: "Blue direction code",
    parseCaption: "3x3 reference-frame decoder",
    type: "directionGrid",
    selected: 1,
    labels: ["Back left", "Back", "Back right", "Left", "Center", "Right", "Front left", "Front", "Front right"],
    arrows: ["↖", "↑", "↗", "←", "•", "→", "↙", "↓", "↘"],
    referenceNote: "The cat faces the camera; the books lie behind it.",
  },
  mental_modeling: {
    caption: "Magenta corner code",
    parseCaption: "Corner-zone decoder",
    type: "labelCode",
    layout: "corners4",
    labels: ["A", "B", "C", "D"],
    options: ["TV", "Leather loveseat (3 seats)", "Leather loveseat", "Two single sofas"],
    selected: 1,
  },
  grounding: {
    caption: "Cyan point marker",
    parseCaption: "Point centroid parser",
    type: "agenticVisual",
    outputSrc: "assets/examples/grounding_generated.png?v=20260715-final-v4",
    parserTitle: "Cyan marker centroid",
    readoutLabel: "Normalized point",
    readout: "(0.484, 0.709)",
    metric: "Accept if p_pred ∈ M_GT",
    cardPosition: "top",
  },
  multihop: {
    caption: "Magenta corner code",
    parseCaption: "Corner-zone decoder",
    type: "labelCode",
    layout: "corners4",
    labels: ["A", "B", "C", "D"],
    options: ["left-front", "left-back", "right-front", "right"],
    selected: 0,
  },
  prediction: {
    caption: "Reflected trajectory",
    parseCaption: "First-hit hole decoder",
    type: "physicsPrediction",
    path: "M65.35 78.9 L91.37 53.55 L49.84 8.99",
    bounce: [91.37, 53.55],
    holes: 7,
    selected: 3,
    timing: { parsing: 2850, evaluation: 4050 },
  },
  feasibility: {
    caption: "Counterfactual fit rendering",
    parseCaption: "Clearance evidence parser",
    type: "agenticVisual",
    outputSrc: "assets/examples/feasibility_generated.png?v=20260715-final-agentic-v4",
    parserTitle: "Visible collision evidence",
    readoutLabel: "Attempted fit",
    readout: "Overlap on both sides",
    metric: "Parsed choice A · no",
    cardPosition: "bottom",
  },
  affordance: {
    caption: "Binary target mask",
    parseCaption: "Mask precision parser",
    type: "regionMask",
    polygon: [[70.8,17.2],[66.9,46],[67.1,49.6],[66.1,57.3],[67.5,68.4],[68.8,82.8],[70.8,97.2],[71.1,99.7],[100,99.7],[100,17],[75.4,16.5]],
    pixelThreshold: 127,
    successPrecision: 0.5,
    precision: 0.985565,
    recall: 0.998467,
    iou: 0.984077,
  },
  navigation: {
    caption: "Generated next view",
    parseCaption: "CLIP candidate matching",
    type: "stateSimilarity",
    outputSrc: "assets/examples/navigation_generated.png",
    similarities: [0.625016, 0.473999, 0.770308, 0.7363],
    selected: 2,
  },
  trajectory: {
    caption: "Continuous red trajectory",
    parseCaption: "Red-mask skeleton and 20-point path",
    type: "trajectoryProtocol",
    path: "M23.28 32.5 C26.3 33.2 29.4 30 31.88 27.5 C34.2 28 36.3 32.1 38.75 34.58",
    samplePoints: [[23.28,32.5],[24.2,32.57],[25.12,32.4],[26.03,31.98],[26.93,31.35],[27.81,30.58],[28.67,29.73],[29.5,28.9],[30.31,28.16],[31.1,27.66],[31.88,27.5],[32.64,27.75],[33.38,28.34],[34.12,29.19],[34.88,30.2],[35.64,31.29],[36.41,32.36],[37.18,33.31],[37.96,34.08],[38.75,34.58]],
    dfd: "0.324",
    threshold: "0.400",
  },
};

function fmt(value) {
  return Number(value).toFixed(2);
}

function sortedModels() {
  return [...filters[activeFilter]].sort((a, b) => {
    if (a.protocolKey === "human" && b.protocolKey !== "human") return -1;
    if (b.protocolKey === "human" && a.protocolKey !== "human") return 1;
    const aScore = Number.isFinite(a.overall) ? a.overall : -Infinity;
    const bScore = Number.isFinite(b.overall) ? b.overall : -Infinity;
    return bScore - aScore;
  });
}

function scoreCell(value, { emphasis = false, best = false } = {}) {
  if (!Number.isFinite(value)) {
    return `<span class="score-cell is-unavailable"><span>N/A</span></span>`;
  }
  const width = Math.max(0, Math.min(100, value));
  return `<span class="score-cell${emphasis ? " is-primary" : ""}${best ? " is-best" : ""}" style="--score-color: ${heatColor(value)}; --score-width: ${width}%"><span>${fmt(value)}</span></span>`;
}

function protocolBadge(model) {
  return `<span class="badge ${model.protocolKey}">${model.protocol}</span>`;
}

function modelRowsForSummary(data) {
  const models = data.filter((model) => model.protocolKey !== "human");
  return models.length ? models : data;
}

function modelLabel(model) {
  if (!model.provisional) return model.model;
  return `${model.model}<span class="model-testing-label" title="331 of 470 samples completed"> · Testing</span>`;
}

function rankDisplay(rank) {
  if (rank > 3) return `<span class="rank-number">${rank}</span>`;
  const place = ["first", "second", "third"][rank - 1];
  const tone = ["gold", "silver", "bronze"][rank - 1];
  return `
    <span class="rank-trophy rank-trophy-${tone}" role="img" aria-label="${place} place" title="${place} place">
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path d="M6 9a6 6 0 0 0 12 0V3a1 1 0 0 0-1-1H7a1 1 0 0 0-1 1z"/>
        <path d="M6 4H4.5a1 1 0 0 0 0 5H6M18 4h1.5a1 1 0 0 1 0 5H18M12 15v4M8 22h8M10 19h4"/>
      </svg>
    </span>
  `;
}

function renderLeaderboard() {
  const data = sortedModels();
  const body = document.querySelector("#leaderboard-body");
  if (!body) return;
  const rankedModels = modelRowsForSummary(data);
  const bestOverall = Math.max(...rankedModels.map((model) => model.overall).filter(Number.isFinite));
  const bestCapabilities = Object.fromEntries(
    capabilityKeys.map((key) => {
      const scores = rankedModels.map((model) => model.capabilities[key]).filter(Number.isFinite);
      return [key, scores.length ? Math.max(...scores) : null];
    }),
  );

  let rank = 0;
  body.innerHTML = data
    .map((model, index) => {
      const displayedRank = model.protocolKey === "human"
        ? `<span class="rank-reference" aria-label="Human reference">–</span>`
        : rankDisplay(++rank);
      const capabilityCells = capabilityKeys
        .map((key) => `<td class="score-td">${scoreCell(model.capabilities[key], {
          best: model.protocolKey !== "human" && Number.isFinite(model.capabilities[key]) && model.capabilities[key] === bestCapabilities[key],
        })}</td>`)
        .join("");

      return `
        <tr class="${model.protocolKey === "human" ? "is-human-row" : ""}${model.provisional ? " is-provisional-row" : ""}" style="--row-delay: ${Math.min(index, 8) * 18}ms">
          <td class="rank">${displayedRank}</td>
          <th class="model-column" scope="row">
            <div class="model-name">
              <strong>${modelLabel(model)}</strong>
            </div>
          </th>
          <td class="protocol-column">${protocolBadge(model)}</td>
          <td class="score-td overall-td">${scoreCell(model.overall, {
            emphasis: true,
            best: model.protocolKey !== "human" && model.overall === bestOverall,
          })}</td>
          ${capabilityCells}
        </tr>
      `;
    })
    .join("");

  const count = document.querySelector("#leaderboard-model-count");
  if (count) {
    const modelCount = data.filter((model) => model.protocolKey !== "human").length;
    count.innerHTML = `<strong>${modelCount}</strong><span>${filterLabels[activeFilter]}</span>`;
  }

  renderLeaderboardSummary(data);
}

function renderLeaderboardSummary(data) {
  const summary = document.querySelector("#leaderboard-summary");
  if (!summary || !data.length) return;

  const modelData = modelRowsForSummary(data);
  const bestOverall = modelData[0];
  const capabilityWinners = capabilityKeys.map((key) => ({
    key,
    model: modelData.reduce((best, candidate) => (
      candidate.capabilities[key] > best.capabilities[key] ? candidate : best
    ), modelData[0]),
  }));

  summary.innerHTML = `
    <article class="summary-overall">
      <span>Top model · overall</span>
      <strong data-count="${bestOverall.overall}" data-decimals="2">${fmt(bestOverall.overall)}</strong>
      <p>${bestOverall.model} · ${bestOverall.protocol}</p>
    </article>
    <article class="summary-capabilities">
      <span>Best by capability</span>
      <div class="capability-best-grid">
        ${capabilityWinners.map(({ key, model }) => `
          <div>
            <em>${key}</em>
            <strong>${fmt(model.capabilities[key])}</strong>
            <p title="${model.model}">${model.model}</p>
          </div>
        `).join("")}
      </div>
    </article>
  `;
  animateNumbers(summary);
}

function heatColor(value) {
  const clamped = Math.max(0, Math.min(80, value));
  const ratio = clamped / 80;
  const hue = 8 + ratio * 160;
  const sat = 48 + ratio * 10;
  const light = 92 - ratio * 28;
  return `hsl(${hue} ${sat}% ${light}%)`;
}

function renderTaskMatrix() {
  const head = document.querySelector("#task-head");
  const body = document.querySelector("#task-body");
  if (!head || !body) return;
  const data = sortedModels();
  const rankedModels = modelRowsForSummary(data);
  const bestTasks = Object.fromEntries(
    taskKeys.map((key) => {
      const scores = rankedModels.map((model) => model.tasks[key]).filter(Number.isFinite);
      return [key, scores.length ? Math.max(...scores) : null];
    }),
  );

  head.innerHTML = `
    <tr>
      <th scope="col">Model</th>
      ${taskKeys.map((key) => `<th class="${capabilityStartTasks.has(key) ? "is-capability-start" : ""}" scope="col">${taskLabels[key]}</th>`).join("")}
    </tr>
  `;

  body.innerHTML = data
    .map((model, index) => {
      const cells = taskKeys
        .map((key) => {
          const score = model.tasks[key];
          const boundary = capabilityStartTasks.has(key) ? " is-capability-start" : "";
          if (!Number.isFinite(score)) {
            return `<td class="heat-cell is-unavailable${boundary}">N/A</td>`;
          }
          const isBest = model.protocolKey !== "human" && score === bestTasks[key];
          return `<td class="heat-cell${isBest ? " is-best" : ""}${boundary}" style="--heat-color: ${heatColor(score)}">${fmt(score)}</td>`;
        })
        .join("");

      return `
        <tr class="${model.protocolKey === "human" ? "is-human-row" : ""}${model.provisional ? " is-provisional-row" : ""}" style="--row-delay: ${Math.min(index, 8) * 18}ms">
          <th class="task-model" scope="row">${modelLabel(model)}</th>
          ${cells}
        </tr>
      `;
    })
    .join("");
}

function setupTaskStickyHeader() {
  const panel = document.querySelector(".task-panel");
  const wrap = panel?.querySelector(":scope > .table-wrap");
  const table = wrap?.querySelector(".task-table");
  const head = table?.querySelector("thead");
  const colgroup = table?.querySelector("colgroup");
  if (!panel || !wrap || !table || !head || !colgroup || panel.querySelector(".task-sticky-header")) return;

  const sticky = document.createElement("div");
  const clip = document.createElement("div");
  const stickyTable = table.cloneNode(false);
  const stickyHead = head.cloneNode(true);

  sticky.className = "task-sticky-header";
  sticky.setAttribute("aria-hidden", "true");
  clip.className = "task-sticky-clip";
  stickyTable.classList.add("task-sticky-table");
  stickyTable.removeAttribute("aria-label");
  stickyTable.setAttribute("role", "presentation");
  stickyHead.removeAttribute("id");
  stickyTable.append(colgroup.cloneNode(true), stickyHead);
  clip.append(stickyTable);
  sticky.append(clip);
  panel.insertBefore(sticky, wrap);
  panel.classList.add("has-sticky-task-header");

  const syncScroll = () => {
    sticky.style.setProperty("--task-scroll-x", `${wrap.scrollLeft}px`);
    sticky.style.setProperty("--task-scroll-offset", `${-wrap.scrollLeft}px`);
  };
  const syncGeometry = () => {
    const height = head.getBoundingClientRect().height;
    if (height > 0) panel.style.setProperty("--task-header-height", `${height}px`);
    syncScroll();
  };

  wrap.addEventListener("scroll", syncScroll, { passive: true });
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(syncGeometry);
    observer.observe(wrap);
    observer.observe(head);
  } else {
    window.addEventListener("resize", syncGeometry);
  }
  window.requestAnimationFrame(syncGeometry);
}

function examplesForLevel(level) {
  return examples
    .filter((example) => example.capability === level)
    .sort((a, b) => (benchmarkTaskMeta[a.id]?.order || 99) - (benchmarkTaskMeta[b.id]?.order || 99));
}

function atlasPreviewSequence() {
  return exampleLevelOrder.flatMap((level) => examplesForLevel(level));
}

function clearAtlasPreviewRotation() {
  window.clearTimeout(atlasPreviewRotationTimer);
  atlasPreviewRotationTimer = null;
}

function scheduleAtlasPreviewRotation(delay = ATLAS_PREVIEW_ROTATION_MS) {
  clearAtlasPreviewRotation();
  const preview = document.querySelector("#atlas-preview");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (
    !preview
    || !atlasPreviewInView
    || atlasPreviewInteraction
    || reduceMotion
    || document.hidden
    || getComputedStyle(preview).display === "none"
  ) return;

  atlasPreviewRotationTimer = window.setTimeout(() => {
    const sequence = atlasPreviewSequence();
    if (!sequence.length) return;
    atlasPreviewRotationIndex = (atlasPreviewRotationIndex + 1) % sequence.length;
    updateAtlasPreview(sequence[atlasPreviewRotationIndex].id);
    scheduleAtlasPreviewRotation();
  }, delay);
}

function setupAtlasPreviewRotation() {
  const layout = document.querySelector(".capability-atlas-layout");
  if (!layout) return;

  const sequence = atlasPreviewSequence();
  const initialIndex = sequence.findIndex((example) => example.id === activeExampleId);
  atlasPreviewRotationIndex = Math.max(0, initialIndex);

  if (!("IntersectionObserver" in window)) {
    atlasPreviewInView = true;
    scheduleAtlasPreviewRotation(1600);
  } else {
    const observer = new IntersectionObserver(([entry]) => {
      atlasPreviewInView = entry.isIntersecting;
      if (atlasPreviewInView) scheduleAtlasPreviewRotation(1600);
      else clearAtlasPreviewRotation();
    }, { threshold: 0.28 });
    observer.observe(layout);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearAtlasPreviewRotation();
    else scheduleAtlasPreviewRotation(1000);
  });
  window.addEventListener("resize", () => scheduleAtlasPreviewRotation(1000));
}

function scrollToExampleWorkbench() {
  const workbench = document.querySelector("#examples");
  if (!workbench) return;
  const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  window.requestAnimationFrame(() => workbench.scrollIntoView({ behavior, block: "start" }));
}

function renderAtlasPreviewMedia(media, example) {
  const inputSet = example.inputSet?.length > 1 ? example.inputSet : null;
  media.replaceChildren();
  media.classList.toggle("is-gallery", Boolean(inputSet));

  if (!inputSet) {
    const image = document.createElement("img");
    image.id = "atlas-preview-image";
    image.src = example.input;
    image.alt = "";
    image.decoding = "async";
    media.append(image);
    return [image];
  }

  const gallery = document.createElement("div");
  const isNavigation = example.inputSetLayout === "navigation";
  gallery.className = `atlas-preview-gallery ${isNavigation ? "is-navigation" : "is-quad"}`;

  inputSet.forEach((item, index) => {
    const tile = document.createElement("figure");
    tile.className = `atlas-preview-tile${isNavigation && index === 0 ? " is-current" : ""}`;

    const image = document.createElement("img");
    image.src = item.src;
    image.alt = "";
    image.decoding = "async";

    const caption = document.createElement("figcaption");
    caption.textContent = item.label.replace(/^Image \d+\s*·\s*/, "");

    tile.append(image, caption);
    gallery.append(tile);
  });

  media.append(gallery);
  return [...gallery.querySelectorAll("img")];
}

function updateAtlasPreview(exampleId, { immediate = false } = {}) {
  const preview = document.querySelector("#atlas-preview");
  const media = document.querySelector("#atlas-preview-media");
  const example = examples.find((item) => item.id === exampleId);
  if (!preview || !media || !example) return;
  const sequenceIndex = atlasPreviewSequence().findIndex((item) => item.id === example.id);
  if (sequenceIndex >= 0) atlasPreviewRotationIndex = sequenceIndex;
  if (preview.dataset.example === example.id && !immediate) return;

  const token = ++atlasPreviewToken;
  window.clearTimeout(atlasPreviewTimer);
  if (!immediate) preview.classList.add("is-changing");

  const commit = () => {
    if (token !== atlasPreviewToken) return;
    const levelIndex = Math.max(0, exampleLevelOrder.indexOf(example.capability));
    preview.dataset.example = example.id;
    preview.dataset.level = String(levelIndex + 1);
    document.querySelectorAll("#capability-atlas [data-example]").forEach((button) => {
      button.classList.toggle("is-previewing", button.dataset.example === example.id);
    });
    document.querySelector("#atlas-preview-kicker").textContent = `${example.capability} · ${example.label}`;
    document.querySelector("#atlas-preview-title").textContent = example.short;
    document.querySelector("#atlas-preview-question").textContent = example.question;
    const images = renderAtlasPreviewMedia(media, example);

    let didReveal = false;
    const reveal = () => {
      if (didReveal || token !== atlasPreviewToken) return;
      didReveal = true;
      preview.classList.remove("is-changing");
    };

    const pendingImages = images.filter((image) => !image.complete);
    if (!pendingImages.length) {
      window.requestAnimationFrame(reveal);
    } else {
      let remaining = pendingImages.length;
      const settleImage = () => {
        remaining -= 1;
        if (remaining === 0) window.requestAnimationFrame(reveal);
      };
      pendingImages.forEach((image) => {
        image.addEventListener("load", settleImage, { once: true });
        image.addEventListener("error", settleImage, { once: true });
      });
      window.setTimeout(reveal, 700);
    }
  };

  if (immediate) commit();
  else atlasPreviewTimer = window.setTimeout(commit, 90);
}

function createCapabilityAtlas() {
  const atlas = document.querySelector("#capability-atlas");
  if (!atlas) return;

  const capabilityBands = exampleLevelOrder.map((level, levelIndex) => {
    const levelExamples = examplesForLevel(level);
    const meta = benchmarkCapabilityMeta[level];
    const tasks = levelExamples.map((example) => {
      const taskMeta = benchmarkTaskMeta[example.id];
      return `
        <button
          class="atlas-task"
          id="atlas-task-${example.id}"
          type="button"
          data-example="${example.id}"
          role="tab"
          aria-controls="example-stage"
          aria-selected="false"
          tabindex="-1"
        >
          <span class="atlas-task-copy">
            <strong>${example.label}</strong>
            <em>${taskMeta.answerSpace}</em>
          </span>
          <span class="atlas-task-count">n=${taskMeta.samples}</span>
        </button>
      `;
    }).join("");

    return `
      <div class="capability-band capability-${levelIndex + 1}" data-capability="${level}" role="presentation">
        <div class="capability-band-head">
          <span class="capability-band-marker" aria-hidden="true"></span>
          <div>
            <h3>${level}</h3>
            <p>${meta.summary}</p>
            <p class="capability-band-total"><strong>${meta.samples}</strong> samples <i aria-hidden="true">·</i> <strong>${levelExamples.length}</strong> tasks</p>
          </div>
        </div>
        <div class="capability-task-grid" data-task-count="${levelExamples.length}" style="--atlas-task-count: ${levelExamples.length}">
          ${tasks}
        </div>
      </div>
    `;
  }).join("");

  atlas.innerHTML = capabilityBands;

  atlas.addEventListener("pointerover", (event) => {
    const button = event.target.closest("[data-example]");
    if (!button || button.contains(event.relatedTarget)) return;
    atlasPreviewInteraction = true;
    clearAtlasPreviewRotation();
    updateAtlasPreview(button.dataset.example);
  });

  atlas.addEventListener("pointerout", (event) => {
    const button = event.target.closest("[data-example]");
    if (!button) return;
    const nextButton = event.relatedTarget instanceof Element
      ? event.relatedTarget.closest("[data-example]")
      : null;
    if (nextButton) return;
    atlasPreviewInteraction = false;
    scheduleAtlasPreviewRotation(1200);
  });

  atlas.addEventListener("focusin", (event) => {
    const button = event.target.closest("[data-example]");
    if (!button) return;
    atlasPreviewInteraction = true;
    clearAtlasPreviewRotation();
    updateAtlasPreview(button.dataset.example);
  });

  atlas.addEventListener("focusout", () => {
    window.setTimeout(() => {
      if (atlas.contains(document.activeElement)) return;
      atlasPreviewInteraction = false;
      scheduleAtlasPreviewRotation(1200);
    }, 0);
  });

  atlas.addEventListener("click", (event) => {
    const button = event.target.closest("[data-example]");
    if (!button) return;
    setExample(button.dataset.example);
    scrollToExampleWorkbench();
  });

  atlas.addEventListener("keydown", (event) => {
    const button = event.target.closest("[data-example]");
    if (!button) return;
    const buttons = [...atlas.querySelectorAll("[data-example]")];
    const currentIndex = buttons.indexOf(button);
    let nextIndex = currentIndex;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (currentIndex + 1) % buttons.length;
    if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (currentIndex - 1 + buttons.length) % buttons.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = buttons.length - 1;
    if (nextIndex === currentIndex) return;
    event.preventDefault();
    buttons[nextIndex].focus();
    setExample(buttons[nextIndex].dataset.example);
  });
}

function delayedStyle(index) {
  return `style="animation-delay: ${120 + index * 140}ms"`;
}

function pctStyle(item) {
  return `left:${item.x}%; top:${item.y}%;${item.w ? ` width:${item.w}%;` : ""}${item.h ? ` height:${item.h}%;` : ""}`;
}

function clearExampleAutoplay() {
  exampleAutoplayTimers.forEach((timer) => window.clearTimeout(timer));
  exampleAutoplayTimers = [];
  pendingExampleAutoplay = false;
  if (predictionRevealTimer) {
    window.clearTimeout(predictionRevealTimer);
    predictionRevealTimer = null;
  }
}

function scaledExampleDelay(delay) {
  return Math.round(delay * EXAMPLE_TIMING_SCALE);
}

function paceExampleAnimations() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const stage = document.querySelector("#example-stage");
  if (!stage || typeof stage.getAnimations !== "function") return;

  const applyRate = () => {
    stage.getAnimations({ subtree: true }).forEach((animation) => {
      if (animation.playbackRate !== EXAMPLE_MOTION_RATE) {
        animation.playbackRate = EXAMPLE_MOTION_RATE;
      }
    });
  };

  applyRate();
  if (examplePacingFrame) window.cancelAnimationFrame(examplePacingFrame);
  examplePacingFrame = window.requestAnimationFrame(() => {
    examplePacingFrame = null;
    applyRate();
  });
}

function setupExampleAnimationPacing() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const stage = document.querySelector("#example-stage");
  if (!stage) return;

  const paceStartedAnimation = (event) => {
    event.target.getAnimations().forEach((animation) => {
      animation.playbackRate = EXAMPLE_MOTION_RATE;
    });
  };

  stage.addEventListener("animationstart", paceStartedAnimation, true);
  stage.addEventListener("transitionrun", paceStartedAnimation, true);

  const observer = new MutationObserver(() => paceExampleAnimations());
  observer.observe(stage, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["class"],
  });
}

function renderExampleStepper() {
  const stepper = document.querySelector("#example-stepper");
  if (!stepper) return;
  window.clearTimeout(stepperTransitionTimer);
  stepperTransitionTimer = null;
  stepper.innerHTML = pipelineSteps
    .map((step, index) => {
      const nextStep = pipelineSteps[index + 1];
      return `
        <button type="button" data-example-step="${step.id}" role="tab" aria-selected="${step.id === activeExampleStep}">
          <span class="step-icon" data-icon="${step.icon}">${stepIcon(step.icon)}</span>
          <strong>${step.label}</strong>
        </button>
        ${nextStep ? `<span class="step-transition" data-step-transition="${step.id}-${nextStep.id}" aria-hidden="true"></span>` : ""}
      `;
    })
    .join("");

  stepper.querySelectorAll("[data-example-step]").forEach((button) => {
    button.addEventListener("click", () => {
      clearExampleAutoplay();
      setExampleStep(button.dataset.exampleStep, { manual: true });
    });
  });
}

function stepIcon(icon) {
  const icons = {
    image: `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="5" width="16" height="14" rx="2"></rect><path d="m7 16 3.2-3.2 2.6 2.5 2.2-2.1L18 16"></path><circle cx="8.5" cy="9" r="1.2"></circle></svg>`,
    spark: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l1.3 5.1L18 10l-4.7 1.9L12 17l-1.3-5.1L6 10l4.7-1.9z"></path><path d="M18.5 14.5l.6 2.2 1.9.8-1.9.8-.6 2.2-.6-2.2-1.9-.8 1.9-.8z"></path></svg>`,
    parse: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 6h14M5 12h9M5 18h6"></path><path d="m16 15 2 2 4-5"></path></svg>`,
    score: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5h8v3a4 4 0 0 1-8 0z"></path><path d="M6 6H3v2a4 4 0 0 0 4 4M18 6h3v2a4 4 0 0 1-4 4M12 12v5M9 21h6M10 17h4"></path></svg>`,
  };
  return icons[icon] || icons.image;
}

function updateStepperState() {
  const activeIndex = pipelineSteps.findIndex((step) => step.id === activeExampleStep);
  document.querySelectorAll("[data-example-step]").forEach((button, index) => {
    const isActive = button.dataset.exampleStep === activeExampleStep;
    button.classList.toggle("is-active", isActive);
    button.classList.toggle("is-complete", index < activeIndex);
    button.setAttribute("aria-selected", String(isActive));
  });
}

function showStepperTransition(previousStep, nextStep) {
  const stepper = document.querySelector("#example-stepper");
  if (!stepper) return;

  window.clearTimeout(stepperTransitionTimer);
  stepperTransitionTimer = null;
  stepper.querySelectorAll(".step-transition.is-visible").forEach((connector) => {
    connector.classList.remove("is-visible");
  });

  const previousIndex = pipelineSteps.findIndex((step) => step.id === previousStep);
  const nextIndex = pipelineSteps.findIndex((step) => step.id === nextStep);
  if (
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
    || nextIndex !== previousIndex + 1
  ) return;

  const connector = stepper.querySelector(`[data-step-transition="${previousStep}-${nextStep}"]`);
  if (!connector) return;
  connector.classList.add("is-visible");
  stepperTransitionTimer = window.setTimeout(() => {
    connector.classList.remove("is-visible");
    stepperTransitionTimer = null;
  }, scaledExampleDelay(520));
}

function renderVisualBase(example) {
  const canvas = document.querySelector("#example-visual-canvas");
  if (!canvas) return;

  canvas.classList.toggle("is-gallery-canvas", Boolean(example.inputSet?.length));
  canvas.classList.toggle("is-navigation-canvas", example.inputSetLayout === "navigation");

  if (example.inputSet?.length) {
    canvas.innerHTML = example.inputSet
      .map((item) => `
        <div class="visual-input-tile">
          <div class="visual-input-media">
            <img src="${item.src}" alt="${example.label} ${item.label} view.">
          </div>
          <span class="visual-input-label">${item.label}</span>
        </div>
      `)
      .join("") + `<div class="reasoning-overlay" id="example-overlay" aria-hidden="true"></div>`;
    observeVisualGeometry(canvas);
    requestAnimationFrame(() => syncOverlayToVisualTarget());
    return;
  }

  canvas.innerHTML = `
    <img id="example-visual-image" src="${example.input}" alt="${example.label} input image.">
    <div class="reasoning-overlay" id="example-overlay" aria-hidden="true"></div>
  `;

  const image = canvas.querySelector("#example-visual-image");
  image?.addEventListener("load", () => syncOverlayToVisualTarget(), { once: true });
  observeVisualGeometry(canvas);
  requestAnimationFrame(() => syncOverlayToVisualTarget());
}

function observeVisualGeometry(canvas) {
  exampleResizeObserver?.disconnect();
  if (!("ResizeObserver" in window)) return;

  exampleResizeObserver ||= new ResizeObserver(() => queueOverlaySync());
  exampleResizeObserver.observe(canvas);
  const image = canvas.querySelector("#example-visual-image");
  if (image) exampleResizeObserver.observe(image);
}

function updateNavigationTiles(example, config, mode) {
  const canvas = document.querySelector("#example-visual-canvas");
  if (!canvas?.classList.contains("is-navigation-canvas")) return;

  const tiles = [...canvas.querySelectorAll(".visual-input-tile")];
  const currentTile = tiles[0];
  if (!currentTile) return;

  const hasParsedMatch = currentTile.querySelector(".navigation-generated-media")
    && canvas.querySelector(".visual-input-tile.is-similarity-match");
  if (mode === "evaluation" && hasParsedMatch) return;

  tiles.forEach((tile) => {
    tile.classList.remove("is-similarity-candidate", "is-similarity-comparing", "is-similarity-match");
    tile.querySelector(".navigation-similarity-score")?.remove();
    tile.querySelector(".navigation-similarity-status")?.remove();
  });

  if (mode === "input") {
    currentTile.classList.remove("is-generated-state");
    currentTile.querySelector(".navigation-generated-media")?.remove();
    return;
  }

  currentTile.classList.add("is-generated-state");
  const currentMedia = currentTile.querySelector(".visual-input-media");
  if (!currentMedia) return;

  let generated = currentMedia.querySelector(".navigation-generated-media");
  if (!generated) {
    generated = document.createElement("div");
    generated.className = "navigation-generated-media";
    generated.innerHTML = `<img src="${config.outputSrc}" alt="Generated next view."><span class="navigation-generated-label">Generated next frame</span>`;
    currentMedia.append(generated);
  }

  if (mode === "parsing" || mode === "evaluation") {
    generated.classList.add("is-static");
    const options = tiles.slice(1);
    [currentTile, ...options].forEach((tile, index) => {
      tile.classList.add("is-similarity-comparing");
      if (index > 0) tile.classList.add("is-similarity-candidate");

      const media = tile.querySelector(".visual-input-media");
      if (!media) return;

      const score = document.createElement("span");
      score.className = `navigation-similarity-score${index === 0 ? " is-query" : ""}`;
      score.style.setProperty("--similarity-delay", `${150 + index * 30}ms`);
      if (index === 0) {
        score.textContent = "Query";
        score.setAttribute("aria-label", "Generated query frame");
      } else {
        const similarity = Number(config.similarities?.[index - 1]);
        score.textContent = Number.isFinite(similarity) ? similarity.toFixed(3) : "--";
        score.setAttribute("aria-label", `CLIP similarity ${score.textContent}`);
      }
      media.append(score);
    });

    const match = options[config.selected];
    if (match) {
      match.classList.add("is-similarity-match");
      match.querySelector(".navigation-similarity-score")?.setAttribute("aria-label", `Highest CLIP similarity ${config.similarities[config.selected].toFixed(3)}`);
    }
  }
}

function syncOverlayToVisualTarget() {
  const canvas = document.querySelector("#example-visual-canvas");
  const overlay = document.querySelector("#example-overlay");
  if (!canvas || !overlay) return;

  const canvasRect = canvas.getBoundingClientRect();
  const image = canvas.querySelector("#example-visual-image");
  const target = image || canvas;
  const targetRect = target.getBoundingClientRect();

  const left = targetRect.left - canvasRect.left;
  const top = targetRect.top - canvasRect.top;

  overlay.style.left = `${left}px`;
  overlay.style.top = `${top}px`;
  overlay.style.width = `${targetRect.width}px`;
  overlay.style.height = `${targetRect.height}px`;
}

function queueOverlaySync() {
  if (overlaySyncFrame) window.cancelAnimationFrame(overlaySyncFrame);
  overlaySyncFrame = window.requestAnimationFrame(() => {
    overlaySyncFrame = null;
    syncOverlayToVisualTarget();
  });
}

function setAnswerValue(node, value, visible, animate) {
  if (!node) return;
  node.setAttribute("aria-hidden", String(!visible));

  if (!visible) {
    node.textContent = "";
    node.classList.remove("is-value-revealed");
    return;
  }

  node.textContent = String(value);
  if (animate) {
    node.classList.remove("is-value-revealed");
    void node.offsetWidth;
    node.classList.add("is-value-revealed");
    paceExampleAnimations();
  } else {
    node.classList.remove("is-value-revealed");
  }
}

function parsingAnswerDelay(example) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return 0;
  const type = overlayConfigs[example.id]?.type;
  const delays = {
    stars: 620,
    depth: 860,
    directionGrid: 700,
    binaryColor: 720,
    labelCode: 840,
    physicsPrediction: 860,
    regionMask: 820,
    stateSimilarity: 760,
    trajectoryProtocol: 920,
    agenticVisual: 760,
    geometricFit: 900,
  };
  return scaledExampleDelay(delays[type] || 680);
}

function revealPredictionAfterParsing(example) {
  const exampleId = example.id;
  predictionRevealTimer = window.setTimeout(() => {
    predictionRevealTimer = null;
    if (activeExampleId !== exampleId || activeExampleStep !== "parsing") return;
    setAnswerValue(document.querySelector("#example-prediction"), example.prediction, true, true);
    const caption = document.querySelector("#example-stage-caption");
    if (caption) caption.textContent = "Parsed prediction";
  }, parsingAnswerDelay(example));
}

function updateAnswerState(example, step, previousStep) {
  const prediction = document.querySelector("#example-prediction");
  const groundTruth = document.querySelector("#example-ground-truth");
  const verdict = document.querySelector("#example-verdict");
  const showEvaluation = step === "evaluation";

  if (predictionRevealTimer) {
    window.clearTimeout(predictionRevealTimer);
    predictionRevealTimer = null;
  }

  setAnswerValue(groundTruth, example.groundTruth, true, false);

  if (step === "parsing") {
    const preservePrediction = previousStep === "parsing" || previousStep === "evaluation";
    setAnswerValue(prediction, example.prediction, preservePrediction, false);
    if (!preservePrediction) revealPredictionAfterParsing(example);
  } else {
    setAnswerValue(prediction, example.prediction, showEvaluation, false);
  }

  if (!verdict) return;
  verdict.textContent = showEvaluation ? (example.correct ? "Correct" : "Wrong") : "";
  verdict.hidden = !showEvaluation;
  verdict.setAttribute("aria-hidden", String(!showEvaluation));
  verdict.classList.toggle("is-correct", example.correct);
  verdict.classList.toggle("is-wrong", !example.correct);
}

function setExampleStep(stepId, options = {}) {
  if (options.manual) clearExampleAutoplay();

  const example = examples.find((item) => item.id === activeExampleId) || examples[0];
  const step = pipelineSteps.some((item) => item.id === stepId) ? stepId : "input";
  const previousStep = activeExampleStep;
  activeExampleStep = step;

  const stage = document.querySelector("#example-stage");
  stage?.setAttribute("data-step", step);
  stage?.classList.toggle("is-manual-step", Boolean(options.manual));

  updateAnswerState(example, step, previousStep);
  updateStepperState();
  showStepperTransition(previousStep, step);
  renderVisualStep(example, step, previousStep);
  paceExampleAnimations();
}

function renderVisualStep(example, step, previousStep = activeExampleStep) {
  const frame = document.querySelector("#example-visual-frame");
  const overlay = document.querySelector("#example-overlay");
  const caption = document.querySelector("#example-stage-caption");
  const config = overlayConfigs[example.id];
  if (!frame || !overlay) return;

  const isInput = step === "input";
  const isVisual = step === "visual";
  const isParsing = step === "parsing";
  const isEvaluation = step === "evaluation";
  const mode = isVisual ? "visual" : isParsing ? "parsing" : isEvaluation ? "evaluation" : "input";

  if (caption) {
    const captions = {
      input: "Input image",
      visual: config?.caption || "Visual answer",
      parsing: config?.parseCaption || "Parsing visual answer",
      evaluation: example.evaluationCaption || (example.correct ? "Prediction matches ground truth" : "Prediction differs from ground truth"),
    };
    caption.textContent = captions[step];
  }

  if (config?.type === "stateSimilarity") {
    updateNavigationTiles(example, config, mode);
  }

  const preserveParsedOverlay = isEvaluation
    && (previousStep === "parsing" || previousStep === "evaluation")
    && overlay.dataset.exampleId === example.id
    && overlay.childElementCount > 0;

  if (preserveParsedOverlay) {
    overlay.classList.remove("is-parsing");
    overlay.classList.add("is-evaluation", "is-settled");
    overlay.dataset.mode = "evaluation";
    syncOverlayToVisualTarget();
    return;
  }

  const preservePredictionTrace = isParsing && overlay.querySelector(".physics-svg");

  if (preservePredictionTrace) {
    overlay.classList.remove("is-visual");
    overlay.classList.add("is-parsing", "is-settled");
    overlay.dataset.mode = "parsing";
    overlay.querySelector(".physics-svg")?.classList.add("is-static");

    if (!overlay.querySelector(".prediction-hole-grid")) {
      const holes = renderPredictionHoleItems(config);
      const grid = document.createElement("div");
      grid.className = "prediction-hole-grid";
      grid.innerHTML = holes;
      overlay.append(grid);
    }

    syncOverlayToVisualTarget();
    return;
  }

  frame.classList.remove("is-animating");
  overlay.className = `reasoning-overlay ${config ? `overlay-${config.type}` : ""}`;
  overlay.dataset.exampleId = example.id;
  overlay.dataset.mode = step;
  overlay.innerHTML = "";
  syncOverlayToVisualTarget();

  if (!config || isInput) return;

  overlay.classList.add(`is-${mode}`);
  if (isParsing || isEvaluation) overlay.classList.add("is-settled");
  syncOverlayToVisualTarget();
  overlay.innerHTML = renderOverlayContent(config, mode, example);
  requestAnimationFrame(() => frame.classList.add("is-animating"));
}

function renderOverlayContent(config, mode = "visual", example = null) {
  const showParsing = mode === "parsing" || mode === "evaluation";
  const showEvaluation = mode === "evaluation";

  if (config.type === "stars") {
    const markers = pointsToPercent(config);
    const stars = markers
      .map(({ x, y }, index) => {
        const star = config.stars?.[index] || {};
        const delay = showParsing || showEvaluation ? 0 : (star.delay || 120 + index * 130);
        const pulse = !showParsing && !showEvaluation
          ? `<span class="marker-pulse" style="left:${x}%; top:${y}%; animation-delay:${Math.max(0, delay - 90)}ms"></span>`
          : "";
        const settled = showParsing || showEvaluation ? " is-static" : "";
        return `${pulse}<span class="overlay-star${settled}" style="left:${x}%; top:${y}%; --star-size:${star.size || config.starSize || 16}%; --star-rotate:${star.rotate || 0}deg; --star-enter-x:${star.enterX || 0}px; --star-enter-y:${star.enterY || 0}px; animation-delay:${delay}ms"></span>`;
      })
      .join("");
    const parseBoxes = showParsing ? markers
      .map(({ x, y }, index) => `<span class="marker-index" style="left:${x}%; top:${y}%; animation-delay:${index * 40}ms">${index + 1}</span>`)
      .join("") : "";
    const parseBadge = showParsing ? `<span class="parse-badge">Recovered markers: ${markers.length}</span>` : "";
    return `${stars}${parseBoxes}${parseBadge}`;
  }

  if (config.type === "depth") {
    const points = pointsToPercent(config);
    const depthMap = `<img class="overlay-depth-map" src="${config.depthSrc}" alt="" aria-hidden="true">`;
    if (mode === "visual") return depthMap;

    const rings = points.map((point, index) => `
      <span class="depth-ring${point.active ? " is-active" : ""}${showParsing && point.active ? " is-parsed" : ""}" style="left:${point.x}%; top:${point.y}%; animation-delay:${showParsing || showEvaluation ? 0 : 520 + index * 120}ms"></span>
      <span class="depth-label${point.active ? " is-active" : ""}${showParsing && point.active ? " is-parsed" : ""}" style="left:${point.x + 3.6}%; top:${point.y - 7.5}%; animation-delay:${showParsing || showEvaluation ? 0 : 650 + index * 120}ms">${point.label}</span>
    `).join("");
    const loupes = points.map((point, index) => `
      <span class="depth-loupe${point.active ? " is-active" : ""}" style="left:${point.loupeX || point.x}%; top:${point.loupeY || point.y}%; --focus-x:${point.x}%; --focus-y:${point.y}%; background-image:url('${config.depthSrc}'); animation-delay:${showParsing || showEvaluation ? 0 : 900 + index * 180}ms">
        <em>${point.label}</em>
        <strong>${point.mean}</strong>
      </span>
    `).join("");
    const decision = showParsing || showEvaluation ? `<span class="depth-decision">${config.decision}</span>` : "";
    return `${depthMap}${rings}${loupes}${decision}`;
  }

  if (config.type === "binaryColor") {
    const outputImage = `<img class="protocol-output-image${showParsing ? " is-static" : ""}" src="${config.outputSrc}" alt="" aria-hidden="true">`;
    if (!showParsing) return outputImage;
    return `${outputImage}
      <div class="protocol-parser-card binary-parser-card">
        <strong>HSV color presence</strong>
        <span class="color-signal is-green"><i></i>Green mask <b>detected</b></span>
        <span class="color-signal is-blue"><i></i>Blue mask <b>detected</b></span>
        <small>True when colored area &gt; ${config.threshold}</small>
      </div>
      <span class="protocol-scan-line"></span>`;
  }

  if (config.type === "agenticVisual") {
    const outputImage = `<img class="protocol-output-image${showParsing ? " is-static" : ""}" src="${config.outputSrc}" alt="" aria-hidden="true">`;
    if (!showParsing) return outputImage;
    const positionClass = config.cardPosition === "bottom" ? " is-bottom" : "";
    return `${outputImage}
      <div class="protocol-parser-card agentic-parser-card${positionClass}">
        <strong>${config.parserTitle}</strong>
        <span><em>${config.readoutLabel}</em><b>${config.readout}</b></span>
        <small>${config.metric}</small>
      </div>
      <span class="protocol-scan-line"></span>`;
  }

  if (config.type === "geometricFit") {
    const style = [
      `--fit-gap-left:${config.gapLeft}%`,
      `--fit-gap-width:${config.gapWidth}%`,
      `--fit-gap-top:${config.gapTop}%`,
      `--fit-gap-height:${config.gapHeight}%`,
      `--fit-person-left:${config.personLeft}%`,
      `--fit-person-top:${config.personTop}%`,
      `--fit-person-width:${config.personWidth}%`,
      `--fit-person-height:${config.personHeight}%`,
      `--fit-seat-top:${config.seatTop}%`,
      `--fit-seat-height:${config.seatHeight}%`,
    ].join(";");
    const settled = showParsing ? " is-static" : "";
    const parser = showParsing ? `
      <div class="protocol-parser-card fit-parser-card">
        <strong>Clearance evidence</strong>
        <span><em>Available gap</em><b>too narrow</b></span>
        <span><em>Projected placement</em><b>overlap detected</b></span>
        <small>Parsed choice A · no</small>
      </div>
      <span class="protocol-scan-line"></span>` : "";
    return `
      <div class="fit-evidence${settled}" style="${style}">
        <span class="fit-gap-region"></span>
        <span class="fit-gap-boundary is-left"></span>
        <span class="fit-gap-boundary is-right"></span>
        <span class="fit-person-projection">
          <i class="fit-person-head"></i>
          <i class="fit-person-torso"></i>
          <i class="fit-person-seat"></i>
        </span>
        <span class="fit-overlap is-left"></span>
        <span class="fit-overlap is-right"></span>
      </div>${parser}`;
  }

  if (config.type === "labelCode") {
    const labels = config.labels || ["A", "B", "C", "D"];
    const selected = Math.max(0, Math.min(labels.length - 1, Number(config.selected) || 0));
    const isStrip = config.layout === "hstrip";
    const cornerPositions = [[7, 7], [93, 7], [7, 93], [93, 93]];
    const markerX = isStrip ? ((selected + 0.5) / labels.length) * 100 : cornerPositions[selected][0];
    const markerY = isStrip ? 91 : cornerPositions[selected][1];
    const marker = `<span class="label-code-marker${isStrip ? " is-strip" : ""}${showParsing ? " is-static" : ""}" style="left:${markerX}%; top:${markerY}%"></span>`;
    if (!showParsing) return marker;

    if (isStrip) {
      const cells = labels.map((label, index) => `<span class="code-strip-cell${index === selected ? " is-selected" : ""}"><b>${label}</b><i>${index === selected ? "magenta max" : ""}</i></span>`).join("");
      return `${marker}<div class="code-strip-parser">${cells}</div>`;
    }

    const corners = ["top-left", "top-right", "bottom-left", "bottom-right"];
    const options = config.options || labels;
    const keyItems = labels.slice(0, 4).map((label, index) => `
      <span class="code-key-option${index === selected ? " is-selected" : ""}">
        <i class="code-key-glyph is-${corners[index]}" aria-hidden="true"></i>
        <b>${label}</b>
        <em>${options[index] || label}</em>
      </span>
    `).join("");
    return `${marker}
      <div class="code-answer-key">
        <div class="code-answer-key-head"><i></i><strong>Corner code</strong><span>position → option</span></div>
        <div class="code-answer-key-grid">${keyItems}</div>
      </div>`;
  }

  if (config.type === "physicsPrediction") {
    const settled = showParsing ? " is-static" : "";
    const [bounceX, bounceY] = config.bounce;
    const trace = `
      <svg class="physics-svg${settled}" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <mask id="physics-path-reveal" x="-10" y="-10" width="120" height="120" maskUnits="userSpaceOnUse">
            <path class="physics-path-reveal" d="${config.path}"></path>
          </mask>
        </defs>
        <g mask="url(#physics-path-reveal)">
          <path class="physics-path-glow" d="${config.path}"></path>
          <path class="physics-path" d="${config.path}"></path>
        </g>
      </svg>
      <span class="physics-bounce" style="left:${bounceX}%;top:${bounceY}%"></span>`;
    if (!showParsing) return trace;

    const holes = renderPredictionHoleItems(config);
    return `${trace}
      <div class="prediction-hole-grid">${holes}</div>`;
  }

  if (config.type === "regionMask") {
    const polygon = config.polygon.map(([x, y]) => `${Math.min(100, x)},${Math.min(100, y)}`).join(" ");
    const mask = `
      <svg class="region-mask-svg${showParsing ? " is-static" : ""}" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <rect class="region-mask-background" width="100" height="100"></rect>
        <polygon class="region-mask-shape" points="${polygon}"></polygon>
      </svg>`;
    if (!showParsing) return mask;
    return `${mask}
      <span class="mask-parser-scan"></span>
      <div class="protocol-parser-card mask-parser-card">
        <strong>Binary mask parser</strong>
        <span><em>Threshold</em><b>pixel &gt; ${config.pixelThreshold}</b></span>
        <span><em>Output</em><b>640×480 binary mask</b></span>
        <span><em>Metric</em><b>Precision</b></span>
      </div>`;
  }

  if (config.type === "stateSimilarity") {
    return "";
  }

  if (config.type === "trajectoryProtocol") {
    const path = `<svg class="trajectory-svg${showParsing ? " is-static" : ""}" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path class="trajectory-line" d="${config.path}"></path></svg>`;
    if (!showParsing) return path;
    const samples = config.samplePoints.map(([x, y], index) => `<span class="trajectory-sample${index === 0 ? " is-start" : ""}" style="left:${x}%;top:${y}%;animation-delay:${index * 24}ms"></span>`).join("");
    return `${path}${samples}
      <div class="trajectory-parser-strip">
        <b>red mask</b><i>→</i><b>skeleton</b><i>→</i><strong>normalized (x,y) trajectory</strong>
      </div>`;
  }

  if (config.type === "dots") {
    const dots = config.points
      .map((point, index) => `
        <span class="overlay-dot${point.active ? " is-active" : ""}" style="left:${point.x}%; top:${point.y}%; animation-delay:${showParsing ? 0 : 140 + index * 220}ms"></span>
        <span class="overlay-chip" style="left:${point.x + 3}%; top:${point.y - 8}%; animation-delay:${showParsing ? 0 : 260 + index * 220}ms">${point.label}</span>
      `)
      .join("");
    return `${dots}${showParsing ? `<span class="parse-badge">Parsed answer: ${example?.prediction || ""}</span>` : ""}`;
  }

  if (config.type === "grid") {
    return `<div class="overlay-grid">${Array.from({ length: 9 }, (_, index) => `<span class="${index === config.selected ? "is-selected" : ""}"></span>`).join("")}</div>${showParsing ? `<span class="parse-badge">Parsed answer: ${example?.prediction || ""}</span>` : ""}`;
  }

  if (config.type === "directionGrid") {
    const cells = Array.from({ length: 9 }, (_, index) => {
      const selected = index === config.selected;
      const label = config.labels?.[index] || "";
      const arrow = config.arrows?.[index] || "";
      return `
        <span class="direction-cell${selected ? " is-selected" : ""}${(showParsing || showEvaluation) && selected ? " is-parsed" : ""}">
          ${showParsing || showEvaluation ? `<em>${arrow}</em><strong>${label}</strong>` : ""}
        </span>
      `;
    }).join("");
    const parsed = (showParsing || showEvaluation) && !config.referenceNote
      ? `<span class="parse-badge">Parsed answer: ${example?.prediction || ""}</span>`
      : "";
    const reference = (showParsing || showEvaluation) && config.referenceNote
      ? `<span class="perspective-reference"><b>Decoded: ${example?.prediction || ""}</b>${config.referenceNote}</span>`
      : "";
    return `<div class="direction-grid${showParsing || showEvaluation ? " is-parsing" : ""}">${cells}</div>${reference}${parsed}`;
  }

  if (config.type === "boxes") {
    const boxes = [
      ...(config.boxes || []).map((box, index) => `<span class="overlay-box" style="${pctStyle(box)} animation-delay:${showParsing ? 0 : 180 + index * 180}ms"></span>`),
      ...(config.chips || []).map((chip, index) => `<span class="overlay-chip" style="left:${chip.x}%; top:${chip.y}%; animation-delay:${showParsing ? 0 : 520 + index * 160}ms">${chip.text}</span>`),
    ].join("");
    return `${boxes}${showParsing ? `<span class="parse-badge">Parsed answer: ${example?.prediction || ""}</span>` : ""}`;
  }

  if (config.type === "mask") {
    const masks = (config.masks || [])
      .map((mask, index) => `<span class="overlay-mask" style="${pctStyle(mask)} animation-delay:${showParsing ? 0 : 180 + index * 220}ms"></span>`)
      .join("");
    return `${masks}${showParsing ? `<span class="parse-badge">Parsed answer: ${example?.prediction || ""}</span>` : ""}`;
  }

  if (config.type === "chips") {
    const chips = (config.chips || [])
      .map((chip, index) => `<span class="overlay-chip${chip.active ? " is-active" : ""}" style="left:${chip.x}%; top:${chip.y}%; animation-delay:${showParsing ? 0 : 140 + index * 150}ms">${chip.text}</span>`)
      .join("");
    return `${chips}${showParsing ? `<span class="parse-badge">Parsed answer: ${example?.prediction || ""}</span>` : ""}`;
  }

  if (config.type === "path") {
    const chips = (config.chips || [])
      .map((chip, index) => `<span class="overlay-chip" style="left:${chip.x}%; top:${chip.y}%; animation-delay:${showParsing ? 0 : 420 + index * 160}ms">${chip.text}</span>`)
      .join("");
    return `<svg class="overlay-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d="${config.path}"></path></svg>${chips}${showParsing ? `<span class="parse-badge">Parsed answer: ${example?.prediction || ""}</span>` : ""}`;
  }

  return "";
}

function renderPredictionHoleItems(config) {
  return Array.from({ length: config.holes }, (_, index) => {
    const selected = index === config.selected;
    return `<span class="prediction-hole${selected ? " is-selected" : ""}">${selected ? `<b>${index + 1}</b>` : ""}</span>`;
  }).join("");
}

function pointsToPercent(config) {
  if (config.pointsPx?.length && config.baseSize?.length === 2) {
    const [width, height] = config.baseSize;
    return config.pointsPx.map(([x, y]) => ({
      x: Number(((x / width) * 100).toFixed(2)),
      y: Number(((y / height) * 100).toFixed(2)),
    }));
  }
  if (config.points?.length && config.baseSize?.length === 2 && "xPx" in config.points[0]) {
    const [width, height] = config.baseSize;
    return config.points.map((point) => ({
      ...point,
      x: Number(((point.xPx / width) * 100).toFixed(2)),
      y: Number(((point.yPx / height) * 100).toFixed(2)),
    }));
  }
  return (config.points || []).map((point) => Array.isArray(point) ? { x: point[0], y: point[1] } : point);
}

function setExampleLevel(level, preferredExampleId) {
  activeExampleLevel = level;
  const levelIndex = Math.max(0, exampleLevelOrder.indexOf(level));

  document.querySelectorAll("[data-capability]").forEach((band) => {
    band.classList.toggle("is-active", band.dataset.capability === level);
  });
  document.querySelector("#capability-atlas")?.setAttribute("data-level", String(levelIndex + 1));
  document.querySelector("#examples")?.setAttribute("data-level", String(levelIndex + 1));

  const target = preferredExampleId || examplesForLevel(level)[0]?.id;
  setExample(target);
}

function setExample(exampleId) {
  const example = examples.find((item) => item.id === exampleId) || examples[0];

  if (example.capability !== activeExampleLevel) {
    setExampleLevel(example.capability, example.id);
    return;
  }

  clearExampleAutoplay();
  activeExampleId = example.id;
  activeExampleStep = "input";

  const stageShell = document.querySelector("#example-stage");
  stageShell?.classList.add("is-switching");
  stageShell?.setAttribute("data-step", "input");
  if (stageShell) stageShell.dataset.example = example.id;

  renderExampleStepper();
  renderVisualBase(example);
  updateExampleText(example);
  updateAtlasPreview(example.id);
  setExampleStep("input");
  queueExampleAutoplay(example.id);

  document.querySelectorAll("#capability-atlas [data-example]").forEach((button) => {
    const isActive = button.dataset.example === example.id;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
    button.tabIndex = isActive ? 0 : -1;
  });
  const activeTask = document.querySelector(`#capability-atlas [data-example="${example.id}"]`);
  if (activeTask?.id) stageShell?.setAttribute("aria-labelledby", activeTask.id);

  window.setTimeout(() => stageShell?.classList.remove("is-switching"), 180);
}

function queueExampleAutoplay(exampleId = activeExampleId) {
  clearExampleAutoplay();
  pendingExampleAutoplay = true;
  if (examplesInView) {
    pendingExampleAutoplay = false;
    autoplayExample(exampleId);
  }
}

function setupExampleAutoplayObserver() {
  const target = document.querySelector("#examples");
  if (!target) return;

  if (!("IntersectionObserver" in window)) {
    examplesInView = true;
    queueExampleAutoplay(activeExampleId);
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      examplesInView = entry.isIntersecting;
      if (entry.isIntersecting && pendingExampleAutoplay) {
        pendingExampleAutoplay = false;
        clearExampleAutoplay();
        setExampleStep("input");
        autoplayExample(activeExampleId);
      }
    });
  }, { threshold: 0.34 });

  observer.observe(target);
}

function updateExampleText(example) {
  document.querySelector("#example-kicker").textContent = `${example.capability} · ${example.label}`;
  document.querySelector("#example-title").textContent = example.short;
  const question = document.querySelector("#example-question");
  question.textContent = example.question;
  question.title = example.question;
  const protocolName = document.querySelector("#example-protocol-name");
  if (protocolName) protocolName.textContent = example.protocol;
  document.querySelector("#example-source") && (document.querySelector("#example-source").textContent = example.source);
  document.querySelector("#example-protocol") && (document.querySelector("#example-protocol").textContent = example.protocol);
  document.querySelector("#example-model") && (document.querySelector("#example-model").textContent = example.imageModel);
  const prediction = document.querySelector("#example-prediction");
  const groundTruth = document.querySelector("#example-ground-truth");
  const predictionLabel = document.querySelector("#example-prediction-label");
  const groundTruthLabel = document.querySelector("#example-ground-truth-label");
  if (predictionLabel) predictionLabel.textContent = example.predictionLabel || "Prediction";
  if (groundTruthLabel) groundTruthLabel.textContent = example.groundTruthLabel || "Ground truth";
  [
    [prediction, example.predictionDetail],
    [groundTruth, example.groundTruthDetail],
  ].forEach(([node, detail]) => {
    if (!node) return;
    if (detail) {
      node.title = detail;
      node.setAttribute("aria-label", detail);
    } else {
      node.removeAttribute("title");
      node.removeAttribute("aria-label");
    }
  });
  const note = document.querySelector("#example-note");
  note.textContent = example.note || "";
  note.hidden = !example.note;
}

function autoplayExample(exampleId) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    setExampleStep("evaluation");
    return;
  }

  const timing = overlayConfigs[exampleId]?.timing || {};
  const sequence = [
    { step: "input", delay: 0 },
    { step: "visual", delay: scaledExampleDelay(650) },
    { step: "parsing", delay: scaledExampleDelay(timing.parsing || 2450) },
    { step: "evaluation", delay: scaledExampleDelay(timing.evaluation || 3600) },
  ];

  sequence.forEach(({ step, delay }) => {
    const timer = window.setTimeout(() => {
      if (activeExampleId === exampleId) setExampleStep(step);
    }, delay);
    exampleAutoplayTimers.push(timer);
  });
}


function setFilter(filter) {
  activeFilter = filter;
  document.querySelectorAll("[data-filter]").forEach((button) => {
    const isActive = button.dataset.filter === filter;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });
  renderLeaderboard();
  renderTaskMatrix();
}

function animateNumbers(root = document) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  root.querySelectorAll("[data-count]").forEach((node) => {
    const target = Number(node.dataset.count);
    if (!Number.isFinite(target)) return;
    const decimals = Number(node.dataset.decimals || 0);
    const duration = 720;
    const start = performance.now();

    const tick = (now) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      node.textContent = (target * eased).toFixed(decimals);
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

function setupReveal() {
  const nodes = document.querySelectorAll(".reveal-on-scroll");
  if (!nodes.length) return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) {
    nodes.forEach((node) => node.classList.add("is-visible"));
    return;
  }

  document.body.classList.add("motion-ready");

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  nodes.forEach((node) => observer.observe(node));
}

function setupTableMotion() {
  const panels = document.querySelectorAll(".table-motion");
  if (!panels.length) return;

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) {
    panels.forEach((panel) => panel.classList.add("is-table-visible"));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-table-visible");
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.08 });

  panels.forEach((panel) => observer.observe(panel));
}

function setupActiveNav() {
  const links = [...document.querySelectorAll(".nav-links a")];
  const sections = links
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);
  if (!links.length || !("IntersectionObserver" in window)) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      links.forEach((link) => {
        link.classList.toggle("is-active", link.getAttribute("href") === `#${entry.target.id}`);
      });
    });
  }, { rootMargin: "-36% 0px -54% 0px", threshold: 0.01 });

  sections.forEach((section) => observer.observe(section));
}

function restoreInitialAnchor() {
  if (!window.location.hash) return;

  let id;
  try {
    id = decodeURIComponent(window.location.hash.slice(1));
  } catch {
    return;
  }

  const target = document.getElementById(id);
  if (!target) return;

  const revealNodes = [
    ...(target.matches(".reveal-on-scroll") ? [target] : []),
    ...target.querySelectorAll(".reveal-on-scroll"),
  ];
  revealNodes.forEach((node) => {
    node.classList.add("is-visible");
    if (node.matches(".table-motion")) {
      node.classList.add("is-table-visible");
    }
  });

  window.requestAnimationFrame(() => {
    target.scrollIntoView({ behavior: "instant", block: "start" });
  });
}

function setupCitationCopy() {
  const button = document.querySelector("#copy-citation");
  const bibtex = document.querySelector("#bibtex");
  if (!button || !bibtex) return;

  button.addEventListener("click", async () => {
    const original = button.textContent;
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      button.textContent = "Copied";
    } catch {
      button.textContent = "Copy failed";
    }
    window.setTimeout(() => {
      button.textContent = original;
    }, 1400);
  });
}

async function setupPageViewCounter() {
  const counter = document.querySelector("#page-view-counter");
  const countNode = document.querySelector("#page-view-count");
  const labelNode = document.querySelector("#page-view-label");
  if (!counter || !countNode || !labelNode) return;

  const endpoint = "https://api.counterapi.dev/v1/show-dont-tell-spatialgen-bench/homepage";
  const sessionKey = "show-dont-tell-page-view-counted";
  const isLocalPreview = ["", "localhost", "127.0.0.1"].includes(window.location.hostname);
  let shouldIncrement = !isLocalPreview;

  try {
    if (window.sessionStorage.getItem(sessionKey)) shouldIncrement = false;
  } catch {
    shouldIncrement = !isLocalPreview;
  }

  try {
    const response = await fetch(`${endpoint}/${shouldIncrement ? "up" : ""}`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
      referrerPolicy: "no-referrer",
    });
    if (!response.ok) return;

    const data = await response.json();
    const count = Number(data.count);
    if (!Number.isFinite(count) || count < 0) return;

    if (shouldIncrement) {
      try {
        window.sessionStorage.setItem(sessionKey, "1");
      } catch {
        // The counter still works when browser storage is unavailable.
      }
    }

    countNode.textContent = new Intl.NumberFormat("en-US").format(count);
    labelNode.textContent = count === 1 ? "view" : "views";
    counter.hidden = false;
  } catch {
    counter.hidden = true;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (!window.location.hash) {
    window.scrollTo(0, 0);
  }

  document.querySelectorAll("[data-filter]").forEach((button) => {
    button.addEventListener("click", () => setFilter(button.dataset.filter));
  });
  document.querySelector("#example-replay")?.addEventListener("click", () => {
    clearExampleAutoplay();
    setExampleStep("input", { manual: true });
    queueExampleAutoplay(activeExampleId);
  });
  createCapabilityAtlas();
  setFilter(activeFilter);
  setupTaskStickyHeader();
  setupExampleAnimationPacing();
  const requestedExampleId = new URLSearchParams(window.location.search).get("example");
  const requestedStep = new URLSearchParams(window.location.search).get("step");
  const initialExample = examples.find((example) => example.id === requestedExampleId) || examples[0];
  setExampleLevel(initialExample.capability, initialExample.id);
  if (pipelineSteps.some((step) => step.id === requestedStep)) {
    clearExampleAutoplay();
    setExampleStep(requestedStep, { manual: true });
  }
  window.addEventListener("resize", queueOverlaySync);
  setupExampleAutoplayObserver();
  setupAtlasPreviewRotation();
  setupReveal();
  setupTableMotion();
  setupActiveNav();
  setupCitationCopy();
  setupPageViewCounter();
  restoreInitialAnchor();
});
