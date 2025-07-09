# CustomGPT - AI Emotion Detection & Chatbot System

An advanced emotion detection and chatbot system that combines machine learning models with conversational AI to analyze emotions in text and provide intelligent responses.

# Table of content

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

CustomGPT is a comprehensive AI system that integrates emotion detection with chatbot capabilities. The project features:

- **Emotion Detection Pipeline**: Uses LightGBM models trained on emotion datasets to detect and classify emotions in text
- **Local Chatbot**: Powered by Microsoft's Phi-4 model for offline text generation
- **OpenAI Integration**: Customer service chatbot with emotion-aware responses
- **Multi-Model Support**: Includes RoBERTa and other transformer models for emotion analysis
- **Data Processing**: Complete pipeline for emotion dataset preparation and model training

The system is particularly useful for customer service applications, sentiment analysis, and building emotion-aware conversational AI.

## Installation

You will need Python 3.8+ installed on your machine to run this project.

### Prerequisites

- Python 3.8 or higher
- Git
- Optional: CUDA-capable GPU for faster model inference

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Luxchar/CustomGPT.git
cd CustomGPT
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. **For OpenAI integration**: Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

4. **For emotion detection**: The pre-trained LightGBM models are included in the `notebooks/lightgbm_ultimate_saved/` directory.

## Usage

### 1. Local Chatbot (Offline)
Run the local Phi-4 powered chatbot that works entirely offline:
```bash
streamlit run src/main.py
```

### 2. OpenAI-powered Customer Service Bot
Run the emotion-aware customer service chatbot:
```bash
streamlit run src/main_openai.py
```

### 3. Emotion Detection Pipeline
Use the emotion detection system programmatically:

```python
from src.emotion_pipeline import EmotionDetectionPipeline

# Initialize the pipeline
pipeline = EmotionDetectionPipeline()
model_path = "./notebooks/lightgbm_ultimate_saved/lightgbm_ultimate_rescued_20250706_060702"
pipeline.load_model(model_path)

# Analyze a single text
result = pipeline.predict_single("I absolutely love this product!")
print(f"Major emotion: {result['major_emotion']} ({result['major_confidence']:.3f})")

# Batch processing
texts = ["Great product!", "Terrible service!", "It's okay, I guess."]
df_results = pipeline.predict_batch(texts, save_csv="predictions.csv")
```

### 4. Data Processing and Training
Explore the Jupyter notebooks for data processing and model training:
```bash
jupyter notebook notebooks/
```

Key notebooks:
- `data_prep_yann.ipynb`: Data preparation and preprocessing
- `detecteur_emotion_*.ipynb`: Emotion detection model training
- `text-generation*.ipynb`: Text generation experiments

## Features

### Emotion Detection
- **30+ Emotion Categories**: Detects emotions like joy, anger, sadness, fear, surprise, etc.
- **Multi-Model Architecture**: LightGBM ensemble with TF-IDF and SVD features
- **Batch Processing**: Efficient processing of large text datasets
- **Confidence Scores**: Returns probability scores for all emotions

### Chatbot Capabilities
- **Local Inference**: Phi-4 model runs entirely on your machine
- **Streaming Responses**: Real-time text generation with streaming UI
- **Emotion-Aware Responses**: Customer service bot adapts responses based on detected emotions
- **Memory**: Maintains conversation context across interactions

### Data Processing
- **Multiple Datasets**: Support for GoEmotions and custom emotion datasets
- **Feature Engineering**: Advanced text preprocessing and feature extraction
- **Model Training**: Complete pipeline for training custom emotion detection models

## Model Details

### Emotion Detection Models
- **Primary Model**: LightGBM Ultimate with TF-IDF and SVD features
- **Alternative Models**: RoBERTa-based emotion classifiers
- **Training Data**: GoEmotions dataset and custom review datasets
- **Features**: Word/character n-grams, semantic embeddings, statistical features

### Language Models
- **Local Model**: Microsoft Phi-4-mini-instruct (CPU optimized)
- **Cloud Model**: OpenAI GPT-3.5-turbo integration
- **Optimization**: Quantization and optimization for fast inference

## Configuration

### Environment Variables
Create a `.env` file with the following variables:
```bash
OPENAI_API_KEY=your_openai_api_key
MODEL_PATH=./notebooks/lightgbm_ultimate_saved/lightgbm_ultimate_rescued_20250706_060702
```

### Model Paths
Update model paths in the configuration files:
- Emotion models: `notebooks/lightgbm_ultimate_saved/`
- Language models: Downloaded automatically via Hugging Face

### Streamlit Configuration
For production deployment, configure Streamlit in `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

## Contributing

If you want to contribute to this project, you can fork this repository and make a pull request with your changes.

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation for any API changes

### Areas for Contribution
- Additional emotion categories
- Performance optimizations
- New model architectures
- UI/UX improvements
- Documentation and tutorials

Anyone is welcome to contribute to this project.

## License

This project is under the MIT license.

---

## Quick Start Example

```python
# Quick emotion detection
from src.emotion_pipeline import quick_predict

result = quick_predict("This product is absolutely amazing!")
print(f"Emotion: {result['major_emotion']} (confidence: {result['major_confidence']:.2f})")
```

For more examples and detailed documentation, check the `notebooks/` directory.
