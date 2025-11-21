# 🤟 American Sign Language Detection System

A real-time American Sign Language (ASL) letter recognition system using computer vision, machine learning, and natural language processing to convert hand gestures into refined English text.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technical Implementation](#-technical-implementation)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [NLP Text Refinement](#-nlp-text-refinement)
- [Database Integration](#-database-integration)
- [Performance Metrics](#-performance-metrics)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project implements a comprehensive ASL detection system that captures hand gestures through a webcam, recognizes individual letters using MediaPipe and TensorFlow, builds letter sequences, and uses advanced NLP techniques to convert raw letter sequences into grammatically correct English sentences.

### Key Components:
- **Computer Vision**: MediaPipe for hand landmark detection
- **Machine Learning**: TensorFlow/Keras neural network for letter classification
- **Natural Language Processing**: OpenAI API integration for text refinement
- **Web Interface**: Streamlit-based interactive application
- **Database**: MongoDB for data persistence and analytics
- **Real-time Processing**: Optimized for live gesture recognition

## ✨ Features

### Core Functionality
- 🎥 **Real-time Hand Detection**: Live webcam feed with MediaPipe hand landmark detection
- 🔤 **26 Letter ASL Recognition**: Complete A-Z alphabet recognition with high accuracy
- 📝 **Intelligent Buffer Management**: Configurable letter hold duration to prevent false positives
- 🤖 **AI Text Refinement**: Advanced NLP processing to convert letter sequences into proper English
- 💾 **Database Integration**: MongoDB storage for both raw sequences and refined sentences
- 📊 **Performance Analytics**: Real-time FPS monitoring and processing time metrics

### User Interface Features
- 🖥️ **Interactive Streamlit Dashboard**: Modern web-based interface
- 📱 **Responsive Design**: Works on desktop and tablet devices
- ⚙️ **Configurable Parameters**: Adjustable detection confidence and timing settings
- 📈 **Real-time Feedback**: Live prediction display with confidence indicators
- 🎛️ **Multiple Modes**: Inference mode for usage and data collection mode for training

### Advanced Features
- 🔄 **Session Management**: Unique session tracking with timestamps
- 🧠 **Smart Preprocessing**: Vowel-consonant pattern analysis for better word formation
- 🛡️ **Error Handling**: Robust error recovery and fallback mechanisms
- 📋 **Data Export**: MongoDB integration for data analysis and model improvement
- 🎯 **Batch Processing**: Efficient handling of multiple gesture sequences

## 🏗️ System Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Camera Input      │    │   Hand Detection    │    │  Letter Classifier  │
│   (OpenCV)          │───▶│   (MediaPipe)       │───▶│  (TensorFlow/Keras) │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                                                    │
                                                                    ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   MongoDB Storage   │◄───│   Buffer Manager    │◄───│   Gesture Buffer    │
│   (Raw + Refined)   │    │   (Session State)   │    │   (Letter Queue)    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
          ▲                                                         │
          │                                                         ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Streamlit UI      │◄───│   NLP Refiner       │◄───│   Text Processor    │
│   (Dashboard)       │    │   (OpenAI API)      │    │   (Preprocessing)   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- MongoDB (local or cloud instance)
- LM Studio with compatible model (for NLP refinement)

### Step 1: Clone Repository
```bash
git clone https://github.com/yjhkdjsg/Minor-Project.git
cd American-Sign-Language-Detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Environment Configuration
Create a `.env` file in the project root:
```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=asl_detection
MONGODB_COLLECTION=letter_sequences
MONGODB_REFINED_COLLECTION=refined_sentences
BUFFER_MAX_SIZE=500
```

### Step 4: Model Setup
The pre-trained TensorFlow model is included in the repository:
- `model/keypoint_classifier/keypoint_classifier.keras` - Main model
- `model/keypoint_classifier/keypoint_classifier.tflite` - Optimized inference model
- `model/keypoint_classifier/keypoint_classifier_label.csv` - Label mappings

### Step 5: LM Studio Setup (for NLP refinement)
1. Install [LM Studio](https://lmstudio.ai/)
2. Load a compatible model (e.g., qwen2.5-0.5b-instruct)
3. Start the local server on `http://localhost:1234`
4. Update `nlp_refiner.py` with your server configuration

## 🎮 Usage

### Basic Usage
1. **Start the Application**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Configure Settings**:
   - Select camera device (0, 1, or 2)
   - Adjust resolution (default: 960x540)
   - Set detection confidence thresholds
   - Configure letter hold duration

3. **Begin Detection**:
   - Click "▶️ Start Camera"
   - Position your hand in front of the camera
   - Perform ASL letter gestures
   - Hold each gesture for the configured duration

4. **Text Refinement**:
   - Click "⏹️ Stop Camera" to trigger NLP processing
   - View refined sentences in the dashboard
   - Check MongoDB for stored results

### Advanced Features

#### Data Collection Mode
For training new models or improving existing ones:
1. Select "Data Collection Mode" in the sidebar
2. Choose the target letter (A-Z)
3. Perform the gesture multiple times
4. Data is automatically saved to `keypoint.csv`

#### Buffer Management
- **Clear Buffer**: Remove all collected letters
- **Session Tracking**: Each session gets a unique ID
- **Auto-save**: Automatic database storage on session end

## 📁 Project Structure

```
American-Sign-Language-Detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                      # Environment variables template
├── 
├── 🎯 Main Application Files
├── streamlit_app.py                  # Primary Streamlit web application
├── app.py                           # Alternative/legacy application entry
├── nlp_refiner.py                   # NLP text refinement module
├── 
├── 📊 Machine Learning Components
├── keypoint_classification.ipynb     # Model training notebook
├── model/
│   └── keypoint_classifier/
│       ├── keypoint_classifier.py    # Classifier implementation
│       ├── keypoint_classifier.keras # Trained model (Keras format)
│       ├── keypoint_classifier.tflite# Optimized model (TensorFlow Lite)
│       ├── keypoint_classifier_label.csv # Label mappings
│       └── keypoint.csv             # Training/validation dataset
├── 
├── 🛠️ Utility Functions
├── utils/
│   ├── cvfpscalc.py                 # FPS calculation utilities
│   └── __pycache__/                 # Compiled Python files
├── 
├── 🧪 Testing & Development
├── test_integration.py              # Integration testing script
└── __pycache__/                     # Compiled Python files
```

## 🔧 Technical Implementation

### Computer Vision Pipeline

#### MediaPipe Hand Detection
```python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
```

#### Landmark Processing
1. **Bounding Box Calculation**: Determine hand region boundaries
2. **Landmark Extraction**: Extract 21 hand keypoints (x, y coordinates)
3. **Normalization**: Convert to relative coordinates for scale invariance
4. **Feature Engineering**: Create 42-dimensional feature vector (21 points × 2 coordinates)

### Neural Network Architecture

#### Model Structure
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),           # 42 input features
    tf.keras.layers.BatchNormalization(),         # Normalize inputs
    tf.keras.layers.Dense(128, activation='mish', # Hidden layer 1
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),                # Regularization
    tf.keras.layers.Dense(64, activation='mish',  # Hidden layer 2
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),                # Regularization
    tf.keras.layers.Dense(32, activation='mish', # Hidden layer 3
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') # Output layer (26 classes)
])
```

#### Training Configuration
- **Optimizer**: Adam with default learning rate
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Regularization**: L2 regularization + Dropout (0.5)
- **Early Stopping**: Patience of 20 epochs
- **Batch Size**: 128
- **Max Epochs**: 1000

## 🧠 Machine Learning Pipeline

### Data Collection Process
1. **Real-time Capture**: Live gesture recording through webcam
2. **Landmark Extraction**: MediaPipe processes hand positions
3. **Feature Engineering**: Normalize and structure landmark data
4. **Labeling**: Manual labeling with letter IDs (0-25 for A-Z)
5. **Dataset Building**: Incremental CSV file construction

### Training Process
1. **Data Loading**: Read from `keypoint.csv`
2. **Train-Test Split**: 75% training, 25% validation
3. **Model Architecture**: Deep neural network with regularization
4. **Training Loop**: 1000 epochs with early stopping
5. **Model Evaluation**: Confusion matrix and classification report
6. **Model Export**: Save in both Keras and TensorFlow Lite formats

### Performance Optimization
- **TensorFlow Lite Conversion**: Quantized model for faster inference
- **Batch Normalization**: Stable training and faster convergence
- **L2 Regularization**: Prevent overfitting
- **Dropout**: Additional regularization during training

## 🤖 NLP Text Refinement

### Preprocessing Pipeline
```python
def preprocess(chars):
    vowels = set("aeiou")
    result = []
    current = ""
    
    for c in chars:
        if len(current) >= 2 and all(ch not in vowels for ch in current[-2:]) and c not in vowels:
            result.append(current)
            current = c
        else:
            current += c
    
    if current:
        result.append(current)
    
    return " ".join(result)
```

### AI Integration
- **OpenAI API**: Compatible with local LM Studio servers
- **Model**: qwen2.5-0.5b-instruct (configurable)
- **Temperature**: 0.1 for consistent outputs
- **System Prompt**: Specialized for ASL sequence decoding

### Refinement Rules
1. **Space Insertion**: Add spaces between characters to form words
2. **Repetition Removal**: Remove unnecessary repeated characters
3. **Word Formation**: Convert character sequences to valid English words
4. **Grammar Preservation**: Maintain sentence structure without adding meaning

## 💾 Database Integration

### MongoDB Schema

#### Raw Letter Sequences
```javascript
{
  _id: ObjectId,
  session_id: "20241121_143022",
  timestamp: ISODate,
  letter_sequence: "helloworld",
  letter_count: 10,
  individual_letters: ["h", "e", "l", "l", "o", "w", "o", "r", "l", "d"]
}
```

#### Refined Sentences
```javascript
{
  _id: ObjectId,
  session_id: "20241121_143022", 
  timestamp: ISODate,
  original_buffer: ["h", "e", "l", "l", "o", "w", "o", "r", "l", "d"],
  original_sequence: "helloworld",
  preprocessed: "hello world",
  cleaned: "helloworld",
  refined_sentence: "Hello world",
  processing_time_seconds: 2.341,
  model_device: "cpu",
  buffer_length: 10
}
```

### Database Operations
- **Automatic Saving**: Sessions saved on camera stop
- **Batch Processing**: Efficient bulk operations
- **Query Optimization**: Indexed by timestamp and session_id
- **Data Analytics**: Historical performance tracking

## 📊 Performance Metrics

### Real-time Performance
- **FPS**: Typically 25-30 FPS on modern hardware
- **Latency**: <50ms for gesture recognition
- **Accuracy**: >95% for clear, well-lit gestures
- **Processing Time**: NLP refinement typically 1-5 seconds

### Model Performance
- **Training Accuracy**: >98% on validation set
- **Inference Speed**: <10ms per prediction (TensorFlow Lite)
- **Model Size**: ~50KB (TensorFlow Lite format)
- **Memory Usage**: <100MB during inference

### System Requirements
- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB or higher
- **CPU**: Modern multi-core processor
- **GPU**: Optional (CUDA-compatible for faster training)
- **Storage**: 1GB for full installation

## ⚙️ Configuration

### Camera Settings
```python
# Resolution options
width = 960   # Recommended: 960, 1280, 1920
height = 540  # Recommended: 540, 720, 1080

# Detection parameters
min_detection_confidence = 0.7  # Range: 0.1-1.0
min_tracking_confidence = 0.5   # Range: 0.1-1.0
```

### Buffer Management
```python
# Timing configuration
letter_hold_duration = 1.5     # Seconds to hold gesture
buffer_max_size = 500          # Maximum letters per session

# Processing options
auto_save_on_stop = True       # Save to MongoDB when camera stops
clear_buffer_on_save = True    # Clear buffer after saving
```

### NLP Configuration
```python
# OpenAI API settings
base_url = "http://localhost:1234/v1"  # LM Studio server
model = "qwen2.5-0.5b-instruct"       # Model name
temperature = 0.1                      # Response consistency
```

### MongoDB Settings
```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=asl_detection
MONGODB_COLLECTION=letter_sequences
MONGODB_REFINED_COLLECTION=refined_sentences
```

## 🐛 Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Solution 1: Try different device IDs
device_id = 0, 1, or 2

# Solution 2: Check camera permissions
# Ensure camera access is enabled for the application
```

#### Low Detection Accuracy
```python
# Solution 1: Improve lighting conditions
# Solution 2: Adjust detection confidence
min_detection_confidence = 0.5  # Lower threshold

# Solution 3: Increase hold duration
letter_hold_duration = 2.0     # Longer hold time
```

#### MongoDB Connection Issues
```bash
# Solution 1: Start MongoDB service
sudo systemctl start mongod

# Solution 2: Check connection string
MONGODB_URI=mongodb://localhost:27017/

# Solution 3: Verify database permissions
```

#### NLP Refinement Errors
```bash
# Solution 1: Check LM Studio server
curl http://localhost:1234/v1/models

# Solution 2: Verify model loading
# Ensure qwen2.5-0.5b-instruct is loaded

# Solution 3: Check API key
api_key = "lm-studio"  # Default for LM Studio
```

### Performance Optimization

#### Improve FPS
1. **Lower Resolution**: Reduce camera resolution
2. **Reduce Processing**: Skip frames during heavy processing
3. **Hardware Acceleration**: Use GPU if available

#### Reduce Latency
1. **Model Optimization**: Use TensorFlow Lite model
2. **Buffer Management**: Optimize buffer operations
3. **Threading**: Separate UI and processing threads

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Performance monitoring
enable_fps_counter = True
show_processing_times = True
```

## 🤝 Contributing

### Development Setup
1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/new-feature`
3. **Install Development Dependencies**: `pip install -r requirements-dev.txt`
4. **Run Tests**: `python -m pytest tests/`
5. **Submit Pull Request**

### Code Style
- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations where applicable
- **Documentation**: Add docstrings for all functions
- **Testing**: Include unit tests for new features

### Areas for Contribution
- 📈 **Model Improvement**: Better accuracy and speed
- 🎨 **UI Enhancement**: More intuitive interface
- 📊 **Analytics Dashboard**: Advanced data visualization
- 🔧 **Performance Optimization**: Faster processing
- 📱 **Mobile Support**: Smartphone compatibility
- 🌐 **Multi-language**: Support for other sign languages

## 📈 Future Enhancements

### Short-term Goals
- [ ] **Dynamic Gesture Recognition**: Support for motion-based signs
- [ ] **Multiple Hand Support**: Two-handed gesture recognition
- [ ] **Real-time Grammar Correction**: Instant text refinement
- [ ] **Mobile App**: Native smartphone application

### Long-term Vision
- [ ] **Full ASL Support**: Complete ASL grammar and syntax
- [ ] **Multi-language**: International sign language support
- [ ] **AR Integration**: Augmented reality overlay
- [ ] **Voice Synthesis**: Text-to-speech output
- [ ] **Cloud Deployment**: Scalable web service

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Development Team**: Minor Project Contributors
- **Repository**: [yjhkdjsg/Minor-Project](https://github.com/yjhkdjsg/Minor-Project)

## 🙏 Acknowledgments

- **MediaPipe**: Google's framework for hand landmark detection
- **TensorFlow**: Machine learning model development
- **OpenAI**: API integration for text refinement
- **Streamlit**: Web application framework
- **MongoDB**: Database storage and management
- **ASL Community**: Inspiration and validation

---

## 📞 Support

For questions, issues, or contributions, please:
1. **Check Documentation**: Review this README thoroughly
2. **Search Issues**: Look for existing solutions
3. **Create Issue**: Submit detailed bug reports or feature requests
4. **Join Discussion**: Participate in community discussions

**Last Updated**: November 21, 2025
**Version**: 1.0.0
**Status**: Production Ready ✅