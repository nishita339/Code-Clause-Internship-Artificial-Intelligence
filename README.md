# CV Personality Prediction System

An intelligent system that analyzes CVs/resumes to predict Big Five personality traits using advanced NLP and machine learning techniques.

## Features

### Core Functionality
- **Multi-format Support**: Accepts CVs in PDF, DOCX, TXT, and image formats
- **OCR Capabilities**: Extracts text from scanned documents and images using Tesseract
- **Advanced NLP**: Comprehensive text preprocessing, tokenization, and linguistic analysis
- **Big Five Prediction**: Predicts all five personality traits:
  - Openness to Experience
  - Conscientiousness  
  - Extroversion
  - Agreeableness
  - Neuroticism

### Advanced Features
- **Ensemble Models**: Uses multiple ML algorithms (Random Forest, Gradient Boosting, SVM)
- **Deep Learning Support**: Optional BERT embeddings for enhanced feature extraction
- **Comprehensive Visualization**: Multiple chart types including radar plots and comparisons
- **Detailed Analysis**: Section-based analysis (Education, Experience, Skills, etc.)
- **Sentiment Analysis**: Incorporates emotional tone and linguistic patterns

## Technology Stack

### Core Dependencies
- **Python 3.11+**
- **NLTK**: Natural language processing
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Document Processing
- **PyPDF2**: PDF text extraction
- **python-docx**: DOCX file processing
- **pytesseract**: OCR for image text extraction
- **opencv-python**: Image preprocessing
- **Pillow**: Image handling

### Visualization & Analysis
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualization
- **textblob**: Sentiment analysis

### Optional Deep Learning
- **transformers**: BERT model integration
- **torch**: PyTorch for neural networks

## Installation

```bash
# Clone or download the files
# Install dependencies (already installed in this environment)
python -m pip install nltk scikit-learn pandas numpy matplotlib seaborn textblob PyPDF2 python-docx Pillow opencv-python pytesseract joblib
```

## Usage

### Basic Usage

```python
from personality_predictor import CVPersonalityPredictor

# Initialize the predictor
predictor = CVPersonalityPredictor()

# Analyze a CV
results = predictor.analyze_cv('path/to/cv.pdf')

# Results include:
# - predictions: Dictionary of personality scores
# - summary: Text interpretation
# - visualization_path: Path to generated charts
```

### Advanced Usage

```python
from advanced_personality_predictor import CVPersonalityAnalyzer

# Initialize with deep learning (if available)
analyzer = CVPersonalityAnalyzer(use_deep_learning=True)

# Complete analysis with ensemble models
results = analyzer.analyze_cv_complete('path/to/cv.pdf')

# Access detailed results
personality_scores = results['scores']
interpretation = results['interpretation']
```

## File Structure

```
cv-personality-predictor/
├── personality_predictor.py          # Basic system implementation
├── advanced_personality_predictor.py # Enhanced version with deep learning
├── demo_cv.txt                      # Sample CV for testing
├── models/                          # Trained model storage
├── results/                         # Analysis results (JSON/CSV)
├── visualizations/                  # Generated charts and plots
└── README.md                        # This file
```

## Output Examples

### Personality Scores
```json
{
  "openness": 0.75,
  "conscientiousness": 0.85,
  "extroversion": 0.60,
  "agreeableness": 0.70,
  "neuroticism": 0.25
}
```

### Generated Visualizations
- **Bar Charts**: Individual trait scores with confidence indicators
- **Radar Plots**: Comprehensive personality profile overview
- **Comparison Charts**: Scores vs. population norms
- **Detailed Interpretations**: Text-based personality summaries

## Model Architecture

### Feature Extraction
1. **Linguistic Features**: Word count, sentence structure, vocabulary richness
2. **POS Tagging**: Part-of-speech analysis for communication patterns
3. **Sentiment Analysis**: Emotional tone and subjectivity measures
4. **Keyword Analysis**: Personality-specific vocabulary detection
5. **Section Analysis**: Role-specific content analysis (Education, Experience, etc.)

### Machine Learning Pipeline
1. **Text Preprocessing**: Cleaning, tokenization, lemmatization
2. **Feature Engineering**: Multi-dimensional feature vector creation
3. **Ensemble Training**: Multiple algorithm combination
4. **Cross-Validation**: Model performance optimization
5. **Prediction Aggregation**: Weighted ensemble outputs

## Accuracy & Validation

The system uses synthetic training data for demonstration purposes. In a production environment, you would:

1. **Collect Validated Data**: CVs with known personality assessments
2. **Professional Validation**: Use standardized personality tests as ground truth
3. **Cross-Industry Training**: Include diverse professional backgrounds
4. **Continuous Learning**: Update models with new validated examples

## Ethical Considerations

- **Bias Awareness**: Be mindful of cultural and demographic biases
- **Transparency**: Clearly communicate prediction limitations
- **Privacy**: Ensure secure handling of personal information
- **Consent**: Obtain explicit permission for personality analysis
- **Professional Use**: Supplement, don't replace, human judgment

## Limitations

- **Synthetic Training**: Current models use demonstrative data
- **Language Support**: Optimized for English-language CVs
- **Cultural Context**: May not account for cultural communication differences
- **Professional Domains**: Best suited for technical/professional roles

## Future Enhancements

- **Multi-language Support**: Extend to other languages
- **Real Training Data**: Integrate with validated personality datasets
- **Web Interface**: Browser-based analysis platform
- **API Integration**: RESTful service for enterprise use
- **Continuous Learning**: Adaptive model improvement

## Contributing

This system is designed for educational and research purposes. To contribute:

1. **Data Collection**: Help gather validated CV-personality pairs
2. **Model Improvement**: Enhance feature extraction techniques
3. **Validation Studies**: Conduct empirical accuracy assessments
4. **Ethical Guidelines**: Develop responsible use frameworks

## License

This project is for educational and research purposes. Please ensure compliance with privacy laws and ethical guidelines when using personality prediction systems.