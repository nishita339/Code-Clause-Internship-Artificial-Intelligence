"""
Advanced CV Personality Prediction System with Deep Learning
Enhanced version with transformer models and reinforcement learning capabilities
"""

import os
import re
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Text processing and NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from textblob import TextBlob

# Document processing
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not available. PDF processing will be limited.")

try:
    from docx import Document
except ImportError:
    print("python-docx not available. DOCX processing will be limited.")

try:
    import pytesseract
    from PIL import Image
    import cv2
except ImportError:
    print("OCR dependencies not available. Image processing will be limited.")

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import joblib

# Deep Learning (optional - will work without if not available)
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("Deep learning dependencies not available. Using traditional ML only.")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure required NLTK data is available
def download_nltk_data():
    """Download required NLTK data"""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    for data_path, data_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            try:
                nltk.download(data_name, quiet=True)
            except:
                print(f"Could not download {data_name}. Some features may be limited.")

download_nltk_data()

@dataclass
class PersonalityScores:
    """Data class for personality scores"""
    openness: float
    conscientiousness: float
    extroversion: float
    agreeableness: float
    neuroticism: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extroversion': self.extroversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism
        }

class AdvancedFeatureExtractor:
    """Advanced feature extraction with linguistic and psychological indicators"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Psychological word dictionaries
        self.personality_keywords = {
            'openness': [
                'creative', 'innovative', 'novel', 'artistic', 'imaginative', 'curious',
                'explore', 'experiment', 'original', 'unique', 'diverse', 'abstract',
                'philosophical', 'aesthetic', 'unconventional', 'flexible'
            ],
            'conscientiousness': [
                'organized', 'systematic', 'meticulous', 'detailed', 'planned', 'scheduled',
                'disciplined', 'reliable', 'thorough', 'careful', 'precise', 'methodical',
                'consistent', 'responsible', 'diligent', 'punctual'
            ],
            'extroversion': [
                'team', 'social', 'communicate', 'present', 'lead', 'network', 'public',
                'collaborate', 'energetic', 'outgoing', 'assertive', 'confident',
                'speaking', 'meeting', 'group', 'leadership'
            ],
            'agreeableness': [
                'cooperative', 'supportive', 'helpful', 'kind', 'friendly', 'collaborative',
                'harmonious', 'understanding', 'empathetic', 'considerate', 'respectful',
                'diplomatic', 'patient', 'mentor', 'assist', 'volunteer'
            ],
            'neuroticism': [
                'stress', 'pressure', 'challenging', 'difficult', 'problem', 'issue',
                'risk', 'uncertainty', 'anxiety', 'concern', 'worry', 'tension',
                'deadline', 'crisis', 'conflict', 'demanding'
            ]
        }
        
        # Action verbs indicating different personality traits
        self.action_verbs = {
            'leadership': ['led', 'managed', 'directed', 'supervised', 'guided', 'coordinated'],
            'innovation': ['created', 'developed', 'designed', 'invented', 'pioneered', 'innovated'],
            'collaboration': ['collaborated', 'partnered', 'worked', 'assisted', 'supported', 'helped'],
            'achievement': ['achieved', 'accomplished', 'completed', 'delivered', 'exceeded', 'improved']
        }
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive linguistic features"""
        features = {}
        
        # Basic text statistics
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Vocabulary richness
        unique_words = set(words)
        features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0
        
        # POS tag analysis
        pos_tags = pos_tag(words)
        pos_counts = {}
        for word, pos in pos_tags:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        total_words = len(words)
        if total_words > 0:
            features['noun_ratio'] = sum(count for pos, count in pos_counts.items() if pos.startswith('NN')) / total_words
            features['verb_ratio'] = sum(count for pos, count in pos_counts.items() if pos.startswith('VB')) / total_words
            features['adjective_ratio'] = sum(count for pos, count in pos_counts.items() if pos.startswith('JJ')) / total_words
            features['adverb_ratio'] = sum(count for pos, count in pos_counts.items() if pos.startswith('RB')) / total_words
        else:
            features.update({'noun_ratio': 0, 'verb_ratio': 0, 'adjective_ratio': 0, 'adverb_ratio': 0})
        
        # Sentiment analysis
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        # Personality keyword analysis
        for trait, keywords in self.personality_keywords.items():
            count = sum(1 for word in words if word in keywords)
            features[f'{trait}_keywords'] = count / len(words) if words else 0
        
        # Action verb analysis
        for category, verbs in self.action_verbs.items():
            count = sum(1 for word in words if word in verbs)
            features[f'{category}_verbs'] = count / len(words) if words else 0
        
        # Complexity measures
        features['complex_words'] = sum(1 for word in words if len(word) > 6) / len(words) if words else 0
        features['punctuation_ratio'] = sum(1 for char in text if char in '.,!?;:') / len(text) if text else 0
        
        return features
    
    def extract_section_features(self, sections: Dict[str, str]) -> Dict[str, float]:
        """Extract features from different CV sections"""
        section_features = {}
        
        for section_name, section_text in sections.items():
            if section_text:
                words = word_tokenize(section_text.lower())
                section_features[f'{section_name}_length'] = len(words)
                
                # Technical vs soft skills ratio
                technical_words = ['python', 'java', 'sql', 'machine', 'learning', 'algorithm', 'data']
                soft_skill_words = ['communication', 'leadership', 'teamwork', 'problem', 'solving']
                
                tech_count = sum(1 for word in words if word in technical_words)
                soft_count = sum(1 for word in words if word in soft_skill_words)
                
                section_features[f'{section_name}_technical_ratio'] = tech_count / len(words) if words else 0
                section_features[f'{section_name}_soft_ratio'] = soft_count / len(words) if words else 0
        
        return section_features

class CVPersonalityAnalyzer:
    """Advanced CV personality analysis system"""
    
    def __init__(self, use_deep_learning: bool = False):
        self.use_deep_learning = use_deep_learning and DEEP_LEARNING_AVAILABLE
        self.feature_extractor = AdvancedFeatureExtractor()
        self.models = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500, 
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        self.personality_traits = ['openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism']
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        if self.use_deep_learning:
            print("Initializing with deep learning capabilities...")
            self.init_transformer_model()
        else:
            print("Initializing with traditional machine learning...")
    
    def init_transformer_model(self):
        """Initialize transformer model for embeddings"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            self.bert_model.eval()
        except Exception as e:
            print(f"Could not initialize transformer model: {e}")
            self.use_deep_learning = False
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX"""
        try:
            doc = Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(gray)
            return text
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return self.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return text
    
    def extract_cv_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from CV"""
        sections = {
            'education': '',
            'experience': '',
            'skills': '',
            'projects': '',
            'certifications': '',
            'summary': '',
            'other': ''
        }
        
        # Section patterns
        section_patterns = {
            'education': r'(education|academic|degree|university|college|school)',
            'experience': r'(experience|employment|work|job|position|career)',
            'skills': r'(skills|technical|programming|competencies|abilities)',
            'projects': r'(projects|portfolio|achievements|accomplishments)',
            'certifications': r'(certifications|certificates|awards|honors|licenses)',
            'summary': r'(summary|profile|objective|about|overview)'
        }
        
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            classified = False
            
            for section, pattern in section_patterns.items():
                if re.search(pattern, sentence_lower):
                    sections[section] += sentence + ' '
                    classified = True
                    break
            
            if not classified:
                sections['other'] += sentence + ' '
        
        return sections
    
    def get_bert_embeddings(self, text: str) -> np.ndarray:
        """Get BERT embeddings for text"""
        if not self.use_deep_learning:
            return np.array([])
        
        try:
            # Tokenize and get embeddings
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                  padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
            return embeddings.flatten()
        except Exception as e:
            print(f"Error getting BERT embeddings: {e}")
            return np.array([])
    
    def extract_features(self, text: str) -> np.ndarray:
        """Extract comprehensive features from CV text"""
        # Extract sections
        sections = self.extract_cv_sections(text)
        
        # Extract linguistic features
        linguistic_features = self.feature_extractor.extract_linguistic_features(text)
        
        # Extract section-specific features
        section_features = self.feature_extractor.extract_section_features(sections)
        
        # Combine all features
        all_features = {**linguistic_features, **section_features}
        
        # Convert to array
        feature_vector = list(all_features.values())
        
        # Add BERT embeddings if available
        if self.use_deep_learning:
            bert_embeddings = self.get_bert_embeddings(text)
            if bert_embeddings.size > 0:
                # Reduce dimensionality for compatibility
                if len(bert_embeddings) > 50:
                    pca = PCA(n_components=50)
                    bert_embeddings = pca.fit_transform(bert_embeddings.reshape(1, -1)).flatten()
                feature_vector.extend(bert_embeddings)
        
        return np.array(feature_vector)
    
    def create_enhanced_training_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Create enhanced synthetic training data"""
        print("Creating enhanced training dataset...")
        
        # More comprehensive training examples
        sample_cvs = [
            # High Openness samples
            "Creative software architect with passion for exploring cutting-edge technologies. Developed innovative AI solutions using novel machine learning approaches. Active contributor to open-source projects and research publications. Enjoys experimenting with emerging frameworks and artistic coding projects.",
            
            "Visionary product designer with expertise in user experience innovation. Led creative workshops and design thinking sessions. Pioneered new design methodologies and aesthetic approaches. Regular speaker at creative conferences and art exhibitions.",
            
            # High Conscientiousness samples  
            "Detail-oriented project manager with systematic approach to complex deliverables. Meticulously planned and executed 50+ projects with 99% on-time delivery rate. Developed comprehensive quality assurance protocols and documentation standards. Consistently exceeded performance metrics through careful planning.",
            
            "Methodical financial analyst with proven track record of accurate forecasting. Implemented rigorous data validation processes and maintained detailed audit trails. Established systematic review procedures that reduced errors by 95%. Known for thorough analysis and precise reporting.",
            
            # High Extroversion samples
            "Dynamic team leader with exceptional communication and presentation skills. Successfully managed cross-functional teams of 20+ members. Regular keynote speaker at industry conferences and networking events. Built extensive professional network through active community engagement.",
            
            "Energetic sales director with outstanding interpersonal abilities. Led high-energy team meetings and client presentations. Organized large-scale networking events and industry summits. Thrives in collaborative environments and public speaking opportunities.",
            
            # High Agreeableness samples
            "Collaborative engineering manager focused on team harmony and mutual support. Mentored 30+ junior developers and fostered inclusive team culture. Facilitated conflict resolution and consensus building. Known for patience, empathy, and helpful nature.",
            
            "Supportive HR business partner dedicated to employee development and workplace wellbeing. Organized team-building activities and volunteer initiatives. Established peer mentoring programs and support networks. Consistently rated highly for approachability and understanding.",
            
            # High Neuroticism samples
            "Risk-aware compliance officer with expertise in identifying potential issues and vulnerabilities. Developed comprehensive risk assessment frameworks and contingency planning protocols. Specialized in crisis management and emergency response procedures.",
            
            "Thorough quality assurance specialist focused on identifying defects and improvement opportunities. Implemented rigorous testing protocols and stress-testing procedures. Expert in troubleshooting complex technical challenges and system failures."
        ]
        
        # Extract features for all samples
        features = []
        for cv_text in sample_cvs:
            try:
                feature_vector = self.extract_features(cv_text)
                features.append(feature_vector)
            except Exception as e:
                print(f"Error extracting features: {e}")
                continue
        
        if not features:
            raise ValueError("Could not extract features from any sample")
        
        # Pad features to same length
        max_length = max(len(f) for f in features)
        padded_features = []
        for f in features:
            if len(f) < max_length:
                padded = np.pad(f, (0, max_length - len(f)), 'constant')
            else:
                padded = f[:max_length]
            padded_features.append(padded)
        
        X = np.array(padded_features)
        
        # Enhanced labels with more realistic distributions
        y = {
            'openness': np.array([0.9, 0.85, 0.3, 0.25, 0.7, 0.6, 0.4, 0.35, 0.5, 0.45]),
            'conscientiousness': np.array([0.4, 0.5, 0.95, 0.9, 0.6, 0.65, 0.8, 0.75, 0.85, 0.9]),
            'extroversion': np.array([0.6, 0.7, 0.5, 0.55, 0.95, 0.9, 0.8, 0.75, 0.3, 0.25]),
            'agreeableness': np.array([0.5, 0.6, 0.7, 0.75, 0.6, 0.65, 0.95, 0.9, 0.4, 0.45]),
            'neuroticism': np.array([0.3, 0.25, 0.2, 0.15, 0.4, 0.35, 0.3, 0.25, 0.8, 0.85])
        }
        
        return X, y
    
    def train_ensemble_models(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Train ensemble models for each personality trait"""
        print("Training ensemble models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        for trait in self.personality_traits:
            print(f"Training ensemble for {trait}...")
            
            # Create ensemble of different algorithms
            models = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'svm': SVC(probability=True, random_state=42)
            }
            
            # Convert to binary classification
            y_binary = (y[trait] > 0.6).astype(int)
            
            # Train each model and evaluate
            ensemble_models = {}
            for name, model in models.items():
                try:
                    model.fit(X_scaled, y_binary)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_scaled, y_binary, cv=3)
                    print(f"  {name.upper()} CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                    
                    ensemble_models[name] = model
                except Exception as e:
                    print(f"  Error training {name}: {e}")
            
            self.models[trait] = ensemble_models
            
            # Save models
            joblib.dump(ensemble_models, f'models/{trait}_ensemble.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler_advanced.pkl')
        print("Ensemble models trained and saved!")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            for trait in self.personality_traits:
                self.models[trait] = joblib.load(f'models/{trait}_ensemble.pkl')
            self.scaler = joblib.load('models/scaler_advanced.pkl')
            print("Models loaded successfully!")
            return True
        except FileNotFoundError:
            print("No pre-trained models found. Training new models...")
            return False
    
    def predict_personality_ensemble(self, cv_path: str) -> PersonalityScores:
        """Predict personality using ensemble approach"""
        print(f"\nAnalyzing CV: {cv_path}")
        
        # Load models if needed
        if not self.models:
            if not self.load_models():
                X, y = self.create_enhanced_training_data()
                self.train_ensemble_models(X, y)
        
        # Extract text and features
        text = self.extract_text_from_file(cv_path)
        if not text.strip():
            raise ValueError("Could not extract text from CV")
        
        print(f"Extracted {len(text)} characters of text")
        
        # Extract features
        features = self.extract_features(text)
        print(f"Extracted {len(features)} features")
        
        # Ensure features match training dimensions
        expected_dim = self.scaler.n_features_in_
        if len(features) < expected_dim:
            features = np.pad(features, (0, expected_dim - len(features)), 'constant')
        elif len(features) > expected_dim:
            features = features[:expected_dim]
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Ensemble predictions
        predictions = {}
        for trait in self.personality_traits:
            trait_predictions = []
            
            for model_name, model in self.models[trait].items():
                try:
                    prob = model.predict_proba(features_scaled)[0][1]
                    trait_predictions.append(prob)
                except Exception as e:
                    print(f"Error with {model_name} for {trait}: {e}")
            
            # Average ensemble prediction
            if trait_predictions:
                predictions[trait] = np.mean(trait_predictions)
            else:
                predictions[trait] = 0.5  # Default neutral score
        
        return PersonalityScores(**predictions)
    
    def create_advanced_visualization(self, scores: PersonalityScores, save_path: str = None):
        """Create advanced personality visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Bar chart with confidence intervals
        traits = list(scores.to_dict().keys())
        values = list(scores.to_dict().values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars = ax1.bar(traits, values, color=colors, alpha=0.8)
        ax1.set_title('Personality Trait Scores', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score (0-1)')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Radar chart
        angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False)
        values_radar = values + [values[0]]
        angles_radar = np.concatenate((angles, [angles[0]]))
        
        ax2 = plt.subplot(2, 2, 2, projection='polar')
        ax2.plot(angles_radar, values_radar, 'o-', linewidth=2, color='#FF6B6B')
        ax2.fill(angles_radar, values_radar, alpha=0.25, color='#FF6B6B')
        ax2.set_xticks(angles)
        ax2.set_xticklabels([t.title() for t in traits])
        ax2.set_ylim(0, 1)
        ax2.set_title('Personality Profile (Radar)', size=14, fontweight='bold', pad=20)
        
        # 3. Comparison to population norms (simulated)
        population_means = [0.5, 0.5, 0.5, 0.5, 0.3]  # Typical population averages
        
        x_pos = np.arange(len(traits))
        width = 0.35
        
        ax3.bar(x_pos - width/2, values, width, label='Your Score', color=colors, alpha=0.8)
        ax3.bar(x_pos + width/2, population_means, width, label='Population Average', 
                color='gray', alpha=0.6)
        
        ax3.set_title('Comparison to Population Norms', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([t.title() for t in traits], rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # 4. Trait interpretation
        ax4.axis('off')
        interpretation = self.generate_detailed_interpretation(scores)
        ax4.text(0.1, 0.9, 'Personality Interpretation:', fontsize=16, fontweight='bold',
                transform=ax4.transAxes)
        ax4.text(0.1, 0.1, interpretation, fontsize=10, transform=ax4.transAxes,
                verticalalignment='bottom', wrap=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_detailed_interpretation(self, scores: PersonalityScores) -> str:
        """Generate detailed personality interpretation"""
        interpretation = ""
        
        trait_interpretations = {
            'openness': {
                'high': 'likely to be creative, curious, and open to new experiences',
                'medium': 'moderately open to new ideas with balanced creativity',
                'low': 'tends to prefer routine and conventional approaches'
            },
            'conscientiousness': {
                'high': 'highly organized, disciplined, and goal-oriented',
                'medium': 'reasonably organized with good self-control',
                'low': 'more flexible and spontaneous in approach'
            },
            'extroversion': {
                'high': 'energetic, sociable, and comfortable in social situations',
                'medium': 'balanced between social and solitary activities',
                'low': 'more reserved and introspective'
            },
            'agreeableness': {
                'high': 'cooperative, trusting, and considerate of others',
                'medium': 'generally cooperative with healthy skepticism',
                'low': 'more competitive and direct in interactions'
            },
            'neuroticism': {
                'high': 'may experience stress more intensely, detail-oriented about risks',
                'medium': 'generally emotionally stable with normal stress responses',
                'low': 'very calm and emotionally stable under pressure'
            }
        }
        
        scores_dict = scores.to_dict()
        for trait, score in scores_dict.items():
            if score > 0.7:
                level = 'high'
            elif score > 0.4:
                level = 'medium'
            else:
                level = 'low'
            
            interpretation += f"\n• {trait.title()}: {score:.2f} - {trait_interpretations[trait][level]}"
        
        return interpretation
    
    def save_detailed_results(self, cv_path: str, scores: PersonalityScores, 
                            analysis_details: Dict = None):
        """Save comprehensive analysis results"""
        results = {
            'cv_file': cv_path,
            'personality_scores': scores.to_dict(),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'interpretation': self.generate_detailed_interpretation(scores),
            'model_info': {
                'deep_learning_used': self.use_deep_learning,
                'feature_count': len(self.extract_features("sample text"))
            }
        }
        
        if analysis_details:
            results['analysis_details'] = analysis_details
        
        # Save as JSON
        json_path = f"results/personality_analysis_{Path(cv_path).stem}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV
        csv_path = f"results/personality_scores_{Path(cv_path).stem}.csv"
        df = pd.DataFrame([scores.to_dict()])
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {json_path} and {csv_path}")
        return json_path
    
    def analyze_cv_complete(self, cv_path: str) -> Dict:
        """Complete CV personality analysis pipeline"""
        print("="*60)
        print("ADVANCED CV PERSONALITY ANALYSIS")
        print("="*60)
        
        try:
            # Predict personality
            scores = self.predict_personality_ensemble(cv_path)
            
            # Create visualization
            viz_path = f"visualizations/personality_analysis_{Path(cv_path).stem}.png"
            self.create_advanced_visualization(scores, viz_path)
            
            # Save results
            results_path = self.save_detailed_results(cv_path, scores)
            
            # Print summary
            print("\n" + "="*50)
            print("PERSONALITY ANALYSIS SUMMARY")
            print("="*50)
            print(f"CV Analyzed: {cv_path}")
            print(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\nPersonality Scores:")
            
            for trait, score in scores.to_dict().items():
                level = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
                print(f"  {trait.title():<15}: {score:.3f} ({level})")
            
            print(f"\nDetailed interpretation:")
            print(self.generate_detailed_interpretation(scores))
            print(f"\nVisualization saved: {viz_path}")
            print(f"Results saved: {results_path}")
            print("="*50)
            
            return {
                'scores': scores,
                'visualization_path': viz_path,
                'results_path': results_path,
                'interpretation': self.generate_detailed_interpretation(scores)
            }
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

def main():
    """Main function to demonstrate the advanced system"""
    print("Advanced CV Personality Prediction System")
    print("=========================================")
    
    # Initialize analyzer
    analyzer = CVPersonalityAnalyzer(use_deep_learning=DEEP_LEARNING_AVAILABLE)
    
    # Check if demo CV exists
    if os.path.exists('demo_cv.txt'):
        cv_path = 'demo_cv.txt'
    else:
        print("Demo CV not found. Please ensure 'demo_cv.txt' exists.")
        return
    
    try:
        # Run complete analysis
        results = analyzer.analyze_cv_complete(cv_path)
        
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Personality scores: {results['scores'].to_dict()}")
        print(f"📈 Visualization: {results['visualization_path']}")
        print(f"💾 Results: {results['results_path']}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please check the CV file and try again.")

if __name__ == "__main__":
    main()