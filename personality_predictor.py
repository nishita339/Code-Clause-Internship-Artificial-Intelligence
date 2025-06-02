"""
CV Personality Prediction System
Analyzes CVs/resumes to predict Big Five personality traits using NLP and ML
"""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# PDF and document processing
import PyPDF2
from docx import Document
import pytesseract
from PIL import Image
import cv2

# NLP and ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment analysis
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class CVPersonalityPredictor:
    """Main class for CV personality prediction system"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.scaler = StandardScaler()
        self.models = {}
        self.personality_traits = [
            'openness', 'conscientiousness', 'extroversion', 
            'agreeableness', 'neuroticism'
        ]
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
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
        """Extract text from DOCX file"""
        try:
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            # Read image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply image preprocessing
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            gray = cv2.medianBlur(gray, 3)
            
            # Extract text using tesseract
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
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return self.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return processed_tokens
    
    def extract_cv_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from CV text"""
        sections = {
            'education': '',
            'experience': '',
            'skills': '',
            'projects': '',
            'certifications': '',
            'other': ''
        }
        
        # Define section keywords
        education_keywords = ['education', 'academic', 'degree', 'university', 'college', 'school']
        experience_keywords = ['experience', 'work', 'employment', 'job', 'position', 'role']
        skills_keywords = ['skills', 'technical', 'programming', 'languages', 'tools']
        project_keywords = ['projects', 'portfolio', 'work samples', 'achievements']
        cert_keywords = ['certifications', 'certificates', 'awards', 'honors']
        
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            if any(keyword in sentence_lower for keyword in education_keywords):
                sections['education'] += sentence + ' '
            elif any(keyword in sentence_lower for keyword in experience_keywords):
                sections['experience'] += sentence + ' '
            elif any(keyword in sentence_lower for keyword in skills_keywords):
                sections['skills'] += sentence + ' '
            elif any(keyword in sentence_lower for keyword in project_keywords):
                sections['projects'] += sentence + ' '
            elif any(keyword in sentence_lower for keyword in cert_keywords):
                sections['certifications'] += sentence + ' '
            else:
                sections['other'] += sentence + ' '
        
        return sections
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from text"""
        features = {}
        
        # Basic text statistics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # POS tag features
        pos_tags = pos_tag(words)
        pos_counts = {}
        for word, pos in pos_tags:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        total_words = len(words)
        features['noun_ratio'] = sum(count for pos, count in pos_counts.items() if pos.startswith('NN')) / total_words if total_words else 0
        features['verb_ratio'] = sum(count for pos, count in pos_counts.items() if pos.startswith('VB')) / total_words if total_words else 0
        features['adjective_ratio'] = sum(count for pos, count in pos_counts.items() if pos.startswith('JJ')) / total_words if total_words else 0
        
        # Sentiment features
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        # Assertiveness indicators
        assertive_words = ['achieve', 'lead', 'manage', 'direct', 'create', 'develop', 'improve', 'increase']
        collaborative_words = ['collaborate', 'team', 'support', 'assist', 'help', 'cooperate', 'work with']
        
        features['assertive_word_count'] = sum(1 for word in words if word.lower() in assertive_words)
        features['collaborative_word_count'] = sum(1 for word in words if word.lower() in collaborative_words)
        
        return features
    
    def extract_features(self, text: str) -> np.ndarray:
        """Extract all features from text"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract sections
        sections = self.extract_cv_sections(text)
        
        # Extract linguistic features
        linguistic_features = self.extract_linguistic_features(cleaned_text)
        
        # Combine all text for TF-IDF
        combined_text = ' '.join([cleaned_text] + list(sections.values()))
        
        # Create feature vector
        feature_vector = list(linguistic_features.values())
        
        return np.array(feature_vector)
    
    def create_training_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Create synthetic training data for demonstration"""
        print("Creating synthetic training data...")
        
        # Sample CV texts with different personality patterns
        sample_cvs = [
            # High openness
            "Innovative software engineer with expertise in cutting-edge technologies. Passionate about exploring new frameworks and creative problem-solving. Led development of novel AI solutions.",
            
            # High conscientiousness
            "Detail-oriented project manager with proven track record of delivering projects on time and within budget. Meticulous planning and systematic approach to problem-solving.",
            
            # High extroversion
            "Dynamic team leader with excellent communication skills. Enjoy collaborating with diverse teams and presenting to large audiences. Active in professional networking events.",
            
            # High agreeableness
            "Collaborative developer who values team harmony. Supportive mentor to junior colleagues. Focus on building consensus and maintaining positive relationships.",
            
            # High neuroticism
            "Experienced analyst who pays careful attention to potential risks and challenges. Thorough in identifying issues and developing contingency plans."
        ]
        
        # Create features
        features = []
        for cv_text in sample_cvs:
            feature_vector = self.extract_features(cv_text)
            features.append(feature_vector)
        
        X = np.array(features)
        
        # Create synthetic labels (0-1 scale)
        y = {
            'openness': np.array([0.9, 0.3, 0.6, 0.5, 0.4]),
            'conscientiousness': np.array([0.4, 0.9, 0.7, 0.6, 0.8]),
            'extroversion': np.array([0.5, 0.4, 0.9, 0.7, 0.3]),
            'agreeableness': np.array([0.6, 0.5, 0.7, 0.9, 0.4]),
            'neuroticism': np.array([0.3, 0.2, 0.4, 0.3, 0.8])
        }
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Train personality prediction models"""
        print("Training personality prediction models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train a model for each personality trait
        for trait in self.personality_traits:
            print(f"Training model for {trait}...")
            
            # Convert to binary classification (high vs low)
            y_binary = (y[trait] > 0.5).astype(int)
            
            # Train Random Forest model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y_binary)
            
            self.models[trait] = model
            
            # Save model
            joblib.dump(model, f'models/{trait}_model.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("Models trained and saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            for trait in self.personality_traits:
                self.models[trait] = joblib.load(f'models/{trait}_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            print("Models loaded successfully!")
        except FileNotFoundError:
            print("No pre-trained models found. Training new models...")
            X, y = self.create_training_data()
            self.train_models(X, y)
    
    def predict_personality(self, cv_path: str) -> Dict[str, float]:
        """Predict personality traits from CV"""
        print(f"Analyzing CV: {cv_path}")
        
        # Extract text from CV
        text = self.extract_text_from_file(cv_path)
        if not text.strip():
            raise ValueError("Could not extract text from CV")
        
        # Extract features
        features = self.extract_features(text)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict personality traits
        predictions = {}
        for trait in self.personality_traits:
            # Get probability of high trait
            prob = self.models[trait].predict_proba(features_scaled)[0][1]
            predictions[trait] = prob
        
        return predictions
    
    def visualize_personality(self, predictions: Dict[str, float], save_path: str = None):
        """Visualize personality predictions"""
        traits = list(predictions.keys())
        scores = list(predictions.values())
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(traits, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        
        plt.title('Big Five Personality Traits Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Personality Traits', fontsize=12)
        plt.ylabel('Score (0-1)', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False)
        scores_radar = scores + [scores[0]]  # Complete the circle
        angles_radar = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles_radar, scores_radar, 'o-', linewidth=2, color='#FF6B6B')
        ax.fill(angles_radar, scores_radar, alpha=0.25, color='#FF6B6B')
        ax.set_xticks(angles)
        ax.set_xticklabels(traits)
        ax.set_ylim(0, 1)
        ax.set_title('Personality Profile (Radar Chart)', size=16, fontweight='bold', pad=20)
        
        if save_path:
            radar_path = save_path.replace('.png', '_radar.png')
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_personality_summary(self, predictions: Dict[str, float]) -> str:
        """Generate personality summary text"""
        summary = "Personality Analysis Summary:\n\n"
        
        # Categorize scores
        high_traits = [trait for trait, score in predictions.items() if score > 0.7]
        moderate_traits = [trait for trait, score in predictions.items() if 0.4 <= score <= 0.7]
        low_traits = [trait for trait, score in predictions.items() if score < 0.4]
        
        if high_traits:
            summary += f"High traits: {', '.join(high_traits).title()}\n"
        if moderate_traits:
            summary += f"Moderate traits: {', '.join(moderate_traits).title()}\n"
        if low_traits:
            summary += f"Lower traits: {', '.join(low_traits).title()}\n"
        
        summary += "\nDetailed Analysis:\n"
        
        trait_descriptions = {
            'openness': 'creativity, curiosity, and openness to new experiences',
            'conscientiousness': 'organization, discipline, and goal-oriented behavior',
            'extroversion': 'sociability, assertiveness, and energy in social situations',
            'agreeableness': 'cooperation, trust, and concern for others',
            'neuroticism': 'emotional stability and stress management'
        }
        
        for trait, score in predictions.items():
            level = 'high' if score > 0.7 else 'moderate' if score > 0.4 else 'low'
            summary += f"- {trait.title()}: {score:.2f} ({level}) - indicates {level} {trait_descriptions[trait]}\n"
        
        return summary
    
    def save_results(self, cv_path: str, predictions: Dict[str, float], summary: str):
        """Save prediction results"""
        results = {
            'cv_file': cv_path,
            'personality_scores': predictions,
            'summary': summary,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save as JSON
        json_path = f"results/personality_prediction_{Path(cv_path).stem}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV
        csv_path = f"results/personality_prediction_{Path(cv_path).stem}.csv"
        df = pd.DataFrame([predictions])
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {json_path} and {csv_path}")
    
    def analyze_cv(self, cv_path: str) -> Dict[str, any]:
        """Complete CV analysis pipeline"""
        print("Starting CV personality analysis...")
        
        # Load models if not already loaded
        if not self.models:
            self.load_models()
        
        # Predict personality
        predictions = self.predict_personality(cv_path)
        
        # Generate summary
        summary = self.generate_personality_summary(predictions)
        
        # Visualize results
        viz_path = f"results/personality_chart_{Path(cv_path).stem}.png"
        self.visualize_personality(predictions, viz_path)
        
        # Save results
        self.save_results(cv_path, predictions, summary)
        
        # Print summary
        print("\n" + "="*50)
        print(summary)
        print("="*50)
        
        return {
            'predictions': predictions,
            'summary': summary,
            'visualization_path': viz_path
        }

def main():
    """Main function to demonstrate the system"""
    predictor = CVPersonalityPredictor()
    
    # Create a sample CV for demonstration
    sample_cv_text = """
    John Smith
    Senior Software Engineer
    
    EDUCATION
    Bachelor of Computer Science - University of Technology (2015-2019)
    
    EXPERIENCE
    Senior Software Engineer - Tech Corp (2021-Present)
    - Lead development team of 5 engineers
    - Designed and implemented innovative microservices architecture
    - Collaborated with cross-functional teams to deliver high-quality solutions
    - Mentored junior developers and conducted code reviews
    
    Software Engineer - StartupXYZ (2019-2021)
    - Developed full-stack web applications using React and Node.js
    - Participated in agile development processes
    - Contributed to open-source projects
    
    SKILLS
    Programming Languages: Python, JavaScript, Java, Go
    Frameworks: React, Node.js, Django, Flask
    Databases: PostgreSQL, MongoDB, Redis
    Cloud: AWS, Docker, Kubernetes
    
    PROJECTS
    - Personal Finance Tracker: Built a comprehensive web application for expense tracking
    - AI Chatbot: Developed an intelligent customer service bot using NLP
    - Open Source Contributions: Active contributor to several popular libraries
    
    CERTIFICATIONS
    - AWS Certified Solutions Architect
    - Certified ScrumMaster
    """
    
    # Save sample CV
    with open('sample_cv.txt', 'w') as f:
        f.write(sample_cv_text)
    
    # Analyze the sample CV
    try:
        results = predictor.analyze_cv('sample_cv.txt')
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()