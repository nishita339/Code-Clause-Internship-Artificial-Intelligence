#!/usr/bin/env python3
"""
CV Personality Prediction System - Demo Version
Analyzes CVs and predicts Big Five personality traits using NLP and ML
"""

import os
import re
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Basic text processing
import string
from collections import Counter

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

# Visualization
import matplotlib.pyplot as plt

class SimplePersonalityPredictor:
    """Simplified CV personality prediction system"""
    
    def __init__(self):
        self.personality_traits = ['openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism']
        self.models = {}
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 2))
        
        # Personality keyword dictionaries
        self.trait_keywords = {
            'openness': [
                'creative', 'innovative', 'novel', 'artistic', 'imaginative', 'curious',
                'explore', 'experiment', 'original', 'unique', 'diverse', 'research',
                'design', 'develop', 'invent', 'pioneer', 'breakthrough', 'cutting-edge'
            ],
            'conscientiousness': [
                'organized', 'systematic', 'detailed', 'planned', 'scheduled', 'thorough',
                'disciplined', 'reliable', 'careful', 'precise', 'methodical', 'consistent',
                'responsible', 'diligent', 'punctual', 'quality', 'accuracy', 'compliance'
            ],
            'extroversion': [
                'team', 'social', 'communicate', 'present', 'lead', 'network', 'public',
                'collaborate', 'energetic', 'outgoing', 'confident', 'speaking', 'meeting',
                'group', 'leadership', 'manage', 'coordinate', 'facilitate', 'engage'
            ],
            'agreeableness': [
                'cooperative', 'supportive', 'helpful', 'friendly', 'collaborative', 'understanding',
                'empathetic', 'considerate', 'respectful', 'diplomatic', 'patient', 'mentor',
                'assist', 'volunteer', 'support', 'help', 'work with', 'partner'
            ],
            'neuroticism': [
                'stress', 'pressure', 'challenging', 'difficult', 'problem', 'issue',
                'risk', 'uncertainty', 'deadline', 'crisis', 'conflict', 'demanding',
                'tight', 'urgent', 'critical', 'emergency', 'troubleshoot', 'resolve'
            ]
        }
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from file (supports .txt for demo)"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep spaces
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract personality-relevant features from text"""
        features = {}
        
        # Preprocess text
        clean_text = self.preprocess_text(text)
        words = clean_text.split()
        
        # Basic statistics
        features['word_count'] = len(words)
        features['unique_words'] = len(set(words))
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['vocab_diversity'] = len(set(words)) / len(words) if words else 0
        
        # Personality keyword features
        for trait, keywords in self.trait_keywords.items():
            count = sum(1 for word in words if word in keywords)
            features[f'{trait}_keywords'] = count / len(words) if words else 0
        
        # Action verb patterns
        leadership_verbs = ['led', 'managed', 'directed', 'supervised', 'coordinated']
        innovation_verbs = ['created', 'developed', 'designed', 'built', 'implemented']
        collaboration_verbs = ['collaborated', 'worked', 'partnered', 'assisted', 'supported']
        
        features['leadership_verbs'] = sum(1 for word in words if word in leadership_verbs) / len(words) if words else 0
        features['innovation_verbs'] = sum(1 for word in words if word in innovation_verbs) / len(words) if words else 0
        features['collaboration_verbs'] = sum(1 for word in words if word in collaboration_verbs) / len(words) if words else 0
        
        # Technical vs soft skills
        technical_words = ['python', 'java', 'sql', 'programming', 'software', 'technical', 'algorithm', 'data']
        soft_words = ['communication', 'teamwork', 'problem', 'solving', 'interpersonal', 'presentation']
        
        features['technical_ratio'] = sum(1 for word in words if word in technical_words) / len(words) if words else 0
        features['soft_skills_ratio'] = sum(1 for word in words if word in soft_words) / len(words) if words else 0
        
        return features
    
    def create_training_data(self):
        """Create synthetic training data for demonstration"""
        print("Creating training dataset...")
        
        # Sample CV profiles with different personality patterns
        cv_samples = [
            # High Openness
            "Creative software engineer passionate about innovative solutions. Developed novel machine learning algorithms and artistic coding projects. Research experience in AI and computer vision. Love exploring new technologies and experimental approaches.",
            
            "Innovative product designer with breakthrough creative methodologies. Pioneer in user experience research and design thinking. Published papers on creative problem-solving. Enjoy experimenting with new design tools and artistic projects.",
            
            # High Conscientiousness  
            "Detail-oriented project manager with systematic approach to complex deliverables. Meticulously planned and executed projects with consistent quality standards. Developed comprehensive documentation and quality assurance protocols.",
            
            "Methodical software engineer focused on code quality and thorough testing. Implemented rigorous development processes and maintained detailed project documentation. Known for reliable delivery and attention to detail.",
            
            # High Extroversion
            "Dynamic team leader with exceptional communication and presentation skills. Managed cross-functional teams and led public speaking engagements. Active in professional networking and community events.",
            
            "Energetic sales manager with outstanding interpersonal abilities. Organized team meetings and client presentations. Built extensive professional network through active engagement and public speaking.",
            
            # High Agreeableness  
            "Collaborative engineering manager focused on team harmony and support. Mentored junior developers and facilitated inclusive team culture. Known for patient guidance and helpful nature.",
            
            "Supportive project coordinator dedicated to team collaboration. Organized team-building activities and volunteer initiatives. Consistently praised for empathy and understanding in team interactions.",
            
            # Lower Neuroticism (Emotional Stability)
            "Calm and composed operations manager effective under pressure. Successfully handled crisis situations and tight deadlines. Known for maintaining stability during challenging projects.",
            
            "Steady technical lead with excellent stress management skills. Thrived in high-pressure environments and urgent project deliveries. Maintained quality standards during demanding periods."
        ]
        
        # Extract features from samples
        feature_dicts = [self.extract_features(cv) for cv in cv_samples]
        
        # Convert to feature matrix
        feature_names = list(feature_dicts[0].keys())
        X = np.array([[fd[name] for name in feature_names] for fd in feature_dicts])
        
        # Create personality labels (0-1 scale)
        y = {
            'openness': np.array([0.9, 0.85, 0.4, 0.3, 0.6, 0.5, 0.5, 0.4, 0.6, 0.5]),
            'conscientiousness': np.array([0.5, 0.6, 0.9, 0.85, 0.7, 0.6, 0.8, 0.7, 0.8, 0.75]),
            'extroversion': np.array([0.6, 0.7, 0.5, 0.4, 0.9, 0.85, 0.7, 0.8, 0.6, 0.5]),
            'agreeableness': np.array([0.6, 0.5, 0.6, 0.5, 0.7, 0.6, 0.9, 0.85, 0.7, 0.6]),
            'neuroticism': np.array([0.4, 0.3, 0.3, 0.2, 0.4, 0.3, 0.3, 0.2, 0.1, 0.15])
        }
        
        return X, y, feature_names
    
    def train_models(self, X: np.ndarray, y: Dict[str, np.ndarray], feature_names: List[str]):
        """Train personality prediction models"""
        print("Training personality models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models for each trait
        for trait in self.personality_traits:
            print(f"  Training {trait} model...")
            
            # Convert to binary classification (high vs low)
            y_binary = (y[trait] > 0.6).astype(int)
            
            # Train Random Forest
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            model.fit(X_scaled, y_binary)
            
            # Evaluate with cross-validation
            cv_scores = cross_val_score(model, X_scaled, y_binary, cv=3)
            print(f"    CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            self.models[trait] = model
            
            # Save model
            joblib.dump(model, f'models/{trait}_model.pkl')
        
        # Save scaler and feature names
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(feature_names, 'models/feature_names.pkl')
        
        print("Models trained successfully!")
    
    def load_models(self) -> bool:
        """Load pre-trained models"""
        try:
            for trait in self.personality_traits:
                self.models[trait] = joblib.load(f'models/{trait}_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            print("Models loaded successfully!")
            return True
        except FileNotFoundError:
            print("No pre-trained models found.")
            return False
    
    def predict_personality(self, cv_path: str) -> Dict[str, float]:
        """Predict personality traits from CV"""
        print(f"\nAnalyzing CV: {cv_path}")
        
        # Load models if needed
        if not self.models:
            if not self.load_models():
                X, y, feature_names = self.create_training_data()
                self.feature_names = feature_names
                self.train_models(X, y, feature_names)
        
        # Extract text
        text = self.extract_text_from_file(cv_path)
        if not text.strip():
            raise ValueError("Could not extract text from CV")
        
        print(f"Extracted {len(text)} characters of text")
        
        # Extract features
        feature_dict = self.extract_features(text)
        features = np.array([feature_dict[name] for name in self.feature_names])
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict personality traits
        predictions = {}
        for trait in self.personality_traits:
            proba = self.models[trait].predict_proba(features_scaled)[0]
            # Handle both binary and multi-class predictions
            if len(proba) > 1:
                prob = proba[1]  # Probability of positive class
            else:
                prob = proba[0]  # Single probability
            predictions[trait] = prob
        
        return predictions
    
    def visualize_results(self, predictions: Dict[str, float], cv_name: str):
        """Create visualization of personality predictions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        traits = list(predictions.keys())
        scores = list(predictions.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars = ax1.bar(traits, scores, color=colors, alpha=0.8)
        ax1.set_title(f'Personality Analysis: {cv_name}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Probability Score (0-1)')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.tick_params(axis='x', rotation=45)
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False)
        scores_radar = scores + [scores[0]]
        angles_radar = np.concatenate((angles, [angles[0]]))
        
        ax2 = plt.subplot(1, 2, 2, projection='polar')
        ax2.plot(angles_radar, scores_radar, 'o-', linewidth=2, color='#FF6B6B')
        ax2.fill(angles_radar, scores_radar, alpha=0.25, color='#FF6B6B')
        ax2.set_xticks(angles)
        ax2.set_xticklabels([t.title() for t in traits])
        ax2.set_ylim(0, 1)
        ax2.set_title('Personality Profile', size=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"results/personality_analysis_{cv_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {plot_path}")
        plt.show()
        
        return plot_path
    
    def generate_summary(self, predictions: Dict[str, float]) -> str:
        """Generate personality summary"""
        summary = "PERSONALITY ANALYSIS SUMMARY\n"
        summary += "=" * 40 + "\n\n"
        
        # Categorize traits
        high_traits = [trait for trait, score in predictions.items() if score > 0.7]
        moderate_traits = [trait for trait, score in predictions.items() if 0.4 <= score <= 0.7]
        low_traits = [trait for trait, score in predictions.items() if score < 0.4]
        
        summary += "Personality Profile:\n"
        for trait, score in predictions.items():
            level = "High" if score > 0.7 else "Moderate" if score > 0.4 else "Low"
            summary += f"  {trait.title():<15}: {score:.3f} ({level})\n"
        
        summary += "\nKey Characteristics:\n"
        
        trait_descriptions = {
            'openness': 'creativity and openness to new experiences',
            'conscientiousness': 'organization and attention to detail',
            'extroversion': 'social energy and communication skills',
            'agreeableness': 'cooperation and concern for others',
            'neuroticism': 'emotional sensitivity and stress awareness'
        }
        
        if high_traits:
            summary += f"• Strong in: {', '.join(high_traits)}\n"
            for trait in high_traits:
                summary += f"  - Shows high {trait_descriptions[trait]}\n"
        
        if low_traits:
            summary += f"• More reserved in: {', '.join(low_traits)}\n"
        
        return summary
    
    def save_results(self, cv_path: str, predictions: Dict[str, float], summary: str):
        """Save analysis results"""
        results = {
            'cv_file': cv_path,
            'analysis_date': pd.Timestamp.now().isoformat(),
            'personality_scores': predictions,
            'summary': summary
        }
        
        # Save JSON
        json_path = f"results/analysis_{Path(cv_path).stem}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV
        csv_path = f"results/scores_{Path(cv_path).stem}.csv"
        df = pd.DataFrame([predictions])
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved: {json_path}")
        return json_path
    
    def analyze_cv(self, cv_path: str) -> Dict:
        """Complete CV analysis"""
        print("\n" + "=" * 60)
        print("CV PERSONALITY ANALYSIS SYSTEM")
        print("=" * 60)
        
        try:
            # Predict personality
            predictions = self.predict_personality(cv_path)
            
            # Generate summary
            summary = self.generate_summary(predictions)
            
            # Create visualization
            cv_name = Path(cv_path).stem
            plot_path = self.visualize_results(predictions, cv_name)
            
            # Save results
            results_path = self.save_results(cv_path, predictions, summary)
            
            # Print summary
            print("\n" + summary)
            print("=" * 60)
            
            return {
                'predictions': predictions,
                'summary': summary,
                'plot_path': plot_path,
                'results_path': results_path
            }
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

def main():
    """Demonstrate the personality prediction system"""
    print("CV Personality Prediction System - Demo")
    print("======================================")
    
    # Initialize predictor
    predictor = SimplePersonalityPredictor()
    
    # Check for demo CV
    cv_file = 'demo_cv.txt'
    if not os.path.exists(cv_file):
        print(f"Demo CV file '{cv_file}' not found.")
        return
    
    try:
        # Run analysis
        results = predictor.analyze_cv(cv_file)
        
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Results saved to: {results['results_path']}")
        print(f"📈 Visualization: {results['plot_path']}")
        
        # Show top traits
        predictions = results['predictions']
        top_trait = max(predictions.items(), key=lambda x: x[1])
        print(f"🎯 Strongest trait: {top_trait[0].title()} ({top_trait[1]:.3f})")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()