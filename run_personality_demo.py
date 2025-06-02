#!/usr/bin/env python3
"""
Quick CV Personality Prediction Demo
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def extract_personality_features(text):
    """Extract key personality indicators from CV text"""
    text = text.lower()
    words = text.split()
    
    # Personality keyword indicators
    openness_words = ['creative', 'innovative', 'research', 'design', 'novel', 'experimental']
    conscientiousness_words = ['organized', 'detailed', 'systematic', 'quality', 'thorough', 'planning']
    extroversion_words = ['team', 'leadership', 'communication', 'presentation', 'collaborate', 'manage']
    agreeableness_words = ['supportive', 'mentor', 'help', 'assist', 'cooperative', 'volunteer']
    neuroticism_words = ['stress', 'pressure', 'deadline', 'crisis', 'challenging', 'difficult']
    
    features = {
        'openness_score': sum(1 for word in words if word in openness_words) / len(words) if words else 0,
        'conscientiousness_score': sum(1 for word in words if word in conscientiousness_words) / len(words) if words else 0,
        'extroversion_score': sum(1 for word in words if word in extroversion_words) / len(words) if words else 0,
        'agreeableness_score': sum(1 for word in words if word in agreeableness_words) / len(words) if words else 0,
        'neuroticism_score': sum(1 for word in words if word in neuroticism_words) / len(words) if words else 0,
        'word_count': len(words),
        'unique_words': len(set(words)),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0
    }
    
    return features

def predict_personality(cv_text):
    """Predict Big Five personality traits from CV text"""
    
    # Extract features
    features = extract_personality_features(cv_text)
    
    # Simple rule-based predictions with realistic scoring
    predictions = {
        'openness': min(0.9, max(0.1, features['openness_score'] * 20 + 0.4)),
        'conscientiousness': min(0.9, max(0.1, features['conscientiousness_score'] * 15 + 0.5)),
        'extroversion': min(0.9, max(0.1, features['extroversion_score'] * 12 + 0.45)),
        'agreeableness': min(0.9, max(0.1, features['agreeableness_score'] * 18 + 0.5)),
        'neuroticism': min(0.8, max(0.1, features['neuroticism_score'] * 10 + 0.2))
    }
    
    return predictions

def generate_personality_report(predictions):
    """Generate detailed personality analysis report"""
    
    trait_descriptions = {
        'openness': {
            'high': 'highly creative and open to new experiences',
            'medium': 'moderately creative with balanced openness',
            'low': 'prefers conventional approaches'
        },
        'conscientiousness': {
            'high': 'extremely organized and detail-oriented',
            'medium': 'reasonably organized with good planning skills',
            'low': 'more flexible and spontaneous'
        },
        'extroversion': {
            'high': 'very social and energetic in group settings',
            'medium': 'balanced between social and independent work',
            'low': 'prefers quiet, independent work environments'
        },
        'agreeableness': {
            'high': 'highly cooperative and empathetic',
            'medium': 'generally cooperative with healthy boundaries',
            'low': 'direct and competitive in approach'
        },
        'neuroticism': {
            'high': 'detail-aware about potential challenges',
            'medium': 'balanced emotional responses to stress',
            'low': 'very calm under pressure'
        }
    }
    
    report = "CV PERSONALITY ANALYSIS RESULTS\n"
    report += "=" * 50 + "\n\n"
    
    report += "PERSONALITY TRAIT SCORES:\n"
    report += "-" * 25 + "\n"
    
    for trait, score in predictions.items():
        level = 'high' if score > 0.7 else 'medium' if score > 0.4 else 'low'
        percentage = int(score * 100)
        
        report += f"{trait.title():<15}: {score:.3f} ({percentage}%) - {level}\n"
    
    report += "\nDETAILED INTERPRETATION:\n"
    report += "-" * 25 + "\n"
    
    for trait, score in predictions.items():
        level = 'high' if score > 0.7 else 'medium' if score > 0.4 else 'low'
        description = trait_descriptions[trait][level]
        report += f"• {trait.title()}: {description}\n"
    
    # Find strongest and weakest traits
    max_trait = max(predictions.items(), key=lambda x: x[1])
    min_trait = min(predictions.items(), key=lambda x: x[1])
    
    report += f"\nKEY INSIGHTS:\n"
    report += "-" * 15 + "\n"
    report += f"• Strongest trait: {max_trait[0].title()} ({max_trait[1]:.3f})\n"
    report += f"• Areas for development: {min_trait[0].title()} ({min_trait[1]:.3f})\n"
    
    # Professional summary
    high_traits = [trait for trait, score in predictions.items() if score > 0.7]
    if high_traits:
        report += f"• Professional strengths: {', '.join(t.title() for t in high_traits)}\n"
    
    return report

def main():
    """Run the personality prediction demonstration"""
    
    print("CV Personality Prediction System")
    print("================================\n")
    
    # Read the demo CV
    try:
        with open('demo_cv.txt', 'r', encoding='utf-8') as f:
            cv_text = f.read()
    except FileNotFoundError:
        print("Demo CV file not found. Please ensure 'demo_cv.txt' exists.")
        return
    
    print(f"Analyzing CV... ({len(cv_text)} characters)")
    print("-" * 40)
    
    # Predict personality traits
    predictions = predict_personality(cv_text)
    
    # Generate and display report
    report = generate_personality_report(predictions)
    print(report)
    
    # Save results
    results = {
        'personality_scores': predictions,
        'analysis_summary': report,
        'cv_length': len(cv_text)
    }
    
    with open('results/personality_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: results/personality_results.json")
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()