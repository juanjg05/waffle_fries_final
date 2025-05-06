import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List
import pandas as pd
from collections import defaultdict
import os

def load_speaker_contexts(filepath: str) -> Dict:
    """Load speaker contexts from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_speaker_interactions(contexts: Dict) -> pd.DataFrame:
    """Convert speaker contexts to a DataFrame for analysis"""
    data = []
    for speaker_id, context in contexts.items():
        for entry in context['conversation_history']:
            data.append({
                'speaker_id': speaker_id,
                'timestamp': datetime.fromisoformat(entry['timestamp']),
                'confidence': entry['confidence'],
                'speaker_confidence': entry.get('speaker_confidence', 1.0),
                'has_face_direction': entry.get('has_face_direction', False),
                'num_speakers': entry.get('num_speakers', 1),
                'conversation_index': entry['conversation_index']
            })
    return pd.DataFrame(data)

def plot_confidence_distributions(df: pd.DataFrame, output_dir: str):
    """Plot confidence score distributions"""
    plt.figure(figsize=(12, 6))
    
    # Plot diarization confidence
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='confidence', bins=20)
    plt.title('Diarization Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    
    # Plot speaker confidence
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='speaker_confidence', bins=20)
    plt.title('Speaker Identification Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distributions.png'))
    plt.close()

def plot_speaker_interaction_patterns(df: pd.DataFrame, output_dir: str):
    """Plot speaker interaction patterns over time"""
    plt.figure(figsize=(15, 8))
    
    # Group by speaker and count interactions per conversation
    speaker_counts = df.groupby(['conversation_index', 'speaker_id']).size().unstack(fill_value=0)
    
    # Plot stacked bar chart
    speaker_counts.plot(kind='bar', stacked=True)
    plt.title('Speaker Interaction Patterns Across Conversations')
    plt.xlabel('Conversation Index')
    plt.ylabel('Number of Interactions')
    plt.legend(title='Speaker ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speaker_interaction_patterns.png'))
    plt.close()

def plot_face_direction_analysis(df: pd.DataFrame, output_dir: str):
    """Analyze face direction availability and impact"""
    plt.figure(figsize=(12, 6))
    
    # Plot face direction availability
    plt.subplot(1, 2, 1)
    face_direction_counts = df['has_face_direction'].value_counts()
    labels = ['No Face Direction', 'Face Direction Available']
    values = [face_direction_counts.get(False, 0), face_direction_counts.get(True, 0)]
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title('Face Direction Availability')
    
    # Plot confidence comparison
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='has_face_direction', y='confidence')
    plt.title('Confidence Scores with/without Face Direction')
    plt.xlabel('Face Direction Available')
    plt.ylabel('Confidence Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'face_direction_analysis.png'))
    plt.close()

def plot_speaker_stability(df: pd.DataFrame, output_dir: str):
    """Analyze speaker identification stability"""
    plt.figure(figsize=(12, 6))
    
    # Calculate average confidence per speaker
    speaker_confidence = df.groupby('speaker_id')['speaker_confidence'].agg(['mean', 'std']).reset_index()
    
    # Plot confidence stability
    plt.errorbar(range(len(speaker_confidence)), 
                speaker_confidence['mean'],
                yerr=speaker_confidence['std'],
                fmt='o',
                capsize=5)
    plt.xticks(range(len(speaker_confidence)), speaker_confidence['speaker_id'], rotation=45)
    plt.title('Speaker Identification Stability')
    plt.xlabel('Speaker ID')
    plt.ylabel('Average Confidence Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speaker_stability.png'))
    plt.close()

def generate_metrics_report(df: pd.DataFrame, output_dir: str):
    """Generate a comprehensive metrics report"""
    metrics = {
        'Total Interactions': len(df),
        'Unique Speakers': df['speaker_id'].nunique(),
        'Average Confidence': df['confidence'].mean(),
        'Average Speaker Confidence': df['speaker_confidence'].mean(),
        'Face Direction Availability': df['has_face_direction'].mean() * 100,
        'Average Speakers per Conversation': df['num_speakers'].mean(),
        'Confidence with Face Direction': df[df['has_face_direction']]['confidence'].mean(),
        'Confidence without Face Direction': df[~df['has_face_direction']]['confidence'].mean()
    }
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics_report.txt'), 'w') as f:
        f.write("Speaker Context Analysis Metrics\n")
        f.write("=============================\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.2f}\n")

def main():
    # Create output directory
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from the known location
    json_path = "tests/data/speaker_contexts/speaker_contexts.json"
    
    if not os.path.exists(json_path):
        print(f"Error: Could not find speaker contexts file at {json_path}")
        return
    
    print(f"Loading speaker contexts from: {json_path}")
    contexts = load_speaker_contexts(json_path)
    
    df = analyze_speaker_interactions(contexts)
    
    # Generate visualizations
    plot_confidence_distributions(df, output_dir)
    plot_speaker_interaction_patterns(df, output_dir)
    plot_face_direction_analysis(df, output_dir)
    plot_speaker_stability(df, output_dir)
    
    # Generate metrics report
    generate_metrics_report(df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    main() 