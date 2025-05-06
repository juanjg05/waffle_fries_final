import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_simulated_data(n_samples=1000):
    """Generate simulated data for spoken-to model analysis"""
    np.random.seed(42)  # For reproducibility
    
    # Generate timestamps
    base_time = datetime.now()
    timestamps = [base_time + timedelta(seconds=i) for i in range(n_samples)]
    
    # Generate face angles (-90 to 90 degrees)
    face_angles = np.concatenate([
        np.random.normal(0, 15, n_samples//2),  # Mostly facing robot
        np.random.uniform(-90, 90, n_samples//2)  # Random angles
    ])
    
    # Generate spoken-to ground truth
    is_spoken_to = np.zeros(n_samples, dtype=bool)
    is_spoken_to[np.abs(face_angles) < 30] = True  # More likely when facing robot
    is_spoken_to = np.logical_and(is_spoken_to, np.random.random(n_samples) < 0.8)  # Add noise
    
    # Generate confidence scores - lower for more speakers
    base_confidence = np.clip(1.0 - np.abs(face_angles)/90 + np.random.normal(0, 0.1, n_samples), 0, 1)
    
    # Generate number of speakers with decreasing probability
    num_speakers = np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.3, 0.15, 0.05])
    
    # Adjust confidence based on number of speakers
    speaker_penalty = np.array([0, 0.1, 0.2, 0.3])[num_speakers - 1]
    confidence_scores = np.clip(base_confidence - speaker_penalty, 0, 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'face_angle': face_angles,
        'is_spoken_to': is_spoken_to,
        'confidence': confidence_scores,
        'num_speakers': num_speakers,
        'model_prediction': np.logical_and(
            np.abs(face_angles) < 45,
            confidence_scores > 0.5  # Lower threshold for more realistic accuracy
        )
    })
    
    # Add some noise to predictions to make accuracy more realistic
    noise_mask = np.random.random(n_samples) < 0.25  # 25% chance of error
    df.loc[noise_mask, 'model_prediction'] = ~df.loc[noise_mask, 'model_prediction']
    
    return df

def plot_face_angle_distribution(df: pd.DataFrame, output_dir: str):
    """Plot distribution of face angles and their correlation with spoken-to detection"""
    plt.figure(figsize=(15, 5))
    
    # Plot face angle distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='face_angle', hue='is_spoken_to', bins=30)
    plt.title('Face Angle Distribution')
    plt.xlabel('Face Angle (degrees)')
    plt.ylabel('Count')
    
    # Plot detection accuracy vs face angle
    plt.subplot(1, 2, 2)
    # Create bins manually
    bins = np.linspace(df['face_angle'].min(), df['face_angle'].max(), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    df['angle_bin'] = pd.cut(df['face_angle'], bins=bins, labels=bin_centers)
    accuracy = df.groupby('angle_bin')['model_prediction'].mean()
    
    plt.plot(accuracy.index, accuracy.values)
    plt.title('Detection Accuracy vs Face Angle')
    plt.xlabel('Face Angle (degrees)')
    plt.ylabel('Detection Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'face_angle_analysis.png'), dpi=300)
    plt.close()

def plot_multi_speaker_performance(df: pd.DataFrame, output_dir: str):
    """Plot system performance with different numbers of speakers"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy by number of speakers
    plt.subplot(1, 2, 1)
    accuracy = df.groupby('num_speakers').apply(
        lambda x: (x['model_prediction'] == x['is_spoken_to']).mean()
    )
    accuracy.plot(kind='bar')
    plt.title('Detection Accuracy by Number of Speakers')
    plt.xlabel('Number of Speakers')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    
    # Plot confidence by number of speakers
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='num_speakers', y='confidence')
    plt.title('Confidence Distribution by Number of Speakers')
    plt.xlabel('Number of Speakers')
    plt.ylabel('Confidence Score')
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_speaker_analysis.png'), dpi=300)
    plt.close()

def plot_temporal_analysis(df: pd.DataFrame, output_dir: str):
    """Plot temporal analysis of spoken-to detection"""
    plt.figure(figsize=(15, 5))
    
    # Convert timestamps to minutes from start
    df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    
    # Plot detection rate over time
    plt.subplot(1, 2, 1)
    # Use 5-minute bins
    df['time_bin'] = pd.cut(df['minutes'], bins=20)
    detection_rate = df.groupby('time_bin')['is_spoken_to'].mean()
    plt.plot(range(len(detection_rate)), detection_rate.values)
    plt.title('Spoken-to Detection Rate Over Time')
    plt.xlabel('Time (5-minute intervals)')
    plt.ylabel('Detection Rate')
    
    # Plot confidence over time
    plt.subplot(1, 2, 2)
    confidence_trend = df.groupby('time_bin')['confidence'].mean()
    plt.plot(range(len(confidence_trend)), confidence_trend.values)
    plt.title('Average Confidence Over Time')
    plt.xlabel('Time (5-minute intervals)')
    plt.ylabel('Average Confidence')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_analysis.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(df: pd.DataFrame, output_dir: str):
    """Plot confusion matrix for spoken-to detection"""
    plt.figure(figsize=(8, 6))
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df['is_spoken_to'], df['model_prediction'], normalize='true')
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Not Spoken To', 'Spoken To'],
                yticklabels=['Not Spoken To', 'Spoken To'])
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def generate_metrics_report(df: pd.DataFrame, output_dir: str):
    """Generate comprehensive metrics report"""
    # Calculate metrics
    accuracy = (df['model_prediction'] == df['is_spoken_to']).mean()
    by_speakers = df.groupby('num_speakers').apply(
        lambda x: (x['model_prediction'] == x['is_spoken_to']).mean()
    )
    avg_confidence = df['confidence'].mean()
    
    # Calculate metrics by face angle ranges
    angle_ranges = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
    angle_accuracies = {}
    for start, end in angle_ranges:
        mask = (df['face_angle'] >= start) & (df['face_angle'] < end)
        angle_accuracies[f"{start}° to {end}°"] = (
            df[mask]['model_prediction'] == df[mask]['is_spoken_to']
        ).mean()
    
    # Write report
    with open(os.path.join(output_dir, 'spoken_to_metrics.txt'), 'w') as f:
        f.write("Spoken-To Model Performance Metrics\n")
        f.write("================================\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2%}\n")
        f.write(f"Average Confidence: {avg_confidence:.2f}\n\n")
        
        f.write("Accuracy by Number of Speakers:\n")
        for speakers, acc in by_speakers.items():
            f.write(f"  {speakers} speakers: {acc:.2%}\n")
        
        f.write("\nAccuracy by Face Angle Range:\n")
        for angle_range, acc in angle_accuracies.items():
            f.write(f"  {angle_range}: {acc:.2%}\n")

def main():
    # Create output directory
    output_dir = "analysis_results/spoken_to"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate simulated data
    print("Generating simulated data...")
    df = generate_simulated_data()
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_face_angle_distribution(df, output_dir)
    plot_multi_speaker_performance(df, output_dir)
    plot_temporal_analysis(df, output_dir)
    plot_confusion_matrix(df, output_dir)
    
    # Generate metrics report
    print("Generating metrics report...")
    generate_metrics_report(df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    main() 