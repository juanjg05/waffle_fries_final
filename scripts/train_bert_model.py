import os
import json
import logging
from typing import List, Dict
import argparse
from models.bert_intent_classifier import BERTIntentClassifier, prepare_conversation_data
from utils.rttm_parser import parse_conversation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_conversation_data(data_dir: str) -> List[Dict]:
    """
    Load conversation data from RTTM and transcript files.
    
    Args:
        data_dir: Directory containing RTTM and transcript files
        
    Returns:
        List of conversation dictionaries
    """
    conversations = []
    
    # Walk through data directory
    for root, _, files in os.walk(data_dir):
        # Find matching RTTM and transcript files
        rttm_files = [f for f in files if f.endswith('.rttm')]
        
        for rttm_file in rttm_files:
            # Get corresponding transcript file
            transcript_file = rttm_file.replace('.rttm', '.txt')
            if transcript_file not in files:
                logger.warning(f"No transcript file found for {rttm_file}")
                continue
            
            # Parse conversation
            try:
                rttm_path = os.path.join(root, rttm_file)
                transcript_path = os.path.join(root, transcript_file)
                
                conv_data = parse_conversation(rttm_path, transcript_path)
                conversations.extend(conv_data)
                
            except Exception as e:
                logger.error(f"Error processing {rttm_file}: {str(e)}")
                continue
    
    return conversations

def main():
    parser = argparse.ArgumentParser(description='Train BERT-Tiny model for intent classification')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing RTTM and transcript files')
    parser.add_argument('--output_dir', type=str, default='bert_intent_model',
                      help='Directory to save trained model')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load conversation data
    logger.info("Loading conversation data...")
    conversations = load_conversation_data(args.data_dir)
    
    if not conversations:
        logger.error("No conversation data found!")
        return
    
    logger.info(f"Loaded {len(conversations)} conversation segments")
    
    # Prepare data for training
    logger.info("Preparing training data...")
    prepared_data = prepare_conversation_data(conversations)
    
    # Define label mapping
    label_mapping = {
        'spoken_to_robot': 1,
        'not_spoken_to_robot': 0
    }
    
    # Save label mapping
    with open(os.path.join(args.output_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)
    
    # Initialize classifier
    logger.info("Initializing BERT-Tiny model...")
    classifier = BERTIntentClassifier()
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, val_dataset = classifier.prepare_training_data(
        prepared_data, label_mapping)
    
    # Train model
    logger.info("Starting training...")
    classifier.train(
        train_dataset,
        val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    logger.info(f"Training complete! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 