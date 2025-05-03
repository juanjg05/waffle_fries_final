import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import json
import os
from utils.speaker_context_manager import SpeakerContextManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, 
                 context_manager: SpeakerContextManager, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.context_manager = context_manager
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        speaker_id = text.split()[0].replace('[SPEAKER_', '').replace(']', '')
        
        # Get speaker context
        context = self.context_manager.get_context(speaker_id)
        
        # Add context information to the text
        context_info = (
            f"[INTERACTIONS_{context.interaction_count}]"
            f"[TOP_INTENT_{max(context.common_intents.items(), key=lambda x: x[1])[0] if context.common_intents else 'none'}]"
            f"[AVG_CONF_{context.average_confidence:.2f}]"
            f"[EMBEDDING_{context.embedding_index}]"
        )
        text = f"{text} {context_info}"

        # Add speaker tokens and end-of-utterance tokens
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTIntentClassifier:
    def __init__(self, 
                 model_name: str = "prajjwal1/bert-tiny",
                 num_labels: int = 2,  # Binary classification: spoken to robot or not
                 max_length: int = 128,
                 context_window: int = 10):  # Number of past interactions to consider
        """
        Initialize BERT-Tiny model for intent classification.
        
        Args:
            model_name: Name of the pre-trained model
            num_labels: Number of classification labels
            max_length: Maximum sequence length
            context_window: Number of past interactions to consider
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.context_window = context_window
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Initialize speaker context manager
        self.context_manager = SpeakerContextManager()
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_training_data(self, 
                            conversations: List[Dict],
                            label_mapping: Dict[str, int]) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets.
        
        Args:
            conversations: List of conversation dictionaries with 'text' and 'label' keys
            label_mapping: Dictionary mapping label names to integers
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Convert conversations to format suitable for training
        texts = []
        labels = []
        
        for conv in conversations:
            # Add speaker tokens and end-of-utterance tokens
            text = f"[SPEAKER_{conv['speaker_id']}] {conv['text']}"
            if conv.get('is_end_of_utterance', False):
                text += " [EOU]"
            
            texts.append(text)
            labels.append(label_mapping[conv['label']])
            
            # Update speaker context
            self.context_manager.update_context(
                speaker_id=conv['speaker_id'],
                intent=conv['label'],
                confidence=conv.get('confidence', 1.0),
                transcript=conv['text']
            )
        
        # Split into train and validation sets (80-20 split)
        split_idx = int(len(texts) * 0.8)
        train_texts, val_texts = texts[:split_idx], texts[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        # Create datasets
        train_dataset = ConversationDataset(
            train_texts, train_labels, self.tokenizer, self.context_manager, self.max_length)
        val_dataset = ConversationDataset(
            val_texts, val_labels, self.tokenizer, self.context_manager, self.max_length)
        
        return train_dataset, val_dataset

    def train(self, 
             train_dataset: Dataset,
             val_dataset: Dataset,
             output_dir: str = "bert_intent_model",
             num_epochs: int = 3,
             batch_size: int = 16,
             learning_rate: float = 2e-5):
        """
        Train the BERT-Tiny model.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )

        trainer.train()
        
        # Save the model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save contexts
        self.context_manager.export_contexts(os.path.join(output_dir, 'speaker_contexts.json'))

    def _compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def predict(self, text: str, speaker_id: str) -> Tuple[int, float]:
        """
        Predict intent for a given text.
        
        Args:
            text: Input text
            speaker_id: Speaker ID
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        # Get speaker context
        context = self.context_manager.get_context(speaker_id)
        
        # Add speaker tokens and context information
        context_info = (
            f"[INTERACTIONS_{context.interaction_count}]"
            f"[TOP_INTENT_{max(context.common_intents.items(), key=lambda x: x[1])[0] if context.common_intents else 'none'}]"
            f"[AVG_CONF_{context.average_confidence:.2f}]"
            f"[EMBEDDING_{context.embedding_index}]"
        )
        input_text = f"[SPEAKER_{speaker_id}] {text} {context_info}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Get predicted label and confidence
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_label].item()
            
            # Update speaker context
            self.context_manager.update_context(
                speaker_id=speaker_id,
                intent=str(predicted_label),
                confidence=confidence,
                transcript=text
            )
            
        return predicted_label, confidence

    def get_speaker_history(self, speaker_id: str) -> List[Dict]:
        """Get interaction history for a speaker"""
        return self.context_manager.get_speaker_history(speaker_id)

    def get_speaker_stats(self, speaker_id: str) -> Dict:
        """Get statistics for a speaker"""
        return self.context_manager.get_speaker_stats(speaker_id)

    def save_model(self, model_dir: str):
        """Save model and contexts"""
        # Save model and tokenizer
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Save contexts
        self.context_manager.export_contexts(os.path.join(model_dir, 'speaker_contexts.json'))

    def load_model(self, model_dir: str):
        """Load model and contexts"""
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load contexts
        self.context_manager.import_contexts(os.path.join(model_dir, 'speaker_contexts.json'))
        
        self.model.to(self.device)

def prepare_conversation_data(conversations: List[Dict]) -> List[Dict]:
    """
    Prepare conversation data for training.
    
    Args:
        conversations: List of conversation dictionaries from RTTM parser
        
    Returns:
        List of dictionaries with text and label
    """
    prepared_data = []
    
    for conv in conversations:
        # Determine if this is spoken to robot based on context
        # This is a simple heuristic - you should replace this with your actual logic
        is_spoken_to_robot = (
            "robot" in conv['text'].lower() or
            "hey" in conv['text'].lower() or
            "hello" in conv['text'].lower()
        )
        
        prepared_data.append({
            'text': conv['text'],
            'speaker_id': conv['speaker_id'],
            'label': 'spoken_to_robot' if is_spoken_to_robot else 'not_spoken_to_robot',
            'is_end_of_utterance': conv.get('is_end_of_utterance', False)
        })
    
    return prepared_data

# Example usage:
if __name__ == "__main__":
    # Initialize classifier
    classifier = BERTIntentClassifier()
    
    # Example conversations
    conversations = [
        {
            'speaker_id': 'SPEAKER_1',
            'text': 'Hey robot, can you help me?',
            'is_end_of_utterance': True,
            'label': 'spoken_to_robot',
            'confidence': 0.9
        },
        {
            'speaker_id': 'SPEAKER_2',
            'text': 'I think we should go left',
            'is_end_of_utterance': True,
            'label': 'not_spoken_to_robot',
            'confidence': 0.8
        }
    ]
    
    # Define label mapping
    label_mapping = {
        'spoken_to_robot': 1,
        'not_spoken_to_robot': 0
    }
    
    # Prepare datasets
    train_dataset, val_dataset = classifier.prepare_training_data(
        conversations, label_mapping)
    
    # Train model
    classifier.train(train_dataset, val_dataset)
    
    # Make prediction
    text = "Hey robot, can you help me?"
    speaker_id = "SPEAKER_1"
    label, confidence = classifier.predict(text, speaker_id)
    print(f"Predicted label: {label}, Confidence: {confidence:.2f}")
    
    # Get speaker history
    history = classifier.get_speaker_history(speaker_id)
    print(f"Speaker history: {history}")
    
    # Get speaker stats
    stats = classifier.get_speaker_stats(speaker_id)
    print(f"Speaker stats: {stats}")
    
    # Save model and contexts
    classifier.save_model("bert_intent_model") 