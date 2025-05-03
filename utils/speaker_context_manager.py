import json
import os
from typing import Dict, Optional, List
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpeakerContext:
    speaker_id: str
    interaction_count: int = 0
    last_interaction_time: Optional[datetime] = None
    common_intents: Dict[str, int] = None
    average_confidence: float = 0.0
    conversation_history: deque = None
    embedding_index: Optional[int] = None  # Index in the embedding space
    
    def __post_init__(self):
        if self.common_intents is None:
            self.common_intents = defaultdict(int)
        if self.conversation_history is None:
            self.conversation_history = deque(maxlen=10)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'speaker_id': self.speaker_id,
            'interaction_count': self.interaction_count,
            'last_interaction_time': self.last_interaction_time.isoformat() if self.last_interaction_time else None,
            'common_intents': dict(self.common_intents),
            'average_confidence': self.average_confidence,
            'conversation_history': list(self.conversation_history),
            'embedding_index': self.embedding_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SpeakerContext':
        """Create from dictionary"""
        context = cls(speaker_id=data['speaker_id'])
        context.interaction_count = data['interaction_count']
        context.last_interaction_time = (
            datetime.fromisoformat(data['last_interaction_time'])
            if data['last_interaction_time']
            else None
        )
        context.common_intents = defaultdict(int, data['common_intents'])
        context.average_confidence = data['average_confidence']
        context.conversation_history = deque(
            data['conversation_history'],
            maxlen=10
        )
        context.embedding_index = data.get('embedding_index')
        return context

class SpeakerContextManager:
    def __init__(self, storage_dir: str = "data/speaker_contexts"):
        """
        Initialize the speaker context manager.
        
        Args:
            storage_dir: Directory to store speaker contexts
        """
        self.storage_dir = storage_dir
        self.contexts: Dict[str, SpeakerContext] = {}
        self.embedding_index = 0
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize SQLite database
        self.db_path = os.path.join(storage_dir, 'speaker_contexts.db')
        self._init_db()
        
        # Load existing contexts
        self.load_all_contexts()
    
    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS speaker_contexts (
                    speaker_id TEXT PRIMARY KEY,
                    interaction_count INTEGER,
                    last_interaction_time TEXT,
                    common_intents TEXT,
                    average_confidence REAL,
                    conversation_history TEXT,
                    embedding_index INTEGER
                )
            ''')
    
    def get_context(self, speaker_id: str) -> SpeakerContext:
        """
        Get or create speaker context.
        
        Args:
            speaker_id: Speaker ID
            
        Returns:
            SpeakerContext object
        """
        if speaker_id not in self.contexts:
            # Check if context exists in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT * FROM speaker_contexts WHERE speaker_id = ?',
                    (speaker_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    # Load from database
                    context = SpeakerContext(
                        speaker_id=row[0],
                        interaction_count=row[1],
                        last_interaction_time=datetime.fromisoformat(row[2]) if row[2] else None,
                        common_intents=defaultdict(int, json.loads(row[3])),
                        average_confidence=row[4],
                        conversation_history=deque(json.loads(row[5]), maxlen=10),
                        embedding_index=row[6]
                    )
                else:
                    # Create new context
                    context = SpeakerContext(
                        speaker_id=speaker_id,
                        embedding_index=self.embedding_index
                    )
                    self.embedding_index += 1
                
                self.contexts[speaker_id] = context
                self._save_context(context)
        
        return self.contexts[speaker_id]
    
    def update_context(self, speaker_id: str, intent: str, confidence: float, transcript: str):
        """
        Update speaker context with new interaction.
        
        Args:
            speaker_id: Speaker ID
            intent: Detected intent
            confidence: Confidence score
            transcript: Speech transcript
        """
        context = self.get_context(speaker_id)
        
        # Update context
        context.interaction_count += 1
        context.last_interaction_time = datetime.now()
        context.common_intents[intent] += 1
        context.average_confidence = (
            (context.average_confidence * (context.interaction_count - 1) + confidence)
            / context.interaction_count
        )
        context.conversation_history.append({
            'timestamp': context.last_interaction_time.isoformat(),
            'intent': intent,
            'confidence': confidence,
            'transcript': transcript
        })
        
        # Save to database
        self._save_context(context)
    
    def _save_context(self, context: SpeakerContext):
        """Save context to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO speaker_contexts
                (speaker_id, interaction_count, last_interaction_time, common_intents,
                 average_confidence, conversation_history, embedding_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                context.speaker_id,
                context.interaction_count,
                context.last_interaction_time.isoformat() if context.last_interaction_time else None,
                json.dumps(dict(context.common_intents)),
                context.average_confidence,
                json.dumps(list(context.conversation_history)),
                context.embedding_index
            ))
    
    def load_all_contexts(self):
        """Load all contexts from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM speaker_contexts')
            for row in cursor:
                context = SpeakerContext(
                    speaker_id=row[0],
                    interaction_count=row[1],
                    last_interaction_time=datetime.fromisoformat(row[2]) if row[2] else None,
                    common_intents=defaultdict(int, json.loads(row[3])),
                    average_confidence=row[4],
                    conversation_history=deque(json.loads(row[5]), maxlen=10),
                    embedding_index=row[6]
                )
                self.contexts[row[0]] = context
                self.embedding_index = max(self.embedding_index, row[6] + 1)
    
    def get_speaker_history(self, speaker_id: str) -> List[Dict]:
        """Get interaction history for a speaker"""
        context = self.get_context(speaker_id)
        return list(context.conversation_history)
    
    def get_speaker_stats(self, speaker_id: str) -> Dict:
        """Get statistics for a speaker"""
        context = self.get_context(speaker_id)
        return {
            'interaction_count': context.interaction_count,
            'last_interaction': context.last_interaction_time.isoformat() if context.last_interaction_time else None,
            'common_intents': dict(context.common_intents),
            'average_confidence': context.average_confidence,
            'embedding_index': context.embedding_index
        }
    
    def get_all_speakers(self) -> List[Dict]:
        """Get information about all speakers"""
        return [
            {
                'speaker_id': speaker_id,
                **self.get_speaker_stats(speaker_id)
            }
            for speaker_id in self.contexts
        ]
    
    def export_contexts(self, filepath: str):
        """Export all contexts to a JSON file"""
        data = {
            speaker_id: context.to_dict()
            for speaker_id, context in self.contexts.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_contexts(self, filepath: str):
        """Import contexts from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for speaker_id, context_data in data.items():
            context = SpeakerContext.from_dict(context_data)
            self.contexts[speaker_id] = context
            self._save_context(context)
            self.embedding_index = max(self.embedding_index, context.embedding_index + 1)

# Example usage:
if __name__ == "__main__":
    # Initialize context manager
    manager = SpeakerContextManager()
    
    # Update context for a speaker
    manager.update_context(
        speaker_id="SPEAKER_1",
        intent="greeting",
        confidence=0.9,
        transcript="Hey robot, how are you?"
    )
    
    # Get speaker history
    history = manager.get_speaker_history("SPEAKER_1")
    print(f"Speaker 1 history: {history}")
    
    # Get speaker stats
    stats = manager.get_speaker_stats("SPEAKER_1")
    print(f"Speaker 1 stats: {stats}")
    
    # Export contexts
    manager.export_contexts("speaker_contexts.json") 