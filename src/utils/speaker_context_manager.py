import json
import os
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
import pickle

@dataclass
class SpeakerContext:
    speaker_id: str
    interaction_count: int = 0
    last_interaction_time: Optional[datetime] = None
    common_intents: Dict[str, int] = None
    average_confidence: float = 0.0
    conversation_history: deque = None
    embedding_index: Optional[int] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.common_intents is None:
            self.common_intents = defaultdict(int)
        if self.conversation_history is None:
            self.conversation_history = deque(maxlen=100)
    
    def to_dict(self) -> Dict:
        return {
            'speaker_id': self.speaker_id,
            'interaction_count': self.interaction_count,
            'last_interaction_time': self.last_interaction_time.isoformat() if self.last_interaction_time else None,
            'common_intents': dict(self.common_intents),
            'average_confidence': self.average_confidence,
            'conversation_history': list(self.conversation_history),
            'embedding_index': self.embedding_index,
            'embedding': self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SpeakerContext':
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
            maxlen=100
        )
        context.embedding_index = data.get('embedding_index')
        context.embedding = data.get('embedding')
        return context

class SpeakerContextManager:
    def __init__(self, storage_dir: str = "data/speaker_contexts"):
        self.storage_dir = storage_dir
        self.contexts: Dict[str, SpeakerContext] = {}
        self.embedding_index = 0
        
        os.makedirs(storage_dir, exist_ok=True)
        
        self.db_path = os.path.join(storage_dir, 'speaker_contexts.db')
        self._init_db()
        
        self.load_all_contexts()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DROP TABLE IF EXISTS speaker_contexts')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS speaker_contexts (
                    speaker_id TEXT PRIMARY KEY,
                    interaction_count INTEGER,
                    last_interaction_time TEXT,
                    common_intents TEXT,
                    average_confidence REAL,
                    conversation_history TEXT,
                    embedding_index INTEGER,
                    embedding TEXT,
                    conversation_index INTEGER DEFAULT 1
                )
            ''')
    
    def get_context(self, speaker_id: str) -> SpeakerContext:
        if speaker_id not in self.contexts:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT * FROM speaker_contexts WHERE speaker_id = ?',
                    (speaker_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    context = SpeakerContext(
                        speaker_id=row[0],
                        interaction_count=row[1],
                        last_interaction_time=datetime.fromisoformat(row[2]) if row[2] else None,
                        common_intents=defaultdict(int, json.loads(row[3])),
                        average_confidence=row[4],
                        conversation_history=deque(json.loads(row[5]), maxlen=100),
                        embedding_index=row[6],
                        embedding=json.loads(row[7]) if row[7] else None
                    )
                else:
                    context = SpeakerContext(
                        speaker_id=speaker_id,
                        embedding_index=self.embedding_index
                    )
                    self.embedding_index += 1
                
                self.contexts[speaker_id] = context
                self._save_context(context)
        
        return self.contexts[speaker_id]
    
    def update_context(self, speaker_id: str, start_time: float, end_time: float, confidence: float, transcript: str = "", embedding: Optional[List[float]] = None, conversation_index: int = 1):
        try:
            context = self.get_context(speaker_id)
            
            context.interaction_count += 1
            context.last_interaction_time = datetime.now()
            context.average_confidence = (
                (context.average_confidence * (context.interaction_count - 1) + confidence) /
                context.interaction_count
            )
            
            if embedding is not None:
                context.embedding = embedding
            
            context.conversation_history.append({
                'conversation_index': conversation_index,
                'timestamp': context.last_interaction_time.isoformat(),
                'start_time': start_time,
                'end_time': end_time,
                'confidence': confidence,
                'transcript': transcript
            })
            
            self._save_context(context)
            
        except Exception as e:
            print(f"Error updating context: {str(e)}")
    
    def _save_context(self, context: SpeakerContext):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO speaker_contexts
                (speaker_id, interaction_count, last_interaction_time, common_intents,
                 average_confidence, conversation_history, embedding_index, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                context.speaker_id,
                context.interaction_count,
                context.last_interaction_time.isoformat() if context.last_interaction_time else None,
                json.dumps(dict(context.common_intents)),
                context.average_confidence,
                json.dumps(list(context.conversation_history)),
                context.embedding_index,
                json.dumps(context.embedding) if context.embedding else None
            ))
    
    def load_all_contexts(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM speaker_contexts')
                for row in cursor.fetchall():
                    context = SpeakerContext(
                        speaker_id=row[0],
                        interaction_count=row[1],
                        last_interaction_time=datetime.fromisoformat(row[2]) if row[2] else None,
                        common_intents=defaultdict(int, json.loads(row[3])),
                        average_confidence=row[4],
                        conversation_history=deque(json.loads(row[5]), maxlen=100),
                        embedding_index=row[6],
                        embedding=json.loads(row[7]) if row[7] else None
                    )
                    self.contexts[context.speaker_id] = context
                    
                    if context.embedding_index is not None and context.embedding_index >= self.embedding_index:
                        self.embedding_index = context.embedding_index + 1
        except Exception as e:
            print(f"Error loading contexts: {str(e)}")
    
    def get_speaker_history(self, speaker_id: str) -> List[Dict]:
        context = self.get_context(speaker_id)
        return list(context.conversation_history)
    
    def get_speaker_stats(self, speaker_id: str) -> Dict:
        context = self.get_context(speaker_id)
        return {
            'speaker_id': context.speaker_id,
            'interaction_count': context.interaction_count,
            'last_interaction': context.last_interaction_time.isoformat() if context.last_interaction_time else None,
            'average_confidence': context.average_confidence,
            'history_count': len(context.conversation_history)
        }
    
    def get_all_speakers(self) -> List[Dict]:
        speakers = []
        for speaker_id, context in self.contexts.items():
            speakers.append({
                'speaker_id': speaker_id,
                'interaction_count': context.interaction_count,
                'last_interaction': context.last_interaction_time.isoformat() if context.last_interaction_time else None,
                'embedding': context.embedding
            })
        return speakers
    
    def save_to_file(self, filepath: str):
        try:
            contexts_data = {}
            for speaker_id, context in self.contexts.items():
                contexts_data[speaker_id] = context.to_dict()
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(contexts_data, f, indent=2)
            
            print(f"Exported {len(contexts_data)} contexts to {filepath}")
        except Exception as e:
            print(f"Error exporting contexts: {str(e)}")
    
    def load_from_file(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                contexts_data = json.load(f)
            
            for speaker_id, context_data in contexts_data.items():
                context = SpeakerContext.from_dict(context_data)
                self.contexts[speaker_id] = context
                
                if context.embedding_index is not None and context.embedding_index >= self.embedding_index:
                    self.embedding_index = context.embedding_index + 1
            
            print(f"Imported {len(contexts_data)} contexts from {filepath}")
            
            self._save_all_contexts()
        except Exception as e:
            print(f"Error importing contexts: {str(e)}")
    
    def _save_all_contexts(self):
        with sqlite3.connect(self.db_path) as conn:
            for speaker_id, context in self.contexts.items():
                self._save_context(context)
    
    def add_speech_segment(self, speaker_id: str, transcript: str, start_time: float, end_time: float, embedding: Optional[List[float]] = None, confidence: float = 1.0):
        self.update_context(
            speaker_id=speaker_id,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            transcript=transcript,
            embedding=embedding
        )

if __name__ == "__main__":
    manager = SpeakerContextManager()
    
    manager.update_context(
        speaker_id="SPEAKER_1",
        start_time=0.0,
        end_time=1.0,
        confidence=0.9,
        transcript="Hey robot, how are you?"
    )
    
    history = manager.get_speaker_history("SPEAKER_1")
    print(f"Speaker 1 history: {history}")
    
    stats = manager.get_speaker_stats("SPEAKER_1")
    print(f"Speaker 1 stats: {stats}")
    
    manager.save_to_file("speaker_contexts.json") 