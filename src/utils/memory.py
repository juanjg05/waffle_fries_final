import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class ConversationMemory:
    def __init__(self, storage_dir="memory"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.current_session = {}
        
    def store(self, key: str, value: Any):
        self.current_session[key] = value
        
    def get(self, key: str, default=None) -> Any:
        return self.current_session.get(key, default)
        
    def save_session(self, session_id: str = None):
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        filename = os.path.join(self.storage_dir, f"session_{session_id}.json")
        
        with open(filename, 'w') as f:
            json.dump(self.current_session, f, indent=2)
            
        return filename
        
    def load_session(self, session_id: str):
        filename = os.path.join(self.storage_dir, f"session_{session_id}.json")
        
        try:
            with open(filename, 'r') as f:
                self.current_session = json.load(f)
        except FileNotFoundError:
            self.current_session = {}
        
        return self.current_session
        
    def clear(self):
        self.current_session = {}
