import os
import json
import sqlite3
from datetime import datetime
from collections import defaultdict, deque

def export_contexts_to_json(db_path: str, json_path: str):
    """Export speaker contexts from SQLite to JSON"""
    contexts = {}
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute('SELECT * FROM speaker_contexts')
        columns = [description[0] for description in cursor.description]
        
        for row in cursor.fetchall():
            data = dict(zip(columns, row))
            speaker_id = data['speaker_id']
            
            # Convert string representations back to Python objects
            conversation_history = json.loads(data['conversation_history']) if data['conversation_history'] else []
            common_intents = json.loads(data['common_intents']) if data['common_intents'] else {}
            embedding = json.loads(data['embedding']) if data['embedding'] else None
            
            # Create context dictionary
            context = {
                'speaker_id': speaker_id,
                'interaction_count': data['interaction_count'],
                'last_interaction_time': data['last_interaction_time'],
                'common_intents': common_intents,
                'average_confidence': data['average_confidence'],
                'conversation_history': conversation_history,
                'embedding_index': data['embedding_index'],
                'embedding': embedding,
                'conversation_index': data.get('conversation_index', 1),
                'has_face_direction': bool(data.get('has_face_direction', 0)),
                'num_speakers': data.get('num_speakers', 1)
            }
            
            contexts[speaker_id] = context
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Save to JSON file
    with open(json_path, 'w') as f:
        json.dump(contexts, f, indent=2)
    
    print(f"Exported {len(contexts)} speaker contexts to {json_path}")

def main():
    # Use the known database path
    db_path = "tests/data/speaker_contexts/speaker_contexts.db"
    
    if not os.path.exists(db_path):
        print(f"Error: Could not find speaker contexts database at {db_path}")
        return
    
    print(f"Found database at: {db_path}")
    
    # Export to JSON in the same directory
    json_path = os.path.join(os.path.dirname(db_path), "speaker_contexts.json")
    export_contexts_to_json(db_path, json_path)

if __name__ == "__main__":
    main() 