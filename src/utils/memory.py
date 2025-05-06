class SpeakerMemory:
    def __init__(self):
        # A dictionary to store speaker IDs and their conversations.
        self.memory = {}

    def add_conversation(self, speaker_id, conversation_text):
        """
        Add a conversation text for a speaker ID.
        
        Args:
        - speaker_id (str): The ID of the speaker (e.g., 'speaker_1').
        - conversation_text (str): The conversation text to store.
        """
        if speaker_id not in self.memory:
            self.memory[speaker_id] = []
        # Append the new conversation text for the given speaker
        self.memory[speaker_id].append(conversation_text)

    def get_conversations(self, speaker_id):
        """
        Retrieve all stored conversations for a speaker ID.
        
        Args:
        - speaker_id (str): The ID of the speaker (e.g., 'speaker_1').
        
        Returns:
        - List of conversation texts for the given speaker ID.
        """
        return self.memory.get(speaker_id, [])
