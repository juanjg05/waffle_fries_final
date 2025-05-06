What to do? Figure it out. I'm just kidding.

1. Install requirements (venv???):
   ```
   pip install -r requirements.txt
   ```

2. Please create a .env file with your Hugging Face token to run PyAnnote, you may need to go and accept their rules before you are able to run the code:
```
HF_TOKEN=your_huggingface_token

```

these are the links to the website...:


3. Run the file...:
   ```
   python main.py
   ```

3. Press Enter to start recording, and Enter again to stop.

## File Structures. 

```
waffle_fries/
|
├── data/                             # Data directory
│   ├── conversation_data/            # Processed conversations
│   │   ├── conversation_1/           # Individual conversation folder
│   │   │   ├── audio.wav             # Conversation audio
│   │   │   ├── frames/               # Video frames
│   │   │   └── results.json          # Conversation results
│   │   └── conversation_2/           # Another conversation
│   └── speaker_contexts/             # Speaker data
│       ├── speaker_contexts.json     # Combined speaker data
│       └── speaker_0/                # Individual speaker folder
│           ├── speaker_0.json        # Speaker details
│           ├── speaker_0_embedding.npy    # Speaker voice embedding
│           └── speaker_0_transcript.txt   # Speaker transcripts
├── src/                              # Source code
│   ├── audio_video_processing/       # Main processing package
│   │   └── scripts/                  # Processing scripts
│   ├── models/                       # Model implementations
│   └── utils/                        # Utility functions
├── requirements.txt                  # Dependencies
└── main.py                           # Entry point
```

## ✨ Features

- 🗣️ **Speech Transcription** - Uses Whisper for accurate transcription
- 👁️ **Face Angle Detection** - Determines if someone is looking at the camera
- 👋 **Is Spoken To** - Tells if a person is being spoken to based on face angle
- 🧠 **Speaker Recognition** - Identifies speakers across conversations
- 📝 **Conversation History** - Maintains a record of all conversations

## 🔊 Speaker Identification

The system uses voice embeddings to identify and track speakers across different conversations:

1. For each audio segment, a voice embedding is extracted using either:
   - SpeechBrain's ECAPA-TDNN model (in production environments)
   - MFCC features as a fallback (for Windows compatibility)

2. New speakers are assigned a unique ID (`speaker_0`, `speaker_1`, etc.)

3. For subsequent conversations, the system:
   - Extracts voice embeddings from new audio segments
   - Computes the cosine similarity between these embeddings and stored speaker embeddings
   - Assigns the existing speaker ID if similarity is above the threshold (default: 0.85) (No longer k-clustering algorithm for slower processing)
   - Creates a new speaker ID if no match is found







Please create a .env file with your Hugging Face token to run PyAnnote, you may need to go and accept their rules before you are able to run the code:
```
HF_TOKEN=your_huggingface_token
```

This is the structure of our outputs:

1. **Conversation Results** (`data/conversation_data/conversation_X/results.json`):
   ```json
   {
     "conversation_id": "conversation_1",
     "speaker_ids": ["speaker_0", "speaker_1"],
     "results": [
       {
         "speaker_id": "speaker_0",
         "transcript": "Hello there!",
         "is_spoken_to": true,
         "face_angle": 12.5
       }
     ]
   }
   ```

2. **Speaker Contexts** (`data/speaker_contexts/speaker_ID/speaker_ID.json`):
   This is whats used to match speaker ID's and get context from past conversations.


