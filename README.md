**Waffle Fries Final Project FRI I** 
*Juan Garcia, Archita Singh, Ayah Hassan*


What to do? Figure it out. I'm just kidding.

1. Install requirements (venv???):
   ```
   pip install -r requirements.txt
   ```

2. Please create a .env file with your Hugging Face token to run PyAnnote, you may need to go and accept their rules before you are able to run the code:
```
HF_TOKEN=your_huggingface_token

```

these are the links to the website:
https://huggingface.co/pyannote/speaker-diarization-3.1

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


