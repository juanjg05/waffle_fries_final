# Robot Audio Processing System

This project implements a comprehensive audio processing system for robots, featuring speaker diarization, spatial audio processing, and spoken-to identification.

## Components

### 1. Speaker Diarization
- Uses NeMo EEND for speaker segmentation and clustering
- Processes RTTM files for speaker-attributed transcripts
- Maintains conversation history and speaker identification

### 2. Spatial Audio Processing
- Multi-channel audio processing for speaker direction detection
- GCC-PHAT based delay estimation
- Angle and distance estimation
- Confidence scoring

### 3. Speaker Name Model
- LLM-based speaker name identification
- Maintains speaker profiles with conversation history
- Confidence scoring for name assignments

### 4. Spoken-to Identification
- Multi-modal approach combining:
  - BERT-Tiny for intent classification
  - Prosody analysis (speaking rate, pitch, volume)
  - Spatial features (angle, distance)
  - Number of speakers context

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage
```python
from models.nemo_diarization_model import diarize_speech
from robot.spatial_audio import get_speaker_direction
from models.speaker_name_model import SpeakerNameModel
from models.spoken_to_model import SpokenToModel

# Initialize models
speaker_name_model = SpeakerNameModel(api_key="your-api-key")
spoken_to_model = SpokenToModel()

# Process audio
audio_file = "path/to/audio.wav"
rttm_file = "path/to/output.rttm"

# Get speaker diarization
diarization_result = diarize_speech(audio_file)

# Get speaker direction
direction = get_speaker_direction(audio_file)

# Process each speaker segment
for segment in diarization_result:
    # Update speaker name
    name = speaker_name_model.update_speaker(segment.speaker_id, segment.transcript)
    
    # Check if spoken to robot
    is_spoken_to, confidence = spoken_to_model.is_spoken_to_robot(segment.features)
    
    if is_spoken_to:
        print(f"Speaker {name} is addressing the robot (confidence: {confidence:.2f})")
```

## Project Structure

```
.
├── models/
│   ├── nemo_diarization_model.py
│   ├── speaker_name_model.py
│   └── spoken_to_model.py
├── robot/
│   ├── spatial_audio.py
│   └── movement.py
├── utils/
│   ├── rttm_parser.py
│   └── memory.py
├── requirements.txt
└── README.md
```

## Dependencies

- numpy: Numerical computing
- torch: Deep learning framework
- transformers: BERT model and tokenizers
- librosa: Audio processing
- openai: GPT API access
- nemo-toolkit: Speaker diarization
- scipy: Scientific computing
- soundfile: Audio file I/O
- python-dotenv: Environment variable management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
