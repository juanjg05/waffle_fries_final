# Robot Audio Processing System

This system provides real-time audio processing capabilities for a robot, including:
- Speaker diarization
- Speech recognition
- Speaker direction detection
- Robot movement control

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/robot-audio-processing.git
cd robot-audio-processing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

Basic usage example:

```python
from models.nemo_diarization_model import diarize_speech, combine_diarization_with_transcript
from robot.spatial_audio import get_speaker_direction
from utils.memory import SpeakerMemory

# Initialize components
memory = SpeakerMemory()

# Process audio file
audio_file = "path/to/audio.wav"

# Get speaker direction
direction_info = get_speaker_direction(audio_file)

# Perform diarization
diarization_results = diarize_speech(audio_file)

# Process results
for segment in diarization_results:
    print(f"Speaker {segment.speaker_id}:")
    print(f"  Time: {segment.start_time:.2f} - {segment.end_time:.2f}")
    print(f"  Direction: {direction_info}")
```

## Project Structure

```
robot-audio-processing/
├── models/
│   ├── nemo_diarization_model.py
│   └── spatial_audio.py
├── robot/
│   ├── movement.py
│   └── spatial_audio.py
├── utils/
│   ├── memory.py
│   └── rttm_parser.py
├── tests/
│   └── test_diarization.py
├── requirements.txt
└── README.md
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## License

MIT License
