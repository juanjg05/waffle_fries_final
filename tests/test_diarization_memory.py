import os
import json
import tempfile
import numpy as np
from robot_audio_processor.models.diarization_model import DiarizationResult, RealTimeProcessor

def test_speaker_context_manager_builds_json():
    # Create a temporary directory for context storage
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a RealTimeProcessor with the temp storage dir
        processor = RealTimeProcessor(storage_dir=tmpdir)
        
        # Create mock diarization results
        result = DiarizationResult(
            speaker_id="SPEAKER_1",
            start_time=0.0,
            end_time=2.0,
            transcript="Hello world",
            confidence=0.95,
            speaker_embedding=np.random.rand(192),
            speaker_confidence=0.95
        )
        # Update context
        processor.context_manager.update_context(
            speaker_id=result.speaker_id,
            start_time=result.start_time,
            end_time=result.end_time,
            confidence=result.confidence,
            transcript=result.transcript,
            embedding=result.speaker_embedding.tolist(),
            conversation_index=1
        )
        # Save contexts
        json_path = os.path.join(tmpdir, "speaker_contexts.json")
        processor.save_contexts(json_path)
        # Load and check JSON
        with open(json_path, "r") as f:
            data = json.load(f)
        assert "SPEAKER_1" in data
        assert len(data["SPEAKER_1"]) > 0
        entry = data["SPEAKER_1"][0]
        assert entry["transcript"] == "Hello world"
        assert entry["confidence"] == 0.95
        assert entry["conversation_index"] == 1
        assert isinstance(entry["embedding"], list)
        assert len(entry["embedding"]) == 192 