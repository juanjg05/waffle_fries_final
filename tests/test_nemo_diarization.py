import os
import sys
import logging
import numpy as np
from datetime import datetime
import pytest
from models.nemo_diarization_model import RealTimeProcessor, DiarizationResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data directory
TEST_DATA_DIR = "tests/data"
os.makedirs(TEST_DATA_DIR, exist_ok=True)

@pytest.fixture(scope="session")
def audio_file():
    """Create and return a test audio file path"""
    try:
        import soundfile as sf
        import numpy as np
        
        # Create a 10-second audio file with two speakers
        sample_rate = 16000
        duration = 10
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Speaker 1: 0-3s and 6-8s
        speaker1 = np.sin(2 * np.pi * 440 * t) * 0.5
        speaker1[3*sample_rate:6*sample_rate] = 0
        speaker1[8*sample_rate:] = 0
        
        # Speaker 2: 3-6s and 8-10s
        speaker2 = np.sin(2 * np.pi * 880 * t) * 0.5
        speaker2[:3*sample_rate] = 0
        speaker2[6*sample_rate:8*sample_rate] = 0
        
        # Combine speakers
        audio = speaker1 + speaker2
        
        # Add some noise
        noise = np.random.normal(0, 0.1, len(audio))
        audio = audio + noise
        
        # Save audio file
        audio_path = os.path.join(TEST_DATA_DIR, "test_conversation.wav")
        sf.write(audio_path, audio, sample_rate)
        
        return audio_path
        
    except Exception as e:
        logger.error(f"Error creating test audio: {str(e)}")
        return None

@pytest.fixture(scope="session")
def processor():
    """Fixture to create and return a RealTimeProcessor instance"""
    try:
        processor = RealTimeProcessor(
            model_dir="models/nemo",
            storage_dir=os.path.join(TEST_DATA_DIR, "speaker_contexts"),
            use_vad=True,
            use_enhancement=True
        )
        assert processor is not None
        assert processor.diarization_model is not None
        assert processor.asr_model is not None
        assert processor.bert_model is not None
        if processor.use_vad:
            assert processor.vad_model is not None
        if processor.use_enhancement:
            assert processor.enhancement_model is not None
        logger.info("Processor initialization test passed")
        return processor
    except Exception as e:
        logger.error(f"Processor initialization test failed: {str(e)}")
        raise

def test_processor_initialization(processor):
    """Test processor initialization"""
    assert processor is not None
    assert processor.diarization_model is not None
    assert processor.asr_model is not None
    assert processor.bert_model is not None
    if processor.use_vad:
        assert processor.vad_model is not None
    if processor.use_enhancement:
        assert processor.enhancement_model is not None
    logger.info("Processor initialization test passed")

@pytest.mark.dependency(name="test_audio_preprocessing")
def test_audio_preprocessing(processor, audio_file):
    """Test audio preprocessing"""
    try:
        assert audio_file is not None
        
        processed_path = processor._preprocess_audio(audio_file)
        assert processed_path is not None
        assert os.path.exists(processed_path)
        
        logger.info("Audio preprocessing test passed")
        return processed_path
    except Exception as e:
        logger.error(f"Audio preprocessing test failed: {str(e)}")
        raise

@pytest.mark.dependency(name="test_speaker_diarization", depends=["test_audio_preprocessing"])
def test_speaker_diarization(processor, audio_file):
    """Test speaker diarization"""
    try:
        results = []
        
        def result_callback(result: DiarizationResult):
            results.append(result)
            logger.info(f"Detected speaker {result.speaker_id}:")
            logger.info(f"  Time: {result.start_time:.2f} - {result.end_time:.2f}")
            logger.info(f"  Transcript: {result.transcript}")
            logger.info(f"  Confidence: {result.confidence:.2f}")
            logger.info(f"  Speaker Confidence: {result.speaker_confidence:.2f}")
        
        # Start processing
        processor.start_processing()
        
        # Process audio
        processor.process_audio(
            audio_path=audio_file,
            callback=result_callback,
            num_speakers=2
        )
        
        # Wait for processing to complete
        import time
        time.sleep(15)  # Wait for processing
        
        # Stop processing
        processor.stop_processing()
        
        # Verify results
        assert len(results) > 0
        assert all(isinstance(r, DiarizationResult) for r in results)
        assert all(r.speaker_embedding is not None for r in results)
        
        # Check speaker consistency
        speaker_ids = set(r.speaker_id for r in results)
        assert len(speaker_ids) <= 2  # Should detect at most 2 speakers
        
        logger.info("Speaker diarization test passed")
        return results
    except Exception as e:
        logger.error(f"Speaker diarization test failed: {str(e)}")
        raise

@pytest.mark.dependency(name="test_speaker_matching", depends=["test_speaker_diarization"])
def test_speaker_matching(processor, audio_file):
    """Test speaker matching with new audio"""
    try:
        new_results = []
        
        def new_result_callback(result: DiarizationResult):
            new_results.append(result)
            logger.info(f"New detection - Speaker {result.speaker_id}:")
            logger.info(f"  Time: {result.start_time:.2f} - {result.end_time:.2f}")
            logger.info(f"  Transcript: {result.transcript}")
            logger.info(f"  Confidence: {result.confidence:.2f}")
            logger.info(f"  Speaker Confidence: {result.speaker_confidence:.2f}")
        
        # Start processing
        processor.start_processing()
        
        # Process new audio
        processor.process_audio(
            audio_path=audio_file,
            callback=new_result_callback,
            num_speakers=2
        )
        
        # Wait for processing
        import time
        time.sleep(15)
        
        # Stop processing
        processor.stop_processing()
        
        # Verify results
        assert len(new_results) > 0
        
        # Check if speakers are matched consistently
        speaker_ids = set(r.speaker_id for r in new_results)
        assert len(speaker_ids) <= 2  # Should detect at most 2 speakers
        
        logger.info("Speaker matching test passed")
    except Exception as e:
        logger.error(f"Speaker matching test failed: {str(e)}")
        raise

@pytest.mark.dependency(name="test_context_persistence", depends=["test_speaker_matching"])
def test_context_persistence(processor):
    """Test speaker context persistence"""
    try:
        # Get all speakers
        speakers = processor.context_manager.get_all_speakers()
        assert len(speakers) > 0
        
        # Save contexts
        context_file = os.path.join(TEST_DATA_DIR, "speaker_contexts.json")
        processor.save_contexts(context_file)
        assert os.path.exists(context_file)
        
        # Create new processor
        new_processor = RealTimeProcessor(
            model_dir="models/nemo",
            storage_dir=os.path.join(TEST_DATA_DIR, "speaker_contexts"),
            use_vad=True,
            use_enhancement=True
        )
        
        # Load contexts
        new_processor.load_contexts(context_file)
        
        # Verify contexts are loaded
        new_speakers = new_processor.context_manager.get_all_speakers()
        assert len(new_speakers) == len(speakers)
        
        logger.info("Context persistence test passed")
    except Exception as e:
        logger.error(f"Context persistence test failed: {str(e)}")
        raise

def main():
    """Run all tests"""
    try:
        # Run pytest
        pytest.main([__file__, "-v"])
        
    except Exception as e:
        logger.error(f"Tests failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 