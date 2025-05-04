import pytest
import os
import logging
import numpy as np
import soundfile as sf
from models.diarization_model import RealTimeProcessor, DiarizationResult
from utils.memory import SpeakerMemory
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

@pytest.fixture(scope="session")
def audio_file():
    """Fixture to get the test audio file path"""
        audio_path = os.path.join(TEST_DATA_DIR, "record_out.wav")
    assert os.path.exists(audio_path), f"Test audio file not found at {audio_path}"
        return audio_path

@pytest.fixture(scope="session")
def processor():
    """Fixture to create and return a RealTimeProcessor instance"""
    try:
        processor = RealTimeProcessor(
            model_dir="models",
            storage_dir=os.path.join(TEST_DATA_DIR, "speaker_contexts"),
            use_vad=True,
            use_enhancement=True
        )
        assert processor is not None
        assert processor.diarization_pipeline is not None
        assert processor.asr_model is not None
        assert processor.embedding_model is not None
        logger.info("Processor initialization test passed")
        return processor
    except Exception as e:
        logger.error(f"Processor initialization test failed: {str(e)}")
        raise

def test_processor_initialization(processor):
    """Test processor initialization"""
    assert processor is not None
    assert processor.diarization_pipeline is not None
    assert processor.asr_model is not None
    assert processor.embedding_model is not None
    logger.info("Processor initialization test passed")

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

def test_speaker_diarization(processor, audio_file):
    """Test speaker diarization with embedding and transcript saving"""
    try:
        results = []
        
        def result_callback(result: DiarizationResult):
            results.append(result)
            logger.info(f"Detected speaker {result.speaker_id}:")
            logger.info(f"  Time: {result.start_time:.2f} - {result.end_time:.2f}")
            logger.info(f"  Transcript: {result.transcript}")
            logger.info(f"  Confidence: {result.confidence:.2f}")
            logger.info(f"  Speaker Confidence: {result.speaker_confidence:.2f}")
            
            # Save speaker embedding and transcript
            if result.speaker_embedding is not None:
                embedding_path = os.path.join(TEST_DATA_DIR, f"speaker_{result.speaker_id}_embedding.npy")
                np.save(embedding_path, result.speaker_embedding)
                logger.info(f"Saved speaker embedding to {embedding_path}")
            
            # Save transcript
            transcript_path = os.path.join(TEST_DATA_DIR, f"speaker_{result.speaker_id}_transcript.txt")
            with open(transcript_path, "a") as f:
                f.write(f"[{result.start_time:.2f}-{result.end_time:.2f}] {result.transcript}\n")
            logger.info(f"Saved transcript to {transcript_path}")
        
        # Start processing
        logger.info("Starting audio processing...")
        processor.start_processing()
        
        # Process audio
        logger.info(f"Processing audio file: {audio_file}")
        processor.process_audio(
            audio_path=audio_file,
            callback=result_callback,
            num_speakers=None  # Let the model determine number of speakers
        )
        
        # Wait for processing to complete
        logger.info("Waiting for processing to complete...")
        import time
        time.sleep(15)  # Wait for processing
        
        # Stop processing
        logger.info("Stopping processing...")
        processor.stop_processing()
        
        # Verify results
        logger.info(f"Number of results: {len(results)}")
        if len(results) == 0:
            logger.error("No diarization results were produced")
            raise AssertionError("No diarization results were produced")
            
        assert all(isinstance(r, DiarizationResult) for r in results)
        assert all(r.speaker_embedding is not None for r in results)
        
        # Check speaker consistency
        speaker_ids = set(r.speaker_id for r in results)
        logger.info(f"Detected {len(speaker_ids)} unique speakers")
        
        # Verify saved files
        for speaker_id in speaker_ids:
            embedding_path = os.path.join(TEST_DATA_DIR, f"speaker_{speaker_id}_embedding.npy")
            transcript_path = os.path.join(TEST_DATA_DIR, f"speaker_{speaker_id}_transcript.txt")
            
            assert os.path.exists(embedding_path), f"Speaker embedding not saved for {speaker_id}"
            assert os.path.exists(transcript_path), f"Transcript not saved for {speaker_id}"
            
            # Load and verify embedding
            embedding = np.load(embedding_path)
            assert embedding.shape[0] > 0, f"Empty embedding for {speaker_id}"
            
            # Load and verify transcript
            with open(transcript_path, "r") as f:
                transcript = f.read()
            assert len(transcript) > 0, f"Empty transcript for {speaker_id}"
        
        logger.info("Speaker diarization test passed")
        return results
    except Exception as e:
        logger.error(f"Speaker diarization test failed: {str(e)}")
        raise

def test_speaker_matching(processor):
    """Test speaker matching across multiple recordings"""
    try:
        # Process first recording
        results1 = []
        def callback1(result):
            results1.append(result)
            logger.info(f"Recording 1 - Speaker {result.speaker_id}: {result.transcript}")
            logger.info(f"Time: {result.start_time:.2f} - {result.end_time:.2f}")
        
        logger.info("Processing first recording...")
        processor.process_audio(
            os.path.join(TEST_DATA_DIR, "record_out.wav"),
            callback=callback1
        )
        
        # Get speaker IDs from first recording
        speaker_ids1 = set(r.speaker_id for r in results1)
        logger.info(f"Speakers in recording 1: {speaker_ids1}")
        
        # Verify transcripts from first recording
        for result in results1:
            assert result.transcript, f"Empty transcript for speaker {result.speaker_id} in recording 1"
            logger.info(f"Transcript for {result.speaker_id}: {result.transcript}")
        
        # Process second recording
        results2 = []
        def callback2(result):
            results2.append(result)
            logger.info(f"Recording 2 - Speaker {result.speaker_id}: {result.transcript}")
            logger.info(f"Time: {result.start_time:.2f} - {result.end_time:.2f}")
        
        logger.info("Processing second recording...")
        processor.process_audio(
            os.path.join(TEST_DATA_DIR, "recording_2.wav"),
            callback=callback2
        )
        
        # Get speaker IDs from second recording
        speaker_ids2 = set(r.speaker_id for r in results2)
        logger.info(f"Speakers in recording 2: {speaker_ids2}")
        
        # Verify transcripts from second recording
        for result in results2:
            assert result.transcript, f"Empty transcript for speaker {result.speaker_id} in recording 2"
            logger.info(f"Transcript for {result.speaker_id}: {result.transcript}")
        
        # Check that we have results
        assert len(results1) > 0, "No results from first recording"
        assert len(results2) > 0, "No results from second recording"
        
        # Check that speakers were matched across recordings
        common_speakers = speaker_ids1.intersection(speaker_ids2)
        logger.info(f"Common speakers across recordings: {common_speakers}")
        assert len(common_speakers) > 0, "No speakers matched across recordings"
        
        # Check speaker contexts
        for speaker_id in common_speakers:
            history = processor.get_speaker_history(speaker_id)
            logger.info(f"History for speaker {speaker_id}: {len(history)} entries")
            assert len(history) > 1, f"Speaker {speaker_id} should have multiple entries"
            
            # Verify transcripts in history
            for entry in history:
                assert entry['transcript'], f"Empty transcript in history for speaker {speaker_id}"
                logger.info(f"History entry for {speaker_id}: {entry['transcript']}")
        
        # Save contexts to verify persistence
        contexts_file = os.path.join(TEST_DATA_DIR, "speaker_contexts.json")
        processor.save_contexts(contexts_file)
        logger.info(f"Saved contexts to {contexts_file}")
        
        return results1, results2
        
    except Exception as e:
        logger.error(f"Speaker matching test failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def test_context_persistence(processor):
    """Test speaker context persistence"""
    try:
        # Save contexts
        contexts_file = os.path.join(TEST_DATA_DIR, "speaker_contexts.json")
        processor.save_contexts(contexts_file)
        assert os.path.exists(contexts_file)
        
        # Load contexts
        processor.load_contexts(contexts_file)
        
        # Verify speaker embeddings were loaded
        assert len(processor.speaker_embeddings) > 0
        
        logger.info("Context persistence test passed")
    except Exception as e:
        logger.error(f"Context persistence test failed: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main([__file__]) 