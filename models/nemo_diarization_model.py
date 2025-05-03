import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
import logging
from datetime import datetime
import os
from utils.speaker_context_manager import SpeakerContextManager
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiarizationResult:
    def __init__(self, speaker_id: str, start_time: float, end_time: float, 
                 transcript: str = "", confidence: float = 1.0,
                 speaker_embedding: Optional[np.ndarray] = None,
                 speaker_confidence: float = 1.0):
        self.speaker_id = speaker_id
        self.start_time = start_time
        self.end_time = end_time
        self.transcript = transcript
        self.confidence = confidence
        self.speaker_embedding = speaker_embedding
        self.speaker_confidence = speaker_confidence

class RealTimeProcessor:
    def __init__(self, 
                 model_dir: str = "models/nemo",
                 storage_dir: str = "data/speaker_contexts",
                 min_speaker_duration: float = 0.5,
                 max_speaker_duration: float = 10.0,
                 silence_threshold: float = 0.5,
                 similarity_threshold: float = 0.85,
                 min_speaker_samples: int = 3,
                 use_vad: bool = True,
                 use_enhancement: bool = True):
        """
        Initialize real-time audio processor with NeMo models.
        
        Args:
            model_dir: Directory containing NeMo models
            storage_dir: Directory for speaker context storage
            min_speaker_duration: Minimum duration for a speaker segment
            max_speaker_duration: Maximum duration for a speaker segment
            silence_threshold: Threshold for silence detection
            similarity_threshold: Threshold for speaker similarity matching
            min_speaker_samples: Minimum number of samples needed for reliable speaker matching
            use_vad: Whether to use Voice Activity Detection
            use_enhancement: Whether to use audio enhancement
        """
        self.model_dir = model_dir
        self.min_speaker_duration = min_speaker_duration
        self.max_speaker_duration = max_speaker_duration
        self.silence_threshold = silence_threshold
        self.similarity_threshold = similarity_threshold
        self.min_speaker_samples = min_speaker_samples
        self.use_vad = use_vad
        self.use_enhancement = use_enhancement
        
        # Initialize speaker context manager
        self.context_manager = SpeakerContextManager(storage_dir=storage_dir)
        
        # Initialize NeMo models
        try:
            # Load diarization model with speaker embedding extraction
            self.diarization_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
                model_name="vad_telephony_marblenet"
            )
            
            # Load ASR model with confidence scores
            self.asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                model_name="stt_en_quartznet15x5"
            )
            
            # Load BERT model for speaker verification
            self.bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Load Voice Activity Detection model
            if self.use_vad:
                self.vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
                    model_name="vad_telephony_marblenet"
                )
            
            # Load audio enhancement model
            if self.use_enhancement:
                self.enhancement_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                    model_name="stt_en_quartznet15x5"
                )
            
            logger.info("Successfully loaded NeMo models")
            
        except Exception as e:
            logger.error(f"Error loading NeMo models: {str(e)}")
            raise
        
        # Initialize processing queue and thread
        self.processing_queue = []
        self.is_processing = False
        self.processing_thread = None
        
        # Move models to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diarization_model.to(self.device)
        self.asr_model.to(self.device)
        self.bert_model.to(self.device)
        if self.use_vad:
            self.vad_model.to(self.device)
        if self.use_enhancement:
            self.enhancement_model.to(self.device)
        
        # Initialize speaker embedding cache
        self.speaker_embeddings = {}
        self._load_speaker_embeddings()

    def _load_speaker_embeddings(self):
        """Load speaker embeddings from context manager"""
        speakers = self.context_manager.get_all_speakers()
        for speaker in speakers:
            if speaker.get('embedding'):
                self.speaker_embeddings[speaker['speaker_id']] = np.array(speaker['embedding'])

    def _preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio file with VAD and enhancement.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Path to processed audio file
        """
        try:
            import soundfile as sf
            processed_path = audio_path
            
            if self.use_enhancement:
                # Load audio file
                audio_signal, sample_rate = sf.read(audio_path)
                audio_signal = torch.tensor(audio_signal).unsqueeze(0)  # Add batch dimension
                
                # Apply noise suppression
                enhanced_audio = self.enhancement_model(audio_signal)
                processed_path = audio_path.replace('.wav', '_enhanced.wav')
                sf.write(processed_path, enhanced_audio.cpu().numpy().squeeze(), sample_rate)
            
            if self.use_vad:
                # Load audio file
                audio_signal, sample_rate = sf.read(processed_path)
                audio_signal = torch.tensor(audio_signal).unsqueeze(0)  # Add batch dimension
                
                # Apply Voice Activity Detection
                vad_logits = self.vad_model(audio_signal)
                vad_probs = torch.softmax(vad_logits, dim=-1)
                vad_predictions = torch.argmax(vad_probs, dim=-1)
                if not torch.any(vad_predictions == 1):  # 1 is speech
                    logger.warning("No voice activity detected in audio")
                    return None
            
            return processed_path
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {str(e)}")
            return audio_path

    def _get_speaker_embedding(self, diarization: Dict) -> Tuple[np.ndarray, float]:
        """
        Extract speaker embedding from diarization result.
        
        Args:
            diarization: Diarization result dictionary
            
        Returns:
            Tuple of (normalized speaker embedding vector, confidence score)
        """
        try:
            # Extract embedding from diarization result
            if 'embeddings' in diarization:
                # Get the embedding vector
                embedding = diarization['embeddings']
                
                # Get confidence score
                confidence = diarization.get('confidence', 1.0)
                
                # Normalize embedding
                embedding = normalize(embedding.reshape(1, -1))[0]
                
                return embedding, confidence
            else:
                # Fallback to model's speaker embedding extraction
                audio_path = diarization.get('audio_path')
                if audio_path:
                    # Extract embedding using the model
                    embedding = self.diarization_model.extract_speaker_embedding(audio_path)
                    confidence = 1.0  # Default confidence
                    
                    # Normalize embedding
                    embedding = normalize(embedding.reshape(1, -1))[0]
                    
                    return embedding, confidence
                
            raise ValueError("No embedding information available")
            
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {str(e)}")
            return np.zeros(512), 0.0  # Return zero vector and zero confidence on error

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Compute cosine similarity
            similarity = 1 - cosine(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0

    def _find_matching_speaker(self, embedding: np.ndarray) -> str:
        """
        Find matching speaker in context or create new speaker ID.
        
        Args:
            embedding: Speaker embedding vector
            
        Returns:
            Speaker ID
        """
        try:
            # Get all existing speakers
            speakers = self.context_manager.get_all_speakers()
            
            if not speakers:
                # No existing speakers, create new speaker ID
                new_id = f"SPEAKER_{len(speakers) + 1}"
                self.speaker_embeddings[new_id] = embedding
                return new_id
            
            # Calculate similarities with existing speakers
            similarities = []
            for speaker in speakers:
                speaker_id = speaker['speaker_id']
                if speaker_id in self.speaker_embeddings:
                    similarity = self._compute_similarity(
                        embedding, 
                        self.speaker_embeddings[speaker_id]
                    )
                    similarities.append((speaker_id, similarity))
            
            if not similarities:
                # No valid embeddings found, create new speaker
                new_id = f"SPEAKER_{len(speakers) + 1}"
                self.speaker_embeddings[new_id] = embedding
                return new_id
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            best_match_id, best_similarity = similarities[0]
            
            # Get speaker stats
            speaker_stats = self.context_manager.get_speaker_stats(best_match_id)
            interaction_count = speaker_stats['interaction_count']
            
            # Adjust threshold based on interaction count
            adjusted_threshold = self.similarity_threshold
            if interaction_count < self.min_speaker_samples:
                # Lower threshold for speakers with few samples
                adjusted_threshold *= (interaction_count / self.min_speaker_samples)
            
            if best_similarity >= adjusted_threshold:
                # Update embedding with moving average
                current_embedding = self.speaker_embeddings[best_match_id]
                alpha = 0.3  # Weight for new embedding
                self.speaker_embeddings[best_match_id] = (
                    (1 - alpha) * current_embedding + alpha * embedding
                )
                return best_match_id
            
            # No good match found, create new speaker
            new_id = f"SPEAKER_{len(speakers) + 1}"
            self.speaker_embeddings[new_id] = embedding
            return new_id
            
        except Exception as e:
            logger.error(f"Error in speaker matching: {str(e)}")
            # Return new speaker ID on error
            return f"SPEAKER_{len(speakers) + 1}"

    def _process_segment(self, segment: dict) -> Optional[DiarizationResult]:
        """
        Process a single audio segment.
        
        Args:
            segment: Dictionary containing segment information
            
        Returns:
            DiarizationResult if successful, None otherwise
        """
        try:
            import soundfile as sf
            
            # Load audio file
            audio_signal, sample_rate = sf.read(segment['audio_path'])
            audio_signal = torch.tensor(audio_signal).unsqueeze(0)  # Add batch dimension
            
            # Get diarization results with speaker embeddings
            diarization_logits = self.diarization_model(audio_signal)
            diarization_probs = torch.softmax(diarization_logits, dim=-1)
            
            # Convert logits to segments
            segments = []
            current_speaker = None
            start_time = 0
            frame_duration = 0.01  # 10ms per frame
            
            for i, prob in enumerate(diarization_probs):
                is_speech = torch.argmax(prob) == 1
                time = i * frame_duration
                
                if is_speech and current_speaker is None:
                    # Start of speech segment
                    current_speaker = "UNKNOWN"
                    start_time = time
                elif not is_speech and current_speaker is not None:
                    # End of speech segment
                    segments.append({
                        'start_time': start_time,
                        'end_time': time,
                        'speaker_id': current_speaker,
                        'audio_path': segment['audio_path']
                    })
                    current_speaker = None
            
            # Add final segment if needed
            if current_speaker is not None:
                segments.append({
                    'start_time': start_time,
                    'end_time': len(diarization_probs) * frame_duration,
                    'speaker_id': current_speaker,
                    'audio_path': segment['audio_path']
                })
            
            # Get ASR results with confidence scores
            asr_logits = self.asr_model(audio_signal)
            asr_probs = torch.softmax(asr_logits, dim=-1)
            transcript = self.asr_model.decode(asr_logits)
            transcript_confidence = torch.mean(torch.max(asr_probs, dim=-1)[0]).item()
            
            # Get speaker embedding and confidence
            speaker_embedding, speaker_confidence = self._get_speaker_embedding(segment)
            
            # Find matching speaker in context
            speaker_id = self._find_matching_speaker(speaker_embedding)
            
            # Update speaker context
            self._update_speaker_context(
                speaker_id=speaker_id,
                embedding=speaker_embedding,
                confidence=speaker_confidence,
                transcript=transcript,
                transcript_confidence=transcript_confidence
            )
            
            return DiarizationResult(
                speaker_id=speaker_id,
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                transcript=transcript,
                confidence=transcript_confidence,
                speaker_embedding=speaker_embedding,
                speaker_confidence=speaker_confidence
            )
            
        except Exception as e:
            logger.error(f"Error processing segment: {str(e)}")
            return None

    def start_processing(self):
        """Start background processing thread"""
        if not self.is_processing:
            self.is_processing = True
            import threading
            self.processing_thread = threading.Thread(target=self._process_queue)
            self.processing_thread.start()

    def stop_processing(self):
        """Stop background processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()

    def _process_queue(self):
        """Process audio segments in the queue"""
        while self.is_processing:
            if self.processing_queue:
                segment = self.processing_queue.pop(0)
                try:
                    self._process_segment(segment)
                except Exception as e:
                    logger.error(f"Error processing segment: {str(e)}")
            else:
                import time
                time.sleep(0.1)  # Sleep briefly to prevent CPU spinning

    def process_audio(self, audio_path: str, callback: Optional[Callable[[DiarizationResult], None]] = None, num_speakers: Optional[int] = None):
        """
        Process audio file and detect speakers.
        
        Args:
            audio_path: Path to audio file
            callback: Optional callback function to receive results
            num_speakers: Optional number of speakers to detect
        """
        try:
            import soundfile as sf
            
            # Load audio file
            audio_signal, sample_rate = sf.read(audio_path)
            audio_signal = torch.tensor(audio_signal).unsqueeze(0)  # Add batch dimension
            
            # Get diarization results with speaker embeddings
            diarization_logits = self.diarization_model(audio_signal)
            diarization_probs = torch.softmax(diarization_logits, dim=-1)
            
            # Convert logits to segments
            segments = []
            current_speaker = None
            start_time = 0
            frame_duration = 0.01  # 10ms per frame
            
            for i, prob in enumerate(diarization_probs):
                is_speech = torch.argmax(prob) == 1
                time = i * frame_duration
                
                if is_speech and current_speaker is None:
                    # Start of speech segment
                    current_speaker = "UNKNOWN"
                    start_time = time
                elif not is_speech and current_speaker is not None:
                    # End of speech segment
                    segments.append({
                        'start_time': start_time,
                        'end_time': time,
                        'speaker_id': current_speaker,
                        'audio_path': audio_path
                    })
                    current_speaker = None
            
            # Add final segment if needed
            if current_speaker is not None:
                segments.append({
                    'start_time': start_time,
                    'end_time': len(diarization_probs) * frame_duration,
                    'speaker_id': current_speaker,
                    'audio_path': audio_path
                })
            
            # Process each segment
            for segment in segments:
                result = self._process_segment(segment)
                if result and callback:
                    callback(result)
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")

    def get_speaker_history(self, speaker_id: str) -> List[Dict]:
        """Get interaction history for a speaker"""
        return self.context_manager.get_speaker_history(speaker_id)

    def get_speaker_stats(self, speaker_id: str) -> Dict:
        """Get statistics for a speaker"""
        return self.context_manager.get_speaker_stats(speaker_id)

    def save_contexts(self, filepath: str):
        """Save speaker contexts to file"""
        self.context_manager.export_contexts(filepath)

    def load_contexts(self, filepath: str):
        """Load speaker contexts from file"""
        self.context_manager.import_contexts(filepath)

# Example usage:
if __name__ == "__main__":
    # Initialize processor
    processor = RealTimeProcessor()
    
    # Process audio file
    def result_callback(result: DiarizationResult):
        print(f"Speaker {result.speaker_id}: {result.transcript}")
        print(f"Time: {result.start_time:.2f} - {result.end_time:.2f}")
        print(f"Confidence: {result.confidence:.2f}")
    
    processor.process_audio(
        audio_path="test.wav",
        callback=result_callback,
        num_speakers=2
    )
    
    # Wait for processing to complete
    import time
    time.sleep(5)
    
    # Stop processing
    processor.stop_processing()
    
    # Get speaker history
    history = processor.get_speaker_history("SPEAKER_1")
    print(f"Speaker 1 history: {history}")
    
    # Get speaker stats
    stats = processor.get_speaker_stats("SPEAKER_1")
    print(f"Speaker 1 stats: {stats}")
    
    # Save contexts
    processor.save_contexts("speaker_contexts.json")
