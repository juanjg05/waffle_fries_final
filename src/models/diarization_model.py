import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
import logging
from datetime import datetime
import os
from utils.speaker_context_manager import SpeakerContextManager
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import traceback
from sklearn.cluster import KMeans
from collections import defaultdict

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
                 model_dir: str = "models",
                 storage_dir: str = "data/speaker_contexts",
                 min_speaker_duration: float = 0.5,
                 max_speaker_duration: float = 10.0,
                 silence_threshold: float = 0.5,
                 similarity_threshold: float = 0.85,
                 min_speaker_samples: int = 3,
                 use_vad: bool = True,
                 use_enhancement: bool = True,
                 n_clusters: int = 2):
        """
        Initialize real-time audio processor with PyAnnote and Whisper.
        
        Args:
            model_dir: Directory containing models
            storage_dir: Directory for speaker context storage
            min_speaker_duration: Minimum duration for a speaker segment
            max_speaker_duration: Maximum duration for a speaker segment
            silence_threshold: Threshold for silence detection
            similarity_threshold: Threshold for speaker similarity matching
            min_speaker_samples: Minimum number of samples needed for reliable speaker matching
            use_vad: Whether to use Voice Activity Detection
            use_enhancement: Whether to use audio enhancement
            n_clusters: Number of speaker clusters to maintain
        """
        self.model_dir = model_dir
        self.min_speaker_duration = min_speaker_duration
        self.max_speaker_duration = max_speaker_duration
        self.silence_threshold = silence_threshold
        self.similarity_threshold = similarity_threshold
        self.min_speaker_samples = min_speaker_samples
        self.use_vad = use_vad
        self.use_enhancement = use_enhancement
        self.n_clusters = n_clusters
        
        # Initialize clustering
        self.kmeans = None
        self.cluster_centers = []
        self.cluster_to_speaker = {}
        self.speaker_embeddings_list = []
        self.speaker_ids_list = []
        self.conversation_index = 0  # Track conversation number
        
        # Initialize speaker context manager
        self.context_manager = SpeakerContextManager(storage_dir=storage_dir)
        
        # Initialize models
        try:
            # Get Hugging Face token
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError(
                    "Hugging Face token not found. Please set the HF_TOKEN environment variable. "
                    "You can get a token from https://hf.co/settings/tokens and accept the user "
                    "conditions at https://hf.co/pyannote/speaker-diarization"
                )
            
            # Load PyAnnote diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=hf_token
            )
            
            # Load Whisper ASR model
            self.asr_model = whisper.load_model("base")
            
            # Load speaker embedding model
            self.embedding_model = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            
            logger.info("Successfully loaded PyAnnote and Whisper models")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
        
        # Initialize processing queue and thread
        self.processing_queue = []
        self.is_processing = False
        self.processing_thread = None
        
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
            import librosa
            
            # Load audio file
            audio_signal, sample_rate = sf.read(audio_path)
            
            # Handle mono/stereo
            if len(audio_signal.shape) > 1:
                audio_signal = audio_signal[:, 0]  # Take first channel if stereo
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                logger.info(f"Resampling audio from {sample_rate}Hz to 16000Hz")
                audio_signal = librosa.resample(
                    y=audio_signal,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
                sample_rate = 16000
            
            # Apply noise reduction if enabled
            if self.use_enhancement:
                import noisereduce as nr
                logger.info("Applying noise reduction")
                audio_signal = nr.reduce_noise(y=audio_signal, sr=sample_rate)
            
            # Save processed audio
            processed_path = audio_path.replace('.wav', '_enhanced.wav')
            sf.write(processed_path, audio_signal, sample_rate)
            logger.info(f"Saved processed audio to {processed_path}")
            
            return processed_path
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return audio_path

    def _get_speaker_embedding(self, audio_segment: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, float]:
        """Extract speaker embedding from audio segment"""
        try:
            # Ensure audio is mono
            if len(audio_segment.shape) > 1:
                audio_segment = audio_segment.mean(axis=1)
            
            # Pad short segments
            min_duration = 0.5  # seconds
            min_samples = int(min_duration * sample_rate)
            if len(audio_segment) < min_samples:
                audio_segment = np.pad(audio_segment, (0, min_samples - len(audio_segment)))
            
            # Convert to tensor and add batch and channel dimensions
            waveform = torch.from_numpy(audio_segment).float()
            waveform = waveform.unsqueeze(0)  # Add batch dimension
            waveform = waveform.unsqueeze(0)  # Add channel dimension
            
            # Extract embedding
            embedding = self.embedding_model(waveform)
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding, 1.0  # Return normalized embedding and confidence
            
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, 0.0

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

    def _update_clusters(self):
        """Update speaker clusters using k-means"""
        if len(self.speaker_embeddings_list) < self.n_clusters:
            return
            
        # Convert embeddings to numpy array
        X = np.array(self.speaker_embeddings_list)
        
        # Fit k-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(X)
        
        # Update cluster centers
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Map clusters to speakers
        cluster_speakers = defaultdict(list)
        for embedding, speaker_id, cluster in zip(
            self.speaker_embeddings_list,
            self.speaker_ids_list,
            self.kmeans.labels_
        ):
            cluster_speakers[cluster].append(speaker_id)
        
        # Assign most common speaker ID to each cluster
        for cluster, speakers in cluster_speakers.items():
            from collections import Counter
            most_common = Counter(speakers).most_common(1)[0][0]
            self.cluster_to_speaker[cluster] = most_common
            
        logger.info(f"Updated clusters: {self.cluster_to_speaker}")

    def _find_matching_speaker(self, embedding: np.ndarray) -> str:
        """
        Find matching speaker for embedding using k-means clustering.
        
        Args:
            embedding: Speaker embedding vector
            
        Returns:
            Speaker ID of best match
        """
        try:
            # Add to embeddings list
            self.speaker_embeddings_list.append(embedding)
            
            # If we have enough embeddings, use clustering
            if len(self.speaker_embeddings_list) >= self.n_clusters:
                if self.kmeans is None or len(self.speaker_embeddings_list) % 10 == 0:
                    # Update clusters periodically
                    self._update_clusters()
                
                # Predict cluster
                cluster = self.kmeans.predict([embedding])[0]
                
                # Get speaker ID for cluster
                if cluster in self.cluster_to_speaker:
                    speaker_id = self.cluster_to_speaker[cluster]
                    logger.info(f"Matched to cluster {cluster} (speaker {speaker_id})")
                    self.speaker_ids_list.append(speaker_id)
                    return speaker_id
            
            # Fallback to similarity matching if clustering not ready
            best_match = None
            best_score = -1
            
            # Compare with known speakers
            for speaker_id, known_embedding in self.speaker_embeddings.items():
                score = self._compute_similarity(embedding, known_embedding)
                logger.info(f"Similarity score with {speaker_id}: {score:.3f}")
                if score > best_score and score > self.similarity_threshold:
                    best_score = score
                    best_match = speaker_id
            
            if best_match is None:
                # Create new speaker ID
                best_match = f"SPEAKER_{len(self.speaker_embeddings) + 1:02d}"  # Use 2-digit format
                self.speaker_embeddings[best_match] = embedding
                logger.info(f"Created new speaker: {best_match}")
            else:
                logger.info(f"Matched with existing speaker: {best_match} (score: {best_score:.3f})")
            
            # Add to lists for clustering
            self.speaker_ids_list.append(best_match)
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding matching speaker: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"SPEAKER_{len(self.speaker_embeddings) + 1:02d}"

    def _process_segment(self, segment: dict) -> Optional[DiarizationResult]:
        """
        Process a single audio segment.
        
        Args:
            segment: Dictionary containing segment information
            
        Returns:
            DiarizationResult if successful, None otherwise
        """
        try:
            # Extract audio segment
            audio_segment = segment['audio']
            sample_rate = segment['sample_rate']
            
            # Get speaker embedding
            embedding, confidence = self._get_speaker_embedding(audio_segment, sample_rate)
            if embedding is None:
                return None
                
            # Find matching speaker
            speaker_id = self._find_matching_speaker(embedding)
            
            # Create result
            result = DiarizationResult(
                speaker_id=speaker_id,
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                transcript=segment.get('transcript', ''),
                confidence=confidence,
                speaker_embedding=embedding,
                speaker_confidence=confidence
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing segment: {str(e)}")
            return None

    def start_processing(self):
        """Start the processing thread"""
        self.is_processing = True
        import threading
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.start()

    def stop_processing(self):
        """Stop the processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()

    def _process_queue(self):
        """Process queued audio segments"""
        while self.is_processing:
            if self.processing_queue:
                segment = self.processing_queue.pop(0)
                result = self._process_segment(segment)
                if result:
                    self.result_callback(result)
            else:
                import time
                time.sleep(0.1)  # Sleep to prevent busy waiting

    def process_audio(self, audio_path: str, callback: Callable[[DiarizationResult], None], num_speakers: Optional[int] = None):
        """
        Process audio file and call callback with results.
        
        Args:
            audio_path: Path to audio file
            callback: Function to call with results
            num_speakers: Optional number of speakers to detect
        """
        try:
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio_path)
            
            # Run diarization
            diarization = self.diarization_pipeline(processed_audio, num_speakers=num_speakers)
            
            # Process results
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Create segment
                segment = {
                    'audio_path': processed_audio,
                    'sample_rate': 16000,
                    'start_time': turn.start,
                    'end_time': turn.end,
                    'speaker_id': speaker
                }
                
                # Process segment
                result = self._process_segment(segment)
                if result:
                    callback(result)
                    
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise

    def get_speaker_history(self, speaker_id: str) -> List[Dict]:
        """Get conversation history for a speaker"""
        return self.context_manager.get_speaker_history(speaker_id)

    def get_speaker_stats(self, speaker_id: str) -> Dict:
        """Get statistics for a speaker"""
        return self.context_manager.get_speaker_stats(speaker_id)

    def save_contexts(self, filepath: str):
        """Save speaker contexts to file"""
        self.context_manager.save_contexts(filepath)

    def load_contexts(self, filepath: str):
        """Load speaker contexts from file"""
        self.context_manager.load_contexts(filepath)
        self._load_speaker_embeddings()

    def result_callback(self, result: DiarizationResult):
        """
        Callback function for processing results.
        Override this in subclasses to handle results.
        
        Args:
            result: DiarizationResult object
        """
        logger.info(f"Processed segment for speaker {result.speaker_id}")
        logger.info(f"Time: {result.start_time:.2f} - {result.end_time:.2f}")
        if result.transcript:
            logger.info(f"Transcript: {result.transcript}")

def diarize_speech(audio_file: str) -> List[DiarizationResult]:
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        List of DiarizationResult objects
    """
    try:
        # Initialize processor
        processor = RealTimeProcessor()
        
        # Preprocess audio
        processed_audio = processor._preprocess_audio(audio_file)
        
        # Load audio
        import soundfile as sf
        audio_data, sample_rate = sf.read(processed_audio)
        
        # Run diarization
        diarization = processor.diarization_pipeline(processed_audio)
        
        # Process results
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Extract audio segment
            start_sample = int(turn.start * sample_rate)
            end_sample = int(turn.end * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            # Create segment
            segment = {
                'audio': segment_audio,
                'sample_rate': sample_rate,
                'start_time': turn.start,
                'end_time': turn.end,
                'speaker_id': speaker
            }
            
            # Process segment
            result = processor._process_segment(segment)
            if result:
                results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in diarize_speech: {str(e)}")
        raise

def combine_diarization_with_transcript(
    diarization_results: List[DiarizationResult],
    transcript: str,
    word_timestamps: List[Tuple[str, float, float]]
) -> List[DiarizationResult]:
    """
    Combine diarization results with transcript and word timestamps.
    
    Args:
        diarization_results: List of DiarizationResult objects
        transcript: Full transcript text
        word_timestamps: List of (word, start_time, end_time) tuples
        
    Returns:
        List of DiarizationResult objects with transcripts
    """
    try:
        # Sort diarization results by start time
        diarization_results.sort(key=lambda x: x.start_time)
        
        # Assign words to segments
        current_segment_idx = 0
        current_segment = diarization_results[current_segment_idx]
        segment_transcript = []
        
        for word, start_time, end_time in word_timestamps:
            # Find appropriate segment
            while (current_segment_idx < len(diarization_results) - 1 and
                   end_time > diarization_results[current_segment_idx + 1].start_time):
                current_segment_idx += 1
                current_segment = diarization_results[current_segment_idx]
                segment_transcript = []
            
            # Add word to current segment
            if start_time >= current_segment.start_time and end_time <= current_segment.end_time:
                segment_transcript.append(word)
            
            # Update segment transcript
            current_segment.transcript = ' '.join(segment_transcript)
        
        return diarization_results
        
    except Exception as e:
        logger.error(f"Error combining diarization with transcript: {str(e)}")
        raise

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
