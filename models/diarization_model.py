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
            
            # Fallback to similarity matching
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
        Process a single diarization segment.
        
        Args:
            segment: Diarization segment dictionary
            
        Returns:
            DiarizationResult if successful, None otherwise
        """
        try:
            # Extract segment info
            start_time = segment['start']
            end_time = segment['end']
            speaker_id = segment['speaker']
            
            # Get speaker embedding
            embedding, confidence = self._get_speaker_embedding(
                segment['audio_path'],
                segment['sample_rate']
            )
            
            if embedding is None:
                return None
            
            # Find matching speaker
            matched_speaker = self._find_matching_speaker(embedding)
            
            # Transcribe segment
            transcript = self.asr_model.transcribe(
                segment['audio_path'],
                start=start_time,
                end=end_time
            )['text']
            
            return DiarizationResult(
                speaker_id=matched_speaker,
                start_time=start_time,
                end_time=end_time,
                transcript=transcript,
                confidence=confidence,
                speaker_embedding=embedding,
                speaker_confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error processing segment: {str(e)}")
            return None

    def start_processing(self):
        """Start processing thread"""
        self.is_processing = True
        import threading
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.start()

    def stop_processing(self):
        """Stop processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()

    def _process_queue(self):
        """Process items in queue"""
        while self.is_processing:
            if self.processing_queue:
                item = self.processing_queue.pop(0)
                self.process_audio(**item)
            import time
            time.sleep(0.1)

    def process_audio(self, audio_path: str, callback: Callable[[DiarizationResult], None], num_speakers: Optional[int] = None):
        """Process audio file for speaker diarization"""
        try:
            # Increment conversation index
            self.conversation_index += 1
            logger.info(f"Starting conversation {self.conversation_index}")
            
            # Preprocess audio
            processed_path = self._preprocess_audio(audio_path)
            logger.info(f"Preprocessed audio saved to: {processed_path}")
            
            # Load audio for diarization
            import soundfile as sf
            audio, sample_rate = sf.read(processed_path)
            logger.info(f"Loaded audio: {audio.shape}, {sample_rate}Hz")
            
            # Convert audio to PyAnnote format (channel, time)
            if len(audio.shape) == 1:
                # Mono audio - add channel dimension
                audio = audio[np.newaxis, :]
            elif len(audio.shape) == 2:
                # Stereo audio - convert to mono and add channel dimension
                audio = audio.mean(axis=1)[np.newaxis, :]
            
            # Convert to torch tensor
            waveform = torch.from_numpy(audio).float()
            
            # Run diarization pipeline
            logger.info("Running diarization pipeline...")
            diarization = self.diarization_pipeline({
                "waveform": waveform,
                "sample_rate": sample_rate,
                "uri": "stream"
            })
            
            # Collect all segments first
            segments = []
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                start_time = segment.start
                end_time = segment.end
                
                # Skip very short segments
                if end_time - start_time < 0.5:
                    logger.warning(f"Skipping short segment: {start_time:.2f}-{end_time:.2f}")
                    continue
                
                # Extract audio segment
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                audio_segment = audio[0, start_sample:end_sample]  # Take first channel
                
                segments.append({
                    'audio': audio_segment,
                    'start_time': start_time,
                    'end_time': end_time,
                    'speaker': speaker,
                    'sample_rate': sample_rate
                })
            
            logger.info(f"Found {len(segments)} segments")
            
            # Process segments in batches
            batch_size = 8
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(segments)-1)//batch_size + 1}")
                
                # Get embeddings for batch
                embeddings = []
                confidences = []
                for segment in batch:
                    embedding, confidence = self._get_speaker_embedding(segment['audio'], segment['sample_rate'])
                    embeddings.append(embedding)
                    confidences.append(confidence)
                
                # Process each segment in batch
                for segment, embedding, confidence in zip(batch, embeddings, confidences):
                    if embedding is not None:
                        try:
                            # Save segment to temporary file for transcription
                            temp_path = f"temp_segment_{segment['start_time']:.2f}_{segment['end_time']:.2f}.wav"
                            sf.write(temp_path, segment['audio'], segment['sample_rate'])
                            logger.info(f"Saved temp segment to {temp_path}")
                            
                            # Transcribe with Whisper
                            logger.info(f"Transcribing segment {segment['start_time']:.2f}-{segment['end_time']:.2f}")
                            transcript_result = self.asr_model.transcribe(temp_path)
                            transcript = transcript_result['text'].strip()
                            logger.info(f"Transcript: {transcript}")
                            
                            # Clean up temporary file
                            os.remove(temp_path)
                            
                            # Create result
                            result = DiarizationResult(
                                speaker_id=segment['speaker'],
                                start_time=segment['start_time'],
                                end_time=segment['end_time'],
                                transcript=transcript,
                                confidence=confidence,
                                speaker_confidence=confidence,
                                speaker_embedding=embedding
                            )
                            
                            # Call callback
                    callback(result)
                            
                            # Update speaker embeddings and save to JSON
                            self.speaker_embeddings[segment['speaker']] = embedding
                            self.context_manager.update_context(
                                speaker_id=segment['speaker'],
                                start_time=segment['start_time'],
                                end_time=segment['end_time'],
                                confidence=confidence,
                                transcript=transcript,
                                embedding=embedding.tolist(),  # Convert numpy array to list for JSON serialization
                                conversation_index=self.conversation_index  # Add conversation index
                            )
                            logger.info(f"Updated context for speaker {segment['speaker']} in conversation {self.conversation_index}")
                            
                        except Exception as e:
                            logger.error(f"Error processing segment: {str(e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                    else:
                        logger.warning(f"Failed to process segment for speaker {segment['speaker']}")
                
            # Save contexts after processing all segments
            self.context_manager.save()
            logger.info("Saved all speaker contexts")
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def get_speaker_history(self, speaker_id: str) -> List[Dict]:
        """Get speaker history"""
        return self.context_manager.get_speaker_history(speaker_id)

    def get_speaker_stats(self, speaker_id: str) -> Dict:
        """Get speaker statistics"""
        return self.context_manager.get_speaker_stats(speaker_id)

    def save_contexts(self, filepath: str):
        """Save speaker contexts"""
        self.context_manager.export_contexts(filepath)

    def load_contexts(self, filepath: str):
        """Load speaker contexts"""
        self.context_manager.import_contexts(filepath)
        self._load_speaker_embeddings()

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
