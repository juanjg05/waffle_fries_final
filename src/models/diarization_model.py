import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
import os
from datetime import datetime
from utils.speaker_context_manager import SpeakerContextManager
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.cluster import KMeans
from collections import defaultdict

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
        self.model_dir = model_dir
        self.min_speaker_duration = min_speaker_duration
        self.max_speaker_duration = max_speaker_duration
        self.silence_threshold = silence_threshold
        self.similarity_threshold = similarity_threshold
        self.min_speaker_samples = min_speaker_samples
        self.use_vad = use_vad
        self.use_enhancement = use_enhancement
        self.n_clusters = n_clusters
        
        self.kmeans = None
        self.cluster_centers = []
        self.cluster_to_speaker = {}
        self.speaker_embeddings_list = []
        self.speaker_ids_list = []
        self.conversation_index = 0
        
        self.context_manager = SpeakerContextManager(storage_dir=storage_dir)
        
        try:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError(
                    "Hugging Face token not found. Please set the HF_TOKEN environment variable. "
                    "You can get a token from https://hf.co/settings/tokens and accept the user "
                    "conditions at https://hf.co/pyannote/speaker-diarization"
                )
            
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=hf_token
            )
            
            self.asr_model = whisper.load_model("base")
            
            self.embedding_model = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
        
        self.processing_queue = []
        self.is_processing = False
        self.processing_thread = None
        
        self.speaker_embeddings = {}
        self._load_speaker_embeddings()

    def _load_speaker_embeddings(self):
        speakers = self.context_manager.get_all_speakers()
        for speaker in speakers:
            if speaker.get('embedding'):
                self.speaker_embeddings[speaker['speaker_id']] = np.array(speaker['embedding'])

    def _preprocess_audio(self, audio_path: str) -> str:
        try:
            import soundfile as sf
            import librosa
            
            audio_signal, sample_rate = sf.read(audio_path)
            
            if len(audio_signal.shape) > 1:
                audio_signal = audio_signal[:, 0]
            
            if sample_rate != 16000:
                audio_signal = librosa.resample(
                    y=audio_signal,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
                sample_rate = 16000
            
            if self.use_enhancement:
                import noisereduce as nr
                audio_signal = nr.reduce_noise(y=audio_signal, sr=sample_rate)
            
            processed_path = audio_path.replace('.wav', '_enhanced.wav')
            sf.write(processed_path, audio_signal, sample_rate)
            
            return processed_path
            
        except Exception as e:
            print(f"Error in audio preprocessing: {str(e)}")
            return audio_path

    def _get_speaker_embedding(self, audio_segment: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, float]:
        try:
            if len(audio_segment.shape) > 1:
                audio_segment = audio_segment.mean(axis=1)
            
            min_duration = 0.5
            min_samples = int(min_duration * sample_rate)
            if len(audio_segment) < min_samples:
                audio_segment = np.pad(audio_segment, (0, min_samples - len(audio_segment)))
            
            waveform = torch.from_numpy(audio_segment).float()
            waveform = waveform.unsqueeze(0)
            waveform = waveform.unsqueeze(0)
            
            embedding = self.embedding_model(waveform)
            
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding, 1.0
            
        except Exception as e:
            print(f"Error extracting speaker embedding: {str(e)}")
            return None, 0.0

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        try:
            similarity = 1 - cosine(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0.0

    def _update_clusters(self):
        if len(self.speaker_embeddings_list) < self.n_clusters:
            return
            
        X = np.array(self.speaker_embeddings_list)
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(X)
        
        self.cluster_centers = self.kmeans.cluster_centers_
        
        cluster_speakers = defaultdict(list)
        for embedding, speaker_id, cluster in zip(
            self.speaker_embeddings_list,
            self.speaker_ids_list,
            self.kmeans.labels_
        ):
            cluster_speakers[cluster].append(speaker_id)
        
        for cluster, speakers in cluster_speakers.items():
            speaker_counts = {}
            for speaker in speakers:
                if speaker in speaker_counts:
                    speaker_counts[speaker] += 1
                else:
                    speaker_counts[speaker] = 1
            
            most_common = max(speaker_counts, key=speaker_counts.get)
            self.cluster_to_speaker[cluster] = most_common
            
    def _find_matching_speaker(self, embedding: np.ndarray) -> str:
        if embedding is None:
            return f"unknown_speaker_{self.conversation_index}"
            
        if not self.speaker_embeddings:
            new_id = f"speaker_{len(self.speaker_embeddings) + 1}_{self.conversation_index}"
            self.speaker_embeddings[new_id] = embedding
            return new_id
        
        best_score = -1
        best_speaker = None
        
        for speaker_id, stored_embedding in self.speaker_embeddings.items():
            similarity = self._compute_similarity(embedding, stored_embedding)
            if similarity > best_score:
                best_score = similarity
                best_speaker = speaker_id
        
        if best_score >= self.similarity_threshold:
            return best_speaker
        else:
            new_id = f"speaker_{len(self.speaker_embeddings) + 1}_{self.conversation_index}"
            self.speaker_embeddings[new_id] = embedding
            
            self.speaker_embeddings_list.append(embedding)
            self.speaker_ids_list.append(new_id)
            
            if len(self.speaker_embeddings_list) >= self.n_clusters:
                self._update_clusters()
                
            return new_id
            
    def _process_segment(self, segment: dict) -> Optional[DiarizationResult]:
        try:
            start_time = segment["segment"]["start"]
            end_time = segment["segment"]["end"]
            transcript = segment.get("text", "")
            
            if not "embedding" in segment:
                return DiarizationResult(
                    speaker_id="unknown",
                    start_time=start_time,
                    end_time=end_time,
                    transcript=transcript,
                    confidence=0.5
                )
            
            embedding = segment["embedding"]
            speaker_id = self._find_matching_speaker(embedding)
            
            self.context_manager.add_speech_segment(
                speaker_id=speaker_id,
                transcript=transcript,
                start_time=start_time,
                end_time=end_time,
                embedding=embedding.tolist() if embedding is not None else None
            )
            
            return DiarizationResult(
                speaker_id=speaker_id,
                start_time=start_time,
                end_time=end_time,
                transcript=transcript,
                confidence=segment.get("confidence", 1.0),
                speaker_embedding=embedding,
                speaker_confidence=segment.get("speaker_confidence", 1.0)
            )
            
        except Exception as e:
            print(f"Error processing segment: {str(e)}")
            return None

    def start_processing(self):
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_queue)
            self.processing_thread.start()
        
    def stop_processing(self):
        if self.is_processing:
            self.is_processing = False
            if self.processing_thread:
                self.processing_thread.join()
                
    def _process_queue(self):
        while self.is_processing and self.processing_queue:
            task = self.processing_queue.pop(0)
            audio_path, callback, num_speakers = task
            try:
                results = self.process_audio(audio_path, callback, num_speakers)
            except Exception as e:
                print(f"Error processing audio: {str(e)}")

    def process_audio(self, audio_path: str, callback: Callable[[DiarizationResult], None], num_speakers: Optional[int] = None):
        try:
            # Preprocess audio if necessary
            if self.use_enhancement:
                audio_path = self._preprocess_audio(audio_path)
            
            # Run diarization
            diarization = self.diarization_pipeline(
                audio_path, 
                num_speakers=num_speakers
            )
            
            # Run transcription
            result = self.asr_model.transcribe(audio_path)
            transcript = result["text"]
            segments = result["segments"]
            
            # Extract audio data
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)
            
            processed_segments = []
            
            # Process each speaker segment
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                
                # Extract segment from audio
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                if end_sample > len(audio_data):
                    end_sample = len(audio_data)
                
                audio_segment = audio_data[start_sample:end_sample]
                
                # Get embedding
                embedding, speaker_confidence = self._get_speaker_embedding(audio_segment, sample_rate)
                
                # Find matching transcript segment
                matching_text = ""
                segment_confidence = 1.0
                
                for segment in segments:
                    seg_start = segment["start"]
                    seg_end = segment["end"]
                    
                    # Check for overlap
                    if (seg_start <= end_time and seg_end >= start_time):
                        matching_text = segment["text"]
                        segment_confidence = segment.get("confidence", 1.0)
                        break
                
                processed_segment = {
                    "segment": {"start": start_time, "end": end_time},
                    "speaker": speaker,
                    "text": matching_text,
                    "confidence": segment_confidence,
                    "embedding": embedding,
                    "speaker_confidence": speaker_confidence
                }
                
                processed_segments.append(processed_segment)
                
                # Call callback with result
                result = self._process_segment(processed_segment)
                if result and callback:
                    callback(result)
            
            return processed_segments
                
        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            return []

    def get_speaker_history(self, speaker_id: str) -> List[Dict]:
        return self.context_manager.get_speaker_history(speaker_id)
    
    def get_speaker_stats(self, speaker_id: str) -> Dict:
        return self.context_manager.get_speaker_stats(speaker_id)
    
    def save_contexts(self, filepath: str):
        self.context_manager.save_to_file(filepath)
    
    def load_contexts(self, filepath: str):
        self.context_manager.load_from_file(filepath)
        self._load_speaker_embeddings()

    def result_callback(self, result: DiarizationResult):
        if not result:
            return
        
        print(f"Speaker: {result.speaker_id}, Time: {result.start_time:.2f}-{result.end_time:.2f}")
        if result.transcript:
            print(f"Transcript: {result.transcript}")
        print("-" * 40)

def diarize_speech(audio_file: str) -> List[DiarizationResult]:
    try:
        processor = RealTimeProcessor()
        results = []
        
        def callback(result):
            results.append(result)
        
        processor.process_audio(audio_file, callback)
        
        return results
    except Exception as e:
        print(f"Error in diarization: {str(e)}")
        return []

def combine_diarization_with_transcript(
    diarization_results: List[DiarizationResult],
    transcript: str,
    word_timestamps: List[Tuple[str, float, float]]
) -> List[DiarizationResult]:
    combined_results = []
    
    if not diarization_results or not word_timestamps:
        return combined_results
    
    # Sort diarization results by start time
    diarization_results = sorted(diarization_results, key=lambda x: x.start_time)
    
    # Sort word timestamps
    word_timestamps = sorted(word_timestamps, key=lambda x: x[1])
    
    # Assign words to speakers
    for diar_result in diarization_results:
        segment_words = []
        
        for word, start_time, end_time in word_timestamps:
            # Check if word is within the speaker's time segment
            if start_time >= diar_result.start_time and end_time <= diar_result.end_time:
                segment_words.append(word)
        
        # Create a new result with the transcript
        new_result = DiarizationResult(
            speaker_id=diar_result.speaker_id,
            start_time=diar_result.start_time,
            end_time=diar_result.end_time,
            transcript=" ".join(segment_words),
            confidence=diar_result.confidence,
            speaker_embedding=diar_result.speaker_embedding,
            speaker_confidence=diar_result.speaker_confidence
        )
        
        combined_results.append(new_result)
    
    return combined_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diarization_model.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    print(f"Processing audio file: {audio_file}")
    
    def result_callback(result: DiarizationResult):
        print(f"Speaker: {result.speaker_id}, Time: {result.start_time:.2f}-{result.end_time:.2f}")
        print(f"Transcript: {result.transcript}")
        print("-" * 40)
    
    processor = RealTimeProcessor()
    processor.process_audio(audio_file, result_callback)
    
    print("Processing complete")
