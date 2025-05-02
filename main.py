from models.nemo_diarization_model import diarize_speech
from utils.rttm_parser import parse_rttm
from robot.movement import move_robot_toward_speaker
from utils.memory import SpeakerMemory
import nemo.collections.asr as nemo_asr

def transcribe_audio(audio_file):
    """
    Transcribes the audio file using a pre-trained ASR model.

    Args:
    - audio_file (str): Path to the audio file.

    Returns:
    - transcription (str): The transcribed text from the audio.
    """
    # Load a pre-trained ASR model (QuartzNet or Jasper)
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained("nvidia/quartznet15x5base-en")

    # Perform transcription
    transcription = asr_model.transcribe([audio_file])
    
    return transcription[0]

def main():
    # Load and process audio
    audio_file = "data/audio/sample_audio.wav"
    rttm_file = "data/rttm/sample_rttm.rttm"
    
    # Step 1: Process RTTM file
    segments = parse_rttm(rttm_file)
    
    # Step 2: Speaker diarization with NeMo EEND model
    diarization_result = diarize_speech(audio_file)
    
    # Step 3: Transcribe the audio to get text
    transcription = transcribe_audio(audio_file)
    
    # Step 4: Update memory with conversations and speaker IDs
    memory = SpeakerMemory()
    text_segments = transcription.split(".")  # Assuming transcription is split into sentences or utterances

    # Assuming that the segments from the RTTM file align with sentences in the transcription:
    for i, segment in enumerate(segments):
        speaker_id = segment['speaker_id']
        # Use the corresponding segment of the transcription
        conversation_text = text_segments[i] if i < len(text_segments) else "No text available"
        memory.add_conversation(speaker_id, conversation_text)
    

if __name__ == "__main__":
    main()
