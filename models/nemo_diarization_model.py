import nemo.collections.asr as nemo_asr

def diarize_speech(audio_file):
    """
    Function to perform speaker diarization on an audio file using NeMo's pre-trained EEND model.
    
    Args:
    - audio_file (str): Path to the audio file to process.

    Returns:
    - diarization_result (list): A list of dictionaries with speaker segments and speaker IDs.
    """
    # Load a pre-trained EEND model
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/eend_asr")
    
    # Process the audio file for speaker segmentation
    diarization_result = model.transcribe([audio_file])
    
    # Output result
    # diarization_result contains a list of speakers with their corresponding time intervals
    return diarization_result
