import numpy as np
import librosa

def get_speaker_direction(audio_file):
    """
    Estimate the direction of the speaker based on stereo audio input.

    Args:
    - audio_file (str): Path to the stereo audio file.

    Returns:
    - direction (str): Estimated direction of the speaker ("left", "right", "center").
    """
    # Load the stereo audio file (assumes two channels)
    audio_data, sr = librosa.load(audio_file, sr=None, mono=False)

    # Split the stereo audio into two channels (left and right)
    left_channel = audio_data[0, :]
    right_channel = audio_data[1, :]

    # Perform simple cross-correlation to find the delay between channels
    delay = np.correlate(left_channel, right_channel, mode='full')
    lag = np.argmax(np.abs(delay)) - len(left_channel) + 1

    # Use the delay to estimate direction (basic logic)
    if lag > 0:
        direction = "right"
    elif lag < 0:
        direction = "left"
    else:
        direction = "center"

    print(f"Speaker is coming from the {direction}.")
    return direction

# Example usage:
if __name__ == "__main__":
    audio_file = "data/audio/sample_stereo_audio.wav"
    direction = get_speaker_direction(audio_file)
