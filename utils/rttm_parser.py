def parse_rttm(rttm_file):
    """
    Parse the RTTM file and extract speaker segments with their start time and end time.

    Args:
    - rttm_file (str): Path to the RTTM file.

    Returns:
    - segments (list): A list of dictionaries containing speaker ID, start time, and end time.
    """
    segments = []

    try:
        with open(rttm_file, 'r') as file:
            for line in file:
                # Split the line into parts
                fields = line.strip().split()

                # Ensure this is a speaker line
                if fields[0] == 'SPEAKER':
                    # Extract the relevant details
                    file_name = fields[1]        # Audio file name
                    start_time = float(fields[3])  # Start time in seconds
                    duration = float(fields[4])   # Duration in seconds
                    speaker_id = fields[7]       # Speaker ID

                    # Calculate the end time
                    end_time = start_time + duration

                    # Store the segment details in a dictionary
                    segment = {
                        'speaker_id': speaker_id,
                        'start_time': start_time,
                        'end_time': end_time
                    }
                    
                    # Append the segment to the segments list
                    segments.append(segment)

    except FileNotFoundError:
        print(f"Error: The file {rttm_file} was not found.")
    except Exception as e:
        print(f"Error while parsing the RTTM file: {e}")

    return segments