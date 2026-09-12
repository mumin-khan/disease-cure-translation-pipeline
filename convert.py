import os
import subprocess

def convert_to_wav():
    # Ask the user for the file path
    file_path = input("Enter the path to the .mp3 or .m4a file: ").strip()
    
    # Clean up the path (remove quotes if user copied the path)
    file_path = file_path.replace('"', '').replace("'", "")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Define the output filename
    base_name = os.path.splitext(file_path)[0]
    output_file = base_name + ".wav"

    # FFmpeg command for high-quality WAV
    command = [
        'ffmpeg',
        '-i', file_path,
        '-acodec', 'pcm_s16le',  # Standard 16-bit PCM
        '-ar', '44100',           # Standard sample rate
        '-ac', '2',               # Stereo
        output_file,
        '-y'                      # Overwrite if file already exists
    ]

    try:
        print(f"Converting to {output_file}...")
        subprocess.run(command, check=True)
        print("Conversion Complete!")
    except FileNotFoundError:
        print("Error: 'ffmpeg' was not found. Please install it and add it to your system PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed with error: {e}")

if __name__ == "__main__":
    convert_to_wav()
