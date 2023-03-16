import os
import shutil
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

import ffmpeg
import requests
from pytube import YouTube

ffmpeg_path = shutil.which('ffmpeg')

extra_length_sec = 2


def media_has_video_steam(media_path: Path) -> bool:
    probe = ffmpeg.probe(media_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    return video_stream is not None


def get_stream_duration(media_path, stream_type='audio'):
    probe = ffmpeg.probe(media_path)
    stream = next((stream for stream in probe['streams'] if stream['codec_type'] == stream_type), None)
    return round(float(stream['duration'])) if stream is not None else None


def mux_video_audio(video_path: Path, audio_path: Path, output_path: str, video_offset_sec: int):
    """Maps the video stream from one file and the audio stream from another file
       and saves the output to a new file using ffmpeg.
    Raises:
        ValueError: If the video and audio files have incompatible codecs.

    """
    # Use ffmpeg to get the streams from the video and audio files
    audio_duration = get_stream_duration(audio_path)
    video = ffmpeg.input(video_path, ss=video_offset_sec, t=audio_duration+extra_length_sec)
    audio = ffmpeg.input(audio_path)

    # Map the video and audio streams to the output file
    output = ffmpeg.output(video.video, audio.audio, output_path).overwrite_output()

    try:
        # Run the ffmpeg command to create the output file
        ffmpeg.run(output)
    except ffmpeg.Error as e:
        # Raise a ValueError if the video and audio files have incompatible codecs
        if 'Invalid data found when processing input' in str(e):
            raise ValueError('The video and audio files have incompatible codecs')
        else:
            raise e


def get_youtube_embed_code(youtube_link) -> str:
    """
    Constructs an embed URL for a YouTube video and returns the corresponding iframe code.

    Returns:
        str: The iframe code for the YouTube video.
    """
    yt = YouTube(youtube_link)
    iframe_code = f'<iframe width="560" height="315" src="{yt.embed_url}" frameborder="0" allowfullscreen></iframe>'
    return iframe_code


def convert_wav_to_mp3_ffmpeg(in_path: Path, out_path: Path):
    res = subprocess.run([
        f"{ffmpeg_path}",
        "-hide_banner",
        "-loglevel", "error",
        "-i", f"{in_path}",
        "-ab", "320k",
        f"{out_path}", "-y"],
        check=True,
        # stdout=subprocess.DEVNULL,
    )
    return res


def extract_and_resample_audio_ffmpeg(in_path: Path, out_path: Path, out_sample_rate: int):
    # command = f"{ffmpeg_path} -i {media_path} -ac 1 -ar 16000 {raw_media_path}"
    # print(os.environ.get('PATH'))
    res = subprocess.run([
        f"{ffmpeg_path}",
        "-hide_banner",
        "-loglevel", "error",
        "-i", f"{in_path}",
        "-ac", "1",
        "-ar", f"{out_sample_rate}",
        # "-acodec", f"pcm_s16le",
        f"{out_path}"],
        check=True,
        # stdout=subprocess.DEVNULL,
    )
    return res


def download_youtube_media(url, output_dir) -> Path:
    """
    Downloads a YouTube video.

    Args:
        url (str): The URL of the YouTube video to download.
        output_dir (str): The directory to save the downloaded video.

    Returns:
        The file path of the downloaded video.
    """
    yt = YouTube(url)

    # Get the first available video stream
    video = yt.streams.filter(progressive=True, file_extension='mp4').first()

    # Download the video
    filename = str(uuid.uuid4())
    video_file = video.download(output_dir, filename)

    return Path(video_file)


def download_media(url: str, save_path: Path) -> bool:
    """
    Downloads media file from the specified URL and saves it to the specified file path.

    Args:
        url: The URL of the media file to download.
        save_path: The file path where the media file will be saved.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        # Make a GET request to the URL to download the media file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Extract the file name from the URL
        file_name = os.path.basename(urlparse(url).path)

        # If a file name was not extracted, generate a random one
        if not file_name:
            file_name = str(uuid.uuid4())

        # Construct the full file path by appending the file name to the save path
        full_file_path = save_path.joinpath(file_name)

        # Save the media file to disk
        with open(full_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(full_file_path)
        return True

    except requests.exceptions.RequestException as e:
        print(f"Failed to download media file from {url}: {str(e)}")
        return False
