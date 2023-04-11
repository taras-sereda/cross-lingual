import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import ffmpeg
import requests
from pytube import YouTube

from config import cfg
from string_utils import get_random_string

ffmpeg_path = shutil.which("ffmpeg")
demucs_path = shutil.which("demucs")

audio_extensions = [
    '.mp3',
    '.wav',
    '.aac',
    '.flac',
    '.wma',
    '.ogg',
    '.m4a'
]

video_extensions = [
    '.mp4',
    '.avi',
    '.mkv',
    '.mov',
    '.wmv',
    '.flv',
    '.webm',
    '.m4v'
]

media_extensions = audio_extensions + video_extensions


def media_has_video_steam(media_path: Path) -> bool:
    probe = ffmpeg.probe(media_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    return video_stream is not None


def get_stream_duration(media_path, stream_type='audio'):
    probe = ffmpeg.probe(media_path)
    stream = next((stream for stream in probe['streams'] if stream['codec_type'] == stream_type), None)
    return round(float(stream['duration'])) if stream is not None else None


def mux_video_audio(video_path: Path, audio_path: Path, output_path: str, video_offset_sec: int = 0):
    """Maps the video stream from one file and the audio stream from another file
       and saves the output to a new file using ffmpeg.
    Raises:
        ValueError: If the video and audio files have incompatible codecs.

    """
    # Use ffmpeg to get the streams from the video and audio files
    audio_duration = get_stream_duration(audio_path)
    video = ffmpeg.input(video_path, ss=video_offset_sec, t=audio_duration+cfg.demo.extra_dur_sec)
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


def extract_audio(media_path: Path, raw_audio_path: Path):
    in_media = ffmpeg.input(media_path)
    in_media.audio.output(str(raw_audio_path), **{'ac': 1}).run()


def resample_audio(audio_path: Path, resample_audio_path: Path, sample_rate: int):
    audio = ffmpeg.input(audio_path).audio
    ffmpeg.output(audio, str(resample_audio_path), **{'ar': sample_rate}).run()


def get_youtube_embed_code(youtube_link) -> str:

    yt = YouTube(youtube_link)
    iframe_code = f'<iframe width="560" height="315" src="{yt.embed_url}" frameborder="0" allowfullscreen></iframe>'
    return iframe_code


def demucs_audio(audio_path: Path):
    command = [
        f"{demucs_path}",
        "--two-stems", "vocals",
        "--out", f"{audio_path.parent}",
        "--filename", "{track}.{stem}.{ext}",
        f"{audio_path}"]
    res = subprocess.run(command, check=True)
    return res


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
    filename = get_random_string()
    video_file = video.download(output_dir, filename)

    return Path(video_file)


def download_media(url: str, save_path: str) -> str | None:
    """
    Downloads media file from the specified URL and saves it to the specified file path.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Extract the file name from the URL
        file_name = os.path.basename(urlparse(url).path)
        if not file_name:
            file_name = get_random_string()
        full_file_path = os.path.join(save_path, file_name)

        with open(full_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return full_file_path

    except requests.exceptions.RequestException as e:
        print(f"Failed to download media file from {url}: {str(e)}")


def download_rss(url: str, save_path: str):
    response = requests.get(url)
    filename = os.path.join(save_path, get_random_string())
    with open(filename, 'wb') as f:
        f.write(response.content)
    return filename
