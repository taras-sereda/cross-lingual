# one-lang


commands
1. list available subs for youtube video: `yt-dlp CxGs9E-QwEA  --list-subs`
```
[info] Available subtitles for CxGs9E-QwEA:
Language Name                     Formats
be       Belarusian               vtt, ttml, srv3, srv2, srv1, json3
cs       Czech                    vtt, ttml, srv3, srv2, srv1, json3
en-GB    English (United Kingdom) vtt, ttml, srv3, srv2, srv1, json3
en-US    English (United States)  vtt, ttml, srv3, srv2, srv1, json3
fr       French                   vtt, ttml, srv3, srv2, srv1, json3
de       German                   vtt, ttml, srv3, srv2, srv1, json3
no       Norwegian                vtt, ttml, srv3, srv2, srv1, json3
pl       Polish                   vtt, ttml, srv3, srv2, srv1, json3
es-419   Spanish (Latin America)  vtt, ttml, srv3, srv2, srv1, json3
es-ES    Spanish (Spain)          vtt, ttml, srv3, srv2, srv1, json3
tr       Turkish                  vtt, ttml, srv3, srv2, srv1, json3
uk       Ukrainian                vtt, ttml, srv3, srv2, srv1, json3
```   
2. download video with subtitles saved as a separate vtt/srt file: `yt-dlp CxGs9E-QwEA --write-subs --sub-langs en-US -x --audio-format wav -k`
```
-x - extract audio
-k - keep video

CxGs9E-QwEA.webm
CxGs9E-QwEA.en-US.vtt
```

3. run tortoise for synthesis: `python ~/git-repos/tortoise-tts/tortoise/read.py --textfile /home/taras/one-lang/klopotenko/CxGs9E-QwEA/input_text.txt  --voice klopotenko --output_path .`

4. regex to remove duplicate lines: `^(.*)(\r?\n\1)+$`

5. for db graph vizualization: `brew install graphviz`

6. Launching from project root directory:

- fastAPI: `uvicorn editor_app.main:app --reload`
  add --host 0.0.0.0 to make app available to LAN.
- gradio: `gradio editor_app/editor2.py editor`


7. combine audio and video: `ffmpeg -i demo_3PVQ0rN_jp4.webm -i demo_3PVQ0rN_jp4_english_shortened.wav -map 0:v -map 1:a  demo_3PVQ0rN_jp4_english.mp4`

8. calculate durations for all *.mp3 files in the directory:
    ```
    find . -name "*.mp3" -exec ffprobe -hide_banner -v error -select_streams a:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 {} \;  | tee durations.txt
    awk '{ sum += $1 } END { print sum/60/60 }' durations.txt
    
    ```

9. play audio in command line start from sec: `play <input_audio.wav> trim <sec>`