# KHF Archive Tools

Generic archival subtitle utility. CI tests use synthetic media only. Archival footage, transcripts and credentials are not committed. Actual transcription must run on an authorised local machine or an appropriate compute service. Machine transcription and translation are drafts requiring review.

Python 3.11 and FFmpeg with libass are required.

```sh
python -m pip install -r requirements.txt
python archive_subtitles.py transcribe input.mp4 drafts
python archive_subtitles.py transcribe input.mp4 drafts --task translate
python archive_subtitles.py render input.mp4 reviewed_english.srt output.mp4
python -m unittest discover -s tests -v
```

CPU INT8 transcription, multilingual detection, checkpointed output and aspect-ratio-preserving 720p rendering. Review all timings and text before publication. Use the same cut for transcription and rendering.
