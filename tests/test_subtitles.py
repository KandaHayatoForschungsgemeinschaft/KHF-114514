import tempfile, unittest, subprocess, shutil
from pathlib import Path
from archive_subtitles import timestamp, srt_text, extract_audio
class TestSubtitles(unittest.TestCase):
    def test_timestamp(self):
        self.assertEqual(timestamp(3661.234),'01:01:01,234')
        self.assertEqual(timestamp(59.9996),'00:01:00,000')
    def test_srt(self):
        self.assertEqual(srt_text([{'start':1,'end':2,'text':'Test'}]),'1\n00:00:01,000 --> 00:00:02,000\nTest\n')
    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg not available')
    def test_audio(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'tone.wav'; pcm=Path(d)/'audio.pcm'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=1','-y',str(source)],check=True)
            extract_audio(source,pcm)
            self.assertEqual(pcm.stat().st_size,32000)
if __name__=='__main__': unittest.main()
