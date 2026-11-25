"""Local CPU transcription and subtitle rendering for archival media."""
import argparse, json, math, os, subprocess, tempfile
from pathlib import Path

def timestamp(seconds):
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'

def srt_text(rows):
    return '\n\n'.join(f"{i}\n{timestamp(r['start'])} --> {timestamp(r['end'])}\n{r['text'].strip()}" for i,r in enumerate(rows,1)) + '\n'

def atomic_write(path, text):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text, encoding='utf-8'); temp.replace(path)

def extract_audio(source, target):
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-vn','-ac','1','-ar','16000','-f','s16le',str(target)],check=True)

def transcribe(args):
    import numpy as np
    from faster_whisper import WhisperModel
    source = Path(args.source).resolve(); out = Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pcm = Path(tmp)/'audio.pcm'; extract_audio(source,pcm)
        audio = np.fromfile(pcm,dtype='<i2').astype(np.float32)/32768
    sr=16000; duration=len(audio)/sr
    if duration <= 0: raise ValueError('Empty audio')
    signature={'source':str(source),'size':source.stat().st_size,'mtime_ns':source.stat().st_mtime_ns,'model':args.model,'duration':duration,'chunk':45,'task':args.task}
    statefile=out/f'{args.task}.json'; srt=out/f'{args.task}.srt'
    state={'source':signature,'segments':[],'next_chunk':0,'complete':False}
    if statefile.exists():
        state=json.loads(statefile.read_text(encoding='utf-8'))
        if state['source'] != signature: raise ValueError('Source or settings changed; use a new output directory')
    if state.get('complete'):
        atomic_write(srt,srt_text(state['segments'])); print('Already complete'); return
    model=WhisperModel(args.model,device='cpu',compute_type='int8',cpu_threads=args.threads,num_workers=1)
    for idx,start in enumerate(range(0,math.ceil(duration),45)):
        if idx<state['next_chunk']: continue
        stop=min(start+45,duration); left=max(0,start-4); right=min(duration,stop+4)
        segs,info=model.transcribe(audio[round(left*sr):round(right*sr)],task=args.task,beam_size=3,condition_on_previous_text=False,vad_filter=False)
        for seg in segs:
            a=max(0,left+seg.start); b=min(duration,left+seg.end)
            if b<=a or not start<=(a+b)/2<stop or not seg.text.strip(): continue
            state['segments'].append({'start':round(a,3),'end':round(b,3),'text':seg.text.strip(),'chunk_language':info.language,'language_probability':float(info.language_probability),'avg_logprob':float(seg.avg_logprob),'no_speech_prob':float(seg.no_speech_prob)})
        state['segments'].sort(key=lambda x:(x['start'],x['end']))
        state['next_chunk']=idx+1
        atomic_write(statefile,json.dumps(state,ensure_ascii=False,indent=2)); atomic_write(srt,srt_text(state['segments']))
        print(f'{stop:.1f}/{duration:.1f} seconds processed',flush=True)
    if not state['segments']: raise RuntimeError('No speech segments recovered')
    state['complete']=True; atomic_write(statefile,json.dumps(state,ensure_ascii=False,indent=2))

def render(args):
    src=Path(args.source).resolve(); sub=Path(args.subtitles).resolve(); output=Path(args.output).resolve()
    if output==src: raise ValueError('Do not overwrite source')
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy2(sub,Path(tmp)/'english.srt')
        vf="scale=-2:720:flags=lanczos,setsar=1,subtitles=english.srt:force_style='FontName=DejaVu Sans,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=0,Alignment=2,MarginV=15'"
        subprocess.run(['ffmpeg','-n','-i',str(src),'-vf',vf,'-map','0:v:0','-map','0:a?','-c:v','libx264','-preset','medium','-crf','18','-c:a','copy','-movflags','+faststart',str(output)],cwd=tmp,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__); sp=p.add_subparsers(dest='command',required=True)
    t=sp.add_parser('transcribe'); t.add_argument('source'); t.add_argument('output'); t.add_argument('--model',default='medium'); t.add_argument('--threads',type=int,default=min(8,os.cpu_count() or 2)); t.add_argument('--task',choices=['transcribe','translate'],default='transcribe'); t.set_defaults(fn=transcribe)
    r=sp.add_parser('render'); r.add_argument('source'); r.add_argument('subtitles'); r.add_argument('output'); r.set_defaults(fn=render)
    a=p.parse_args(); a.fn(a)
if __name__=='__main__': main()
