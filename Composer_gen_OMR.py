#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
composer_gen.py — PDF OMR + symbolic training + UI text outputs

What’s new:
- OMR path: Convert PDF scores → MusicXML via your OMR CLI (e.g., Audiveris), then train.
- Direct training from MusicXML / MIDI / LilyPond.
- UI controls: Enable OMR, set OMR tool path, and open output folder.
- Text viewers for LilyPond and MusicXML in the UI.

Install:
  pip install music21 PyPDF2

Launch UI:
  python composer_gen.py --ui

Notes:
- For Audiveris: OMR Tool Path can be either
    "C:\\Program Files\\Audiveris\\bin\\audiveris.bat"
  or
    "C:\\Program Files\\Java\\bin\\java.exe -jar C:\\tools\\audiveris\\audiveris.jar"
- The app will create a temp OMR output folder inside your output dir unless you pass --omr-out (CLI).
"""

from __future__ import annotations
import argparse, os, sys, math, random, re, collections, warnings, tempfile, shutil, subprocess, webbrowser, json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable

# ---- Quiet PDF warnings (when reading text-only PDFs)
try:
    import PyPDF2  # type: ignore
    try:
        from PyPDF2.errors import PdfReadWarning
        warnings.filterwarnings("ignore", category=PdfReadWarning)
    except Exception:
        pass
except Exception:
    PyPDF2 = None

# ---- music21 (required)
try:
    from music21 import stream, note, chord, meter, tempo, clef, instrument, duration, pitch, interval, harmony, converter
except Exception:
    print("ERROR: music21 is required. Install:  pip install music21", file=sys.stderr)
    raise

# --------------------- Globals & Helpers ---------------------
MODES = {
    "ionian":[0,2,4,5,7,9,11], "dorian":[0,2,3,5,7,9,10], "phrygian":[0,1,3,5,7,8,10],
    "lydian":[0,2,4,6,7,9,11], "mixolydian":[0,2,4,5,7,9,10],
    "aeolian":[0,2,3,5,7,8,10], "locrian":[0,1,3,5,6,8,10],
}
PREFERRED_RANGE = {
    "RH": (pitch.Pitch("C4"), pitch.Pitch("A6")),
    "LH": (pitch.Pitch("C2"), pitch.Pitch("E4")),
}
SUPPORTED_TEXT = (".txt",".md")
SUPPORTED_PDF  = (".pdf",)
SUPPORTED_SYMBOLIC = (".musicxml",".xml",".mxl",".mid",".midi",".ly")

def ensure_dir(p:str): os.makedirs(p, exist_ok=True)
def sanitize_mode(name:str)->str: return name.strip().lower()
def sanitize_filename(s:str)->str: return re.sub(r"[^A-Za-z0-9._-]+","_",s)

# --------------------- Data Structures ---------------------
@dataclass
class ScoreRequest:
    chord_symbol: str = "C"
    mode: str = "Dorian"
    staves: int = 64
    meter: str = "4/4"
    tempo_bpm: int = 120
    style_weights: Dict[str,float] = field(default_factory=lambda: {"jazz":0.4,"latin":0.3,"classical":0.2,"blues":0.1})
    books_path: Optional[str] = None
    grand_staff: bool = True
    seed: Optional[int] = None
    # OMR options
    use_omr: bool = False
    omr_tool: Optional[str] = None   # full command line string or executable path
    omr_out: Optional[str] = None    # optional output folder for OMR results
    omr_workers: int = 1

@dataclass
class Corpus:
    # Free-form text (used for tiny style nudges)
    text_blobs: List[str] = field(default_factory=list)
    # Symbolic files discovered or produced by OMR
    symbolic_files: List[str] = field(default_factory=list)
    # Influence nudges (very small)
    influence: Dict[str,float] = field(default_factory=dict)

@dataclass
class MarkovModel:
    order: int
    counts: Dict[Tuple, collections.Counter] = field(default_factory=lambda: collections.defaultdict(collections.Counter))
    def observe(self, ctx:Tuple, sym):
        self.counts[ctx][sym]+=1
    def sample(self, ctx:Tuple, temperature:float=1.0):
        for k in range(len(ctx), -1, -1):
            c = ctx[-k:] if k>0 else tuple()
            if c in self.counts and self.counts[c]:
                return weighted_sample(self.counts[c], temperature)
        all_syms = list({s for ctr in self.counts.values() for s in ctr.keys()})
        return random.choice(all_syms) if all_syms else None

@dataclass
class StyleModels:
    pitch_deg_model: MarkovModel    # diatonic degrees 0..6 (+7 for chromatic/blue)
    rhythm_model: MarkovModel       # quarter-length transitions
    interval_model: MarkovModel     # melodic semitone intervals (-24..+24) transitions
    texture_model: MarkovModel      # accompaniment pattern transitions

# --------------------- Base Sampling Helpers ---------------------
def weighted_sample(counter: collections.Counter, temperature: float = 1.0):
    items = list(counter.items())
    if not items: return None
    syms, counts = zip(*items)
    # temperature scaling
    weights = [c ** (1.0/max(1e-6, temperature)) for c in counts]
    s = sum(weights) or 1.0
    r, cum = random.random()*s, 0.0
    for sym, w in zip(syms, weights):
        cum += w
        if r <= cum: return sym
    return syms[-1]

def parse_styles(s: str) -> Dict[str,float]:
    out={}
    if s:
        for chunk in s.split(","):
            if "=" in chunk:
                k,v = chunk.split("=",1)
                try: out[k.strip().lower()] = float(v)
                except: pass
    if not out: out = {"jazz":0.4,"latin":0.3,"classical":0.2,"blues":0.1}
    tot = sum(out.values()) or 1.0
    return {k:v/tot for k,v in out.items()}

# --------------------- Ingestion ---------------------
def collect_files(folder:str) -> Tuple[List[str], List[str], List[str]]:
    pdfs, symbolic, texts = [], [], []
    for root,_,files in os.walk(folder):
        for fn in files:
            fp = os.path.join(root,fn)
            ext = os.path.splitext(fn.lower())[1]
            if ext in SUPPORTED_PDF: pdfs.append(fp)
            elif ext in SUPPORTED_SYMBOLIC: symbolic.append(fp)
            elif ext in SUPPORTED_TEXT: texts.append(fp)
    return pdfs, symbolic, texts

def read_texts(text_paths:List[str]) -> List[str]:
    blobs=[]
    for fp in text_paths:
        try:
            with open(fp,"r",encoding="utf-8",errors="ignore") as f:
                blobs.append(f.read())
        except Exception as e:
            print(f"[WARN] text '{fp}' skipped: {e}", file=sys.stderr)
    return blobs

def run_omr_on_pdfs(pdfs:List[str], omr_tool:str, out_dir:str, workers:int=1) -> List[str]:
    """
    Run OMR tool on a list of PDFs, produce MusicXML files in out_dir, return list of created files.
    omr_tool can be either:
      - path to audiveris.bat/exe, or
      - full 'java -jar path/to/audiveris.jar' command string.
    We'll detect if it looks like a Java -jar command and build subprocess accordingly.
    """
    ensure_dir(out_dir)
    produced=[]
    if not pdfs:
        return produced

    def build_cmd(pdf_path:str)->List[str]:
        # Audiveris common invocations:
        #  audiveris -batch -export -output <out_dir> <pdf>
        #  java -jar audiveris.jar -batch -export -output <out_dir> <pdf>
        cmd_str = omr_tool.strip()
        if ".jar" in cmd_str.lower() and "java" in cmd_str.lower():
            # Already a full 'java -jar ...'
            parts = cmd_str.split()
            return parts + ["-batch","-export","-output", out_dir, pdf_path]
        elif cmd_str.lower().endswith(".jar"):
            return ["java","-jar",cmd_str,"-batch","-export","-output",out_dir,pdf_path]
        else:
            # assume native executable / .bat
            return [cmd_str, "-batch","-export","-output", out_dir, pdf_path]

    # Simple sequential or minimal parallel (avoid complex pools to keep single-file)
    for i, pdf in enumerate(pdfs, 1):
        print(f"[OMR] ({i}/{len(pdfs)}) {os.path.basename(pdf)}")
        try:
            cmd = build_cmd(pdf)
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] OMR failed on {pdf}: {e.stdout.decode('utf-8', 'ignore')[:500]}", file=sys.stderr)
            continue
        # Audiveris typically writes *.mxl or *.musicxml in out_dir (maybe subfolders)
        for root,_,files in os.walk(out_dir):
            for fn in files:
                if fn.lower().endswith((".musicxml",".mxl",".xml")):
                    produced.append(os.path.join(root,fn))
    # Deduplicate
    produced = sorted(list({os.path.abspath(p) for p in produced}))
    return produced

def ingest_books(req: ScoreRequest) -> Corpus:
    corpus = Corpus()
    if not req.books_path or not os.path.isdir(req.books_path):
        if req.books_path:
            print(f"[WARN] books_path '{req.books_path}' is not a directory. Skipping ingestion.", file=sys.stderr)
        return corpus

    pdfs, symbolic, texts = collect_files(req.books_path)
    corpus.text_blobs = read_texts(texts)

    if req.use_omr and pdfs:
        omr_out = req.omr_out or os.path.join(tempfile.gettempdir(), "omr_out_markov")
        print(f"[OMR] Converting {len(pdfs)} PDF(s) to MusicXML into: {omr_out}")
        if not req.omr_tool:
            print("[ERROR] --omr specified but no OMR tool path provided.", file=sys.stderr)
        else:
            produced = run_omr_on_pdfs(pdfs, req.omr_tool, omr_out, workers=max(1, req.omr_workers))
            print(f"[OMR] Produced {len(produced)} MusicXML file(s).")
            symbolic += produced

    # Very light style bias from texts (prose)
    full = " ".join((tb.lower() for tb in corpus.text_blobs))
    def bump(k,a): corpus.influence[k] = corpus.influence.get(k,0.0)+a
    for name in MODES:
        if name in full: bump(name,0.05)
    for t in ["swing","bebop","guide tone","upper structure"]:
        if t in full: bump("jazz",0.05)
    for t in ["clave","montuno","tumbao"]:
        if t in full: bump("latin",0.05)
    for t in ["counterpoint","species","voice-leading","part writing","fugue"]:
        if t in full: bump("classical",0.05)
    for t in ["blues","blue note","flattened third","flattened fifth","flattened seventh"]:
        if t in full: bump("blues",0.05)

    corpus.symbolic_files = sorted(list({os.path.abspath(p) for p in symbolic}))
    return corpus

# --------------------- Symbolic Training ---------------------
def chord_from_symbol(symbol:str) -> chord.Chord:
    try:
        h = harmony.ChordSymbol(symbol); c = h.figureToChord()
        if len(c.pitches)>0: return c
    except Exception: pass
    # naive major triad on detected root
    m = re.search(r'([A-Ga-g][#b]?)', symbol)
    root_name = m.group(1).upper() if m else "C"
    root_pitch = pitch.Pitch(root_name+"4")
    third = interval.Interval(4).transposePitch(root_pitch)
    fifth = interval.Interval(7).transposePitch(root_pitch)
    return chord.Chord([root_pitch,third,fifth])

def lily_from_pitch(midi_num:int)->str:
    names=["c","cis","d","dis","e","f","fis","g","gis","a","ais","b"]
    octave=(midi_num//12)-1; name=names[midi_num%12]; delta=octave-4
    marks="'"*delta if delta>0 else ","*(-delta) if delta<0 else ""
    return f"{name}{marks}"

def degrees_for_mode(mode_name:str)->List[int]:
    n=sanitize_mode(mode_name)
    if n not in MODES: raise ValueError(f"Unsupported mode '{mode_name}'.")
    return MODES[n]

def analyze_key_for_part(s:stream.Score)->Optional[pitch.Pitch]:
    try:
        k = s.analyze('key')
        return k.tonic
    except Exception:
        return None

def iter_note_sequences(sc: stream.Score) -> Iterable[List[Tuple[int,float]]]:
    """Yield monophonic note sequences from parts: list of (midi, qLength)."""
    try:
        for part in sc.parts:
            seq=[]
            for n in part.flat.notesAndRests:
                if isinstance(n, note.Note):
                    seq.append((n.pitch.midi, float(n.quarterLength)))
                elif isinstance(n, chord.Chord):
                    # represent as average pitch + duration (simple proxy)
                    mp=int(round(sum(p.midi for p in n.pitches)/len(n.pitches)))
                    seq.append((mp, float(n.quarterLength)))
            if len(seq)>=4:
                yield seq
    except Exception:
        # fallback flatten
        seq=[]
        for n in sc.flat.notesAndRests:
            if isinstance(n, note.Note):
                seq.append((n.pitch.midi, float(n.quarterLength)))
            elif isinstance(n, chord.Chord):
                mp=int(round(sum(p.midi for p in n.pitches)/len(n.pitches)))
                seq.append((mp, float(n.quarterLength)))
        if len(seq)>=4:
            yield seq

def degree_of_midi(midi:int, tonic_pc:int)->int:
    """Map MIDI to diatonic degree 0..6 of Ionian/Aeolian proxy; chromatic -> 7."""
    # Ionian steps from tonic
    major_steps=[0,2,4,5,7,9,11]
    pc=midi%12
    if pc in [(tonic_pc + x)%12 for x in major_steps]:
        idx=[(tonic_pc + x)%12 for x in major_steps].index(pc)
        return idx
    else:
        return 7  # chromatic/blue

def train_from_symbolic(models: StyleModels, sym_paths: List[str]):
    for i, fp in enumerate(sym_paths, 1):
        ext = os.path.splitext(fp.lower())[1]
        try:
            sc = converter.parse(fp)
        except Exception as e:
            print(f"[WARN] Could not parse {fp}: {str(e)[:200]}", file=sys.stderr)
            continue

        tonic = analyze_key_for_part(sc)
        tonic_pc = tonic.midi%12 if tonic else 0

        for seq in iter_note_sequences(sc):
            # Observe rhythms and intervals
            mids=[m for m,_ in seq]; durs=[d for _,d in seq]
            for a,b in zip(durs, durs[1:]): models.rhythm_model.observe((a,), b)
            for a,b,c in zip(durs, durs[1:], durs[2:]): models.rhythm_model.observe((a,b), c)

            ints=[m2-m1 for m1,m2 in zip(mids, mids[1:])]
            for a,b in zip(ints, ints[1:]): models.interval_model.observe((a,), b)
            for a,b,c in zip(ints, ints[1:], ints[2:]): models.interval_model.observe((a,b), c)

            # Observe diatonic degree transitions (rough, major proxy)
            degs=[degree_of_midi(m, tonic_pc) for m in mids]
            for a,b in zip(degs, degs[1:]): models.pitch_deg_model.observe((a,), b)
            for a,b,c in zip(degs, degs[1:], degs[2:]): models.pitch_deg_model.observe((a,b), c)

# --------------------- Model Training ---------------------
def train_models(corpus: Corpus, style_weights: Dict[str,float], seed: Optional[int]=None) -> StyleModels:
    if seed is not None: random.seed(seed)

    pitch_deg_model = MarkovModel(order=2)
    rhythm_model    = MarkovModel(order=2)
    interval_model  = MarkovModel(order=2)
    texture_model   = MarkovModel(order=1)

    # Seed with priors (so it works even with no corpus)
    def seed_pitch_priors():
        step=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,5),(5,4),(4,3),(3,2),(2,1),(1,0)]
        leaps=[(0,2),(2,4),(4,6),(6,4),(4,2),(2,0)]
        blues=[(2,7),(7,3),(7,4)]
        for pairs in (step,leaps,blues):
            for a,b in pairs:
                pitch_deg_model.observe((a,), b); pitch_deg_model.observe((a,b), b)
                for nxt in [b, max(0,min(6,b+1)), max(0,min(6,b-1))]:
                    pitch_deg_model.observe((a,b), nxt)
    seed_pitch_priors()

    def seed_rhythm_priors():
        cells=[[1.0,1.0],[0.5]*4,[1.5,0.5],[0.5,1.0,0.5],[0.75,0.25,0.5,0.5],[1.0]*4,[2.0,2.0]]
        for cell in cells:
            for a,b in zip(cell, cell[1:]): rhythm_model.observe((a,), b)
            for a,b,c in zip(cell, cell[1:], cell[2:]): rhythm_model.observe((a,b), c)
    seed_rhythm_priors()

    def seed_interval_priors():
        common=[0,2,-2,1,-1,3,-3,4,-4,5,-5,7,-7,12,-12]
        for a,b in zip(common, common[1:]):
            interval_model.observe((a,), b)
        for i in range(len(common)-2):
            interval_model.observe((common[i], common[i+1]), common[i+2])
    seed_interval_priors()

    # Simple texture cycle
    for i,j in [(0,1),(1,2),(2,0),(0,3),(3,0),(0,4),(4,0)]:
        texture_model.observe((i,), j)

    # Incorporate tiny stylistic nudges from prose (very small)
    for k,v in corpus.influence.items():
        if k in ["jazz","latin","classical","blues"]:
            # nudge via extra rhythm observations
            if k=="jazz":
                for _ in range(int(10*v)): rhythm_model.observe((1.5,), 0.5)
            if k=="latin":
                for _ in range(int(10*v)): rhythm_model.observe((0.5,), 0.5)
            if k=="classical":
                for _ in range(int(10*v)): rhythm_model.observe((1.0,), 1.0)
            if k=="blues":
                for _ in range(int(10*v)): pitch_deg_model.observe((2,), 7)

    # Train from symbolic files (this is where quality jumps)
    if corpus.symbolic_files:
        print(f"[TRAIN] Learning from {len(corpus.symbolic_files)} symbolic file(s)…")
        train_from_symbolic(StyleModels(pitch_deg_model, rhythm_model, interval_model, texture_model),
                            corpus.symbolic_files)

    return StyleModels(pitch_deg_model, rhythm_model, interval_model, texture_model)

# --------------------- Generation (constraint-aware) ---------------------
def build_scale_pitches(root: pitch.Pitch, mode_name:str)->List[int]:
    base=root.midi%12; return [(base+d)%12 for d in degrees_for_mode(mode_name)]
def blue_note_options(scale_pcset:List[int])->List[int]:
    return [pc for pc in [3,6,10] if pc not in scale_pcset]

def degree_to_pitch_midi(root:pitch.Pitch, mode_name:str, degree:int, octave_center:int)->int:
    degs=degrees_for_mode(mode_name); base_pc=root.midi%12
    if degree==7:
        pcs=blue_note_options([(base_pc+d)%12 for d in degs]) or [(base_pc+degs[2])%12]
        pc=random.choice(pcs)
    else:
        pc=(base_pc + degs[max(0,min(6,degree))])%12
    guess=octave_center*12 + pc
    candidates=[guess+k*12 for k in (-2,-1,0,1,2)]
    return min(candidates, key=lambda m: abs(m-(octave_center*12+base_pc)))

def interval_cost(prev_midi:int, new_midi:int, models:StyleModels)->float:
    """Penalty if interval unlikely under learned interval model."""
    if prev_midi is None or not models.interval_model.counts: return 0.0
    iv=new_midi - prev_midi
    # Score by inverse frequency from model root (0 context)
    ctr=models.interval_model.counts
    # build pseudo-prob
    total=sum(ctr.get(tuple(), collections.Counter()).values()) or 1
    freq=ctr.get(tuple(), collections.Counter()).get(iv, 0) + 1
    return 0.3*(1 - (freq/total))

def voice_leading_cost(prev_midi:int, new_midi:int)->float:
    if prev_midi is None: return 0.0
    dist=abs(new_midi-prev_midi); c=0.05*max(0, dist-2)
    if dist>=12: c+=0.5
    return c

def range_cost(midi_num:int, hand:str)->float:
    lo,hi=PREFERRED_RANGE["RH" if hand=="RH" else "LH"]
    if midi_num<lo.midi: return 0.02*(lo.midi-midi_num)
    if midi_num>hi.midi: return 0.02*(midi_num-hi.midi)
    return 0.0

def avoid_parallels(prev_intervals:List[int], new_interval:int)->float:
    pen=0.0
    if new_interval%12 in (0,7):
        pen+=1.0
        if prev_intervals and (prev_intervals[-1]%12) in (0,7): pen+=1.0
    return pen

def pick_degree_next(model:MarkovModel, ctx:List[int])->int:
    out=model.sample(tuple(ctx[-model.order:]))
    return out if out is not None else random.choice(list(range(8)))

def decode_measure(models:StyleModels, req:ScoreRequest, chord_sym:str, root_pitch:pitch.Pitch,
                   ctx_deg:List[int], ctx_rhy:List[float], hand:str, texture_id:int):
    octave_center=6 if hand=="RH" else 3
    meter_num, meter_den = map(int, req.meter.split("/"))
    beats_left = meter_num * (4.0/ meter_den)
    out_notes=[]
    prev_midi=None; prev_intervals=[]

    while beats_left>1e-6:
        dur=models.rhythm_model.sample(tuple(ctx_rhy[-models.rhythm_model.order:])) or 0.5
        if dur>beats_left: dur=max(0.25, beats_left)
        ctx_rhy.append(dur)

        candidates=[pick_degree_next(models.pitch_deg_model, ctx_deg) for _ in range(4)]
        best=(None, float("inf"), None)
        for deg in candidates:
            midi_guess=degree_to_pitch_midi(root_pitch, req.mode, deg, octave_center)
            # Simple texture nudges
            if texture_id==1 and hand=="RH":
                try:
                    cch=chord_from_symbol(chord_sym)
                    cpcs=[p.midi%12 for p in cch.pitches]
                    midi_guess=min([midi_guess+k for k in (-2,-1,0,1,2)],
                                   key=lambda m: min(abs((m%12)-pc) for pc in cpcs))
                except: pass
            elif texture_id==2 and hand=="RH":
                midi_guess+=random.choice([-4,-3,3,4])
            elif texture_id==3 and hand=="LH":
                try:
                    cch=chord_from_symbol(chord_sym)
                    cps=sorted([p.midi%12 for p in cch.pitches])
                    guide=[cps[1], cps[-1]] if len(cps)>=2 else cps
                    if guide: midi_guess=(midi_guess//12)*12 + random.choice(guide)
                except: pass

            sc=0.0
            if prev_midi is not None:
                sc+=voice_leading_cost(prev_midi, midi_guess)
                sc+=avoid_parallels(prev_intervals, midi_guess-prev_midi)
            sc+=range_cost(midi_guess, hand)
            sc+=interval_cost(prev_midi, midi_guess, models)

            if sc<best[1]: best=(deg, sc, midi_guess)

        degree=best[0] if best[0] is not None else candidates[0]
        midi_num=best[2] if best[2] is not None else degree_to_pitch_midi(root_pitch, req.mode, degree, octave_center)
        n=note.Note(); n.pitch=pitch.Pitch(midi=midi_num); n.duration=duration.Duration(dur)
        out_notes.append(n)

        if prev_midi is not None: prev_intervals.append(midi_num-prev_midi)
        prev_midi=midi_num
        ctx_deg.append(degree)
        beats_left-=dur

    return out_notes, ctx_deg, ctx_rhy

def generate_score(models:StyleModels, req:ScoreRequest)->stream.Score:
    if req.seed is not None: random.seed(req.seed)
    s=stream.Score(); partRH, partLH = stream.Part(), stream.Part()
    partRH.insert(0, instrument.Piano()); partLH.insert(0, instrument.Piano())
    partRH.insert(0, clef.TrebleClef());  partLH.insert(0, clef.BassClef())
    m=meter.TimeSignature(req.meter); mm=tempo.MetronomeMark(number=req.tempo_bpm)
    partRH.insert(0,m); partRH.insert(0,mm)
    partLH.insert(0,meter.TimeSignature(req.meter)); partLH.insert(0,tempo.MetronomeMark(number=req.tempo_bpm))

    try:
        ch=chord_from_symbol(req.chord_symbol); root_p=ch.root() or pitch.Pitch("C")
    except Exception: root_p=pitch.Pitch("C")

    ctx_deg_RH, ctx_rhy_RH = [0,1], [1.0,0.5]
    ctx_deg_LH, ctx_rhy_LH = [5,3], [1.0,1.0]
    tex_id_RH = models.texture_model.sample((0,)) or 0
    tex_id_LH = models.texture_model.sample((3,)) or 3

    for _ in range(max(32, min(108, req.staves))):
        measRH, measLH = stream.Measure(), stream.Measure()
        tex_id_RH=models.texture_model.sample((tex_id_RH,)) or tex_id_RH
        tex_id_LH=models.texture_model.sample((tex_id_LH,)) or tex_id_LH
        notesRH, ctx_deg_RH, ctx_rhy_RH = decode_measure(models, req, req.chord_symbol, root_p, ctx_deg_RH, ctx_rhy_RH, "RH", tex_id_RH)
        notesLH, ctx_deg_LH, ctx_rhy_LH = decode_measure(models, req, req.chord_symbol, root_p, ctx_deg_LH, ctx_rhy_LH, "LH", tex_id_LH)
        for n in notesRH: measRH.append(n)
        for n in notesLH: measLH.append(n)
        partRH.append(measRH); partLH.append(measLH)

    s.insert(0, partRH); s.insert(0, partLH)
    return s

# --------------------- Exporters ---------------------
def export_musicxml(s:stream.Score, out_path:str)->Tuple[str,str]:
    s.write('musicxml', fp=out_path)
    try:
        with open(out_path,"r",encoding="utf-8",errors="ignore") as f:
            xml_text=f.read()
    except Exception:
        xml_text="(Could not read MusicXML text—file saved.)"
    return out_path, xml_text

def export_midi(s:stream.Score, out_path:str)->str:
    s.write('midi', fp=out_path); return out_path

def export_lilypond_text(s:stream.Score, out_path:str)->Tuple[str,str]:
    def duration_to_lily(q:float)->str:
        mapping={4.0:"1",2.0:"2",1.0:"4",0.5:"8",0.25:"16",0.125:"32"}
        if q in mapping: return mapping[q]
        for base,tok in [(2.0,"2"),(1.0,"4"),(0.5,"8"),(0.25,"16")]:
            if abs(q-base*1.5)<1e-6: return tok+"."
        denom_pow=max(0, min(6, round(math.log2(1.0/max(1e-6,q))+2)))
        return str(int(2**denom_pow))
    def part_to_lily(p:stream.Part)->str:
        lines=[]
        for meas in p.getElementsByClass(stream.Measure):
            toks=[]
            for el in meas.notesAndRests:
                if isinstance(el, note.Note):
                    toks.append(f"{lily_from_pitch(el.pitch.midi)}{duration_to_lily(float(el.duration.quarterLength))}")
                elif isinstance(el, chord.Chord):
                    mids=[n.midi for n in el.notes]
                    toks.append(f"<{' '.join(lily_from_pitch(m) for m in mids)}>{duration_to_lily(float(el.duration.quarterLength))}")
                else:
                    toks.append(f"r{duration_to_lily(float(el.duration.quarterLength))}")
            lines.append(" ".join(toks))
        return " |\n  ".join(lines)
    header = r"""
\version "2.24.0"
\paper { indent = 0\mm }
\header { title = "Generated Score" composer = "Markov Fusion Engine" }
"""
    parts=list(s.parts) if len(s.parts)>0 else [s]
    rh=part_to_lily(parts[0]); lh=part_to_lily(parts[1] if len(parts)>1 else parts[0])
    text = header + r"""
\score { \new PianoStaff <<
  \new Staff \relative c' { \clef treble
""" + "  " + rh + r"""
  }
  \new Staff \relative c { \clef bass
""" + "  " + lh + r"""
  }
>> \layout { } }
"""
    with open(out_path,"w",encoding="utf-8") as f: f.write(text)
    return out_path, text

# --------------------- Orchestration ---------------------
def run(req:ScoreRequest, out_dir:str)->Dict[str,str|None]:
    ensure_dir(out_dir)
    corpus=ingest_books(req)
    models=train_models(corpus, req.style_weights, seed=req.seed)
    score=generate_score(models, req)

    base=f"{sanitize_filename(req.chord_symbol)}_{sanitize_mode(req.mode)}_{req.staves}staves"
    musicxml_fp=os.path.join(out_dir, base+".musicxml")
    midi_fp    =os.path.join(out_dir, base+".mid")
    lily_fp    =os.path.join(out_dir, base+".ly")

    print("[INFO] Exporting MusicXML..."); musicxml_fp, musicxml_text = export_musicxml(score, musicxml_fp)
    print("[INFO] Exporting MIDI...");     export_midi(score, midi_fp)
    print("[INFO] Exporting LilyPond..."); lily_fp, lily_text = export_lilypond_text(score, lily_fp)
    return {"musicxml":musicxml_fp,"musicxml_text":musicxml_text,
            "midi":midi_fp,"lilypond":lily_fp,"lilypond_text":lily_text}

# --------------------- UI ---------------------
def launch_ui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText

    root=tk.Tk(); root.title("Markov Fusion Composer (PDF OMR)"); root.geometry("1000x720")

    v_books=tk.StringVar(value="")
    v_out  =tk.StringVar(value="")
    v_chord=tk.StringVar(value="Cmaj7")
    v_mode =tk.StringVar(value="Dorian")
    v_staves=tk.IntVar(value=64)
    v_tempo=tk.IntVar(value=120)
    v_meter=tk.StringVar(value="4/4")
    v_jazz =tk.DoubleVar(value=0.45)
    v_latin=tk.DoubleVar(value=0.25)
    v_class=tk.DoubleVar(value=0.20)
    v_blues=tk.DoubleVar(value=0.10)
    v_seed =tk.StringVar(value="")
    v_use_omr = tk.BooleanVar(value=False)
    v_omr_tool= tk.StringVar(value="")
    v_workers = tk.IntVar(value=1)
    status=tk.StringVar(value="Ready.")

    top=ttk.Frame(root,padding=10); top.pack(fill="x")
    def add_row(lbl, widget):
        r=ttk.Frame(top); r.pack(fill="x",pady=4)
        ttk.Label(r,text=lbl,width=18).pack(side="left")
        widget.pack(side="left",fill="x",expand=True)
        return r

    r1=ttk.Frame(top); r1.pack(fill="x",pady=4)
    ttk.Label(r1,text="Books folder",width=18).pack(side="left")
    e_books=ttk.Entry(r1,textvariable=v_books); e_books.pack(side="left",fill="x",expand=True)
    ttk.Button(r1,text="Browse…",command=lambda: v_books.set(filedialog.askdirectory() or v_books.get())).pack(side="left",padx=6)

    r2=ttk.Frame(top); r2.pack(fill="x",pady=4)
    ttk.Label(r2,text="Output folder",width=18).pack(side="left")
    e_out=ttk.Entry(r2,textvariable=v_out); e_out.pack(side="left",fill="x",expand=True)
    ttk.Button(r2,text="Browse…",command=lambda: v_out.set(filedialog.askdirectory() or v_out.get())).pack(side="left",padx=6)

    add_row("Chord", ttk.Entry(top,textvariable=v_chord))
    rmode=ttk.Frame(top); rmode.pack(fill="x",pady=4)
    ttk.Label(rmode,text="Mode",width=18).pack(side="left")
    ttk.OptionMenu(rmode, v_mode, v_mode.get(), *[m.capitalize() for m in MODES.keys()]).pack(side="left")

    rnum=ttk.Frame(top); rnum.pack(fill="x",pady=4)
    ttk.Label(rnum,text="Staves (32–108)",width=18).pack(side="left")
    ttk.Scale(rnum,from_=32,to=108,orient="horizontal",variable=v_staves).pack(side="left",fill="x",expand=True)

    rtempo=ttk.Frame(top); rtempo.pack(fill="x",pady=4)
    ttk.Label(rtempo,text="Tempo (BPM)",width=18).pack(side="left")
    ttk.Spinbox(rtempo,from_=40,to=240,textvariable=v_tempo,width=8).pack(side="left")
    ttk.Label(rtempo,text="Meter").pack(side="left",padx=8)
    ttk.Entry(rtempo,textvariable=v_meter,width=8).pack(side="left")

    rstyle=ttk.LabelFrame(top,text="Style Weights (auto-normalized)"); rstyle.pack(fill="x",pady=8)
    def style_row(name,var):
        fr=ttk.Frame(rstyle); fr.pack(fill="x",pady=2)
        ttk.Label(fr,text=name,width=10).pack(side="left")
        ttk.Spinbox(fr,from_=0.0,to=1.0,increment=0.05,textvariable=var,width=6).pack(side="left")
    style_row("Jazz",v_jazz); style_row("Latin",v_latin); style_row("Classical",v_class); style_row("Blues",v_blues)

    # OMR config
    omr_fr=ttk.LabelFrame(top,text="PDF OMR (PDF → MusicXML)"); omr_fr.pack(fill="x",pady=6)
    ttk.Checkbutton(omr_fr,text="Enable OMR (convert PDF scores to MusicXML)",variable=v_use_omr).pack(anchor="w",padx=6,pady=2)
    frt=ttk.Frame(omr_fr); frt.pack(fill="x",pady=2)
    ttk.Label(frt,text="OMR Tool Path / Command",width=22).pack(side="left")
    ttk.Entry(frt,textvariable=v_omr_tool).pack(side="left",fill="x",expand=True)
    ttk.Button(frt,text="…",command=lambda: v_omr_tool.set(filedialog.askopenfilename() or v_omr_tool.get())).pack(side="left",padx=6)
    frw=ttk.Frame(omr_fr); frw.pack(fill="x",pady=2)
    ttk.Label(frw,text="OMR workers",width=22).pack(side="left")
    ttk.Spinbox(frw,from_=1,to=8,textvariable=v_workers,width=6).pack(side="left")

    actions=ttk.Frame(top); actions.pack(fill="x",pady=6)
    btn_gen=ttk.Button(actions,text="Generate")
    btn_open=ttk.Button(actions,text="Open Output Folder", command=lambda: (v_out.get() and webbrowser.open(v_out.get())))
    btn_gen.pack(side="left"); btn_open.pack(side="left",padx=8)
    ttk.Label(top,textvariable=status).pack(anchor="w",pady=4)

    # Tabs for text outputs
    from tkinter.scrolledtext import ScrolledText
    nb=ttk.Notebook(root); nb.pack(fill="both",expand=True,padx=10,pady=10)
    tab_ly=ttk.Frame(nb); nb.add(tab_ly,text="LilyPond (.ly)")
    txt_ly=ScrolledText(tab_ly,wrap="none",height=20); txt_ly.pack(fill="both",expand=True,padx=6,pady=6)
    ttk.Button(tab_ly,text="Copy .ly",command=lambda:(root.clipboard_clear(),root.clipboard_append(txt_ly.get("1.0","end-1c")))).pack(anchor="e",padx=6,pady=(0,6))
    tab_xml=ttk.Frame(nb); nb.add(tab_xml,text="MusicXML (raw)")
    txt_xml=ScrolledText(tab_xml,wrap="none",height=20); txt_xml.pack(fill="both",expand=True,padx=6,pady=6)
    ttk.Button(tab_xml,text="Copy XML",command=lambda:(root.clipboard_clear(),root.clipboard_append(txt_xml.get("1.0","end-1c")))).pack(anchor="e",padx=6,pady=(0,6))

    def on_generate():
        try:
            out_dir=v_out.get().strip()
            if not out_dir:
                from tkinter import messagebox
                messagebox.showerror("Missing output folder","Please select an output folder.")
                return
            ensure_dir(out_dir)

            weights={"jazz":float(v_jazz.get()),"latin":float(v_latin.get()),"classical":float(v_class.get()),"blues":float(v_blues.get())}
            tot=sum(weights.values()) or 1.0
            for k in weights: weights[k]=weights[k]/tot

            req=ScoreRequest(
                chord_symbol=v_chord.get().strip() or "C",
                mode=v_mode.get().strip(),
                staves=max(32, min(108, int(v_staves.get()))),
                meter=v_meter.get().strip() or "4/4",
                tempo_bpm=int(v_tempo.get()),
                style_weights=weights,
                books_path=v_books.get().strip() or None,
                grand_staff=True,
                seed=int(v_seed.get()) if v_seed.get().strip() else None,
                use_omr=bool(v_use_omr.get()),
                omr_tool=v_omr_tool.get().strip() or None,
                omr_out=None,
                omr_workers=int(v_workers.get())
            )
            if req.use_omr and not req.omr_tool:
                from tkinter import messagebox
                messagebox.showerror("OMR tool not set", "Please set OMR Tool Path / Command or disable OMR.")
                return

            status.set("Generating… PDF→MusicXML if enabled; then training; then writing files.")
            root.update_idletasks()
            outputs=run(req, out_dir)
            txt_ly.delete("1.0","end");  txt_ly.insert("1.0", outputs.get("lilypond_text",""))
            txt_xml.delete("1.0","end"); txt_xml.insert("1.0", outputs.get("musicxml_text",""))
            status.set("Done. Files saved.")
        except Exception as e:
            status.set("Error.")
            from tkinter import messagebox
            messagebox.showerror("Generation failed", str(e))

    btn_gen.config(command=on_generate)
    root.mainloop()

# --------------------- CLI ---------------------
def main():
    ap=argparse.ArgumentParser(description="Grand-staff generator with PDF OMR training (UI optional).")
    ap.add_argument("--ui", action="store_true", help="Launch the Tkinter UI.")
    ap.add_argument("--books-path", type=str, default=None, help="Folder with scores/books (PDF/MusicXML/MIDI/LY/TXT).")
    ap.add_argument("--out-dir", type=str, help="Output directory.")
    ap.add_argument("--chord", type=str, default="C", help='Chord symbol, e.g. "Cmaj7".')
    ap.add_argument("--mode", type=str, default="Dorian", help='Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian.')
    ap.add_argument("--staves", type=int, default=64, help="32–108 systems.")
    ap.add_argument("--tempo", type=int, default=120, help="Tempo BPM.")
    ap.add_argument("--meter", type=str, default="4/4", help='Time signature, e.g., "4/4".')
    ap.add_argument("--styles", type=str, default="jazz=0.4,latin=0.3,classical=0.2,blues=0.1", help='e.g., jazz=0.5,latin=0.2,classical=0.2,blues=0.1')
    ap.add_argument("--seed", type=int, default=None, help="Random seed.")
    ap.add_argument("--omr", action="store_true", help="Enable OMR for PDFs (PDF→MusicXML).")
    ap.add_argument("--omr-tool", type=str, default=None, help="OMR CLI (bat/exe) or 'java -jar path\\audiveris.jar'")
    ap.add_argument("--omr-out", type=str, default=None, help="Where to place MusicXML from OMR (defaults to temp).")
    ap.add_argument("--omr-workers", type=int, default=1, help="Parallel OMR workers (best effort)")
    args=ap.parse_args()

    if args.ui:
        launch_ui(); return

    if not args.out_dir:
        print("ERROR: --out-dir is required in CLI mode (or use --ui).", file=sys.stderr); sys.exit(2)

    req=ScoreRequest(
        chord_symbol=args.chord,
        mode=args.mode,
        staves=max(32, min(108, int(args.staves))),
        meter=args.meter,
        tempo_bpm=int(args.tempo),
        style_weights=parse_styles(args.styles),
        books_path=args.books_path,
        grand_staff=True,
        seed=args.seed,
        use_omr=bool(args.omr),
        omr_tool=args.omr_tool,
        omr_out=args.omr_out,
        omr_workers=max(1, int(args.omr_workers))
    )
    outputs=run(req, args.out_dir)
    print("\nDone. Files:")
    for k,v in outputs.items():
        if k.endswith("_text"): continue
        print(f"  {k}: {v}")

if __name__=="__main__":
    main()
