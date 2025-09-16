#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
composer_gen.py — UI shows LilyPond (.ly) and MusicXML text outputs

- Ingests a folder of theory books (PDF/TXT/MD)
- Trains variable-order, constraint-aware Markov models
- Generates Grand Staff (32–108 systems) in a jazz+latin+salsa+classical+blues blend
- Exports: MusicXML (.musicxml), MIDI (.mid), LilyPond (.ly)
- UI tab shows the .ly text and the MusicXML so you can copy/paste

Install once:
  pip install music21 PyPDF2

Launch UI:
  python composer_gen.py --ui
"""

from __future__ import annotations
import argparse, os, sys, math, random, re, collections, warnings, webbrowser
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ---- Optional PDF text extraction + quiet warnings
try:
    import PyPDF2  # type: ignore
    try:
        from PyPDF2.errors import PdfReadWarning  # PyPDF2 >= 3.x
        warnings.filterwarnings("ignore", category=PdfReadWarning)
    except Exception:
        pass
except Exception:
    PyPDF2 = None

# ---- music21 (required)
try:
    from music21 import stream, note, chord, meter, tempo, clef, instrument, duration, pitch, interval, harmony
except Exception:
    print("ERROR: music21 is required. Install:  pip install music21", file=sys.stderr)
    raise

# --------------------- Data & helpers ---------------------
MODES = {
    "ionian":[0,2,4,5,7,9,11], "dorian":[0,2,3,5,7,9,10], "phrygian":[0,1,3,5,7,8,10],
    "lydian":[0,2,4,6,7,9,11], "mixolydian":[0,2,4,5,7,9,10],
    "aeolian":[0,2,3,5,7,8,10], "locrian":[0,1,3,5,6,8,10],
}
PREFERRED_RANGE = {
    "RH": (pitch.Pitch("C4"), pitch.Pitch("A6")),
    "LH": (pitch.Pitch("C2"), pitch.Pitch("E4")),
}

@dataclass
class ScoreRequest:
    chord_symbol: str = "C"
    mode: str = "Dorian"
    staves: int = 64
    meter: str = "4/4"
    tempo_bpm: int = 120
    style_weights: Dict[str, float] = field(default_factory=lambda: {"jazz":0.4,"latin":0.3,"classical":0.2,"blues":0.1})
    books_path: Optional[str] = None
    grand_staff: bool = True
    seed: Optional[int] = None

@dataclass
class Corpus:
    text_blobs: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    influence: Dict[str, float] = field(default_factory=dict)

@dataclass
class MarkovModel:
    order: int
    counts: Dict[Tuple, collections.Counter] = field(default_factory=lambda: collections.defaultdict(collections.Counter))
    def observe(self, context: Tuple, symbol):
        self.counts[context][symbol] += 1
    def sample(self, context: Tuple, temperature: float = 1.0):
        for k in range(len(context), -1, -1):
            ctx = context[-k:] if k>0 else tuple()
            if ctx in self.counts and sum(self.counts[ctx].values())>0:
                return weighted_sample(self.counts[ctx], temperature)
        all_syms = list({s for c in self.counts.values() for s in c.keys()})
        return random.choice(all_syms) if all_syms else None

@dataclass
class StyleModels:
    pitch_model: MarkovModel
    rhythm_model: MarkovModel
    texture_model: MarkovModel

def weighted_sample(counter: collections.Counter, temperature: float = 1.0):
    items = list(counter.items())
    if not items: return None
    syms, counts = zip(*items)
    weights = [c ** (1.0/max(1e-6, temperature)) for c in counts]
    total = sum(weights) or 1.0
    r, cum = random.random()*total, 0.0
    for sym, w in zip(syms, weights):
        cum += w
        if r <= cum: return sym
    return syms[-1]

def parse_styles(s: str) -> Dict[str, float]:
    out = {}
    if s:
        for chunk in s.split(","):
            if "=" in chunk:
                k,v = chunk.split("=",1)
                try: out[k.strip().lower()] = float(v)
                except: pass
    if not out: out = {"jazz":0.4,"latin":0.3,"classical":0.2,"blues":0.1}
    total = sum(out.values()) or 1.0
    return {k: v/total for k,v in out.items()}

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def sanitize_mode(name: str) -> str: return name.strip().lower()
def degrees_for_mode(mode_name: str) -> List[int]:
    n = sanitize_mode(mode_name)
    if n not in MODES: raise ValueError(f"Unsupported mode '{mode_name}'.")
    return MODES[n]
def sanitize_filename(s: str) -> str: return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def chord_from_symbol(symbol: str) -> chord.Chord:
    try:
        h = harmony.ChordSymbol(symbol)
        c = h.figureToChord()
        if len(c.pitches)>0: return c
    except Exception: pass
    m = re.search(r'([A-Ga-g][#b]?)', symbol)
    root_name = m.group(1).upper() if m else "C"
    root_pitch = pitch.Pitch(root_name + "4")
    third = interval.Interval(4).transposePitch(root_pitch)
    fifth = interval.Interval(7).transposePitch(root_pitch)
    return chord.Chord([root_pitch, third, fifth])

def lily_from_pitch(midi_num: int) -> str:
    names = ["c","cis","d","dis","e","f","fis","g","gis","a","ais","b"]
    octave = (midi_num // 12) - 1
    name = names[midi_num % 12]
    delta = octave - 4
    marks = "'"*delta if delta>0 else ","*(-delta) if delta<0 else ""
    return f"{name}{marks}"

# --------------------- Ingestion ---------------------
SUPPORTED_TEXT = (".txt", ".md")
SUPPORTED_PDF  = (".pdf",)

def ingest_books(books_path: Optional[str]) -> Corpus:
    corpus = Corpus()
    if not books_path or not os.path.isdir(books_path):
        if books_path and not os.path.isdir(books_path):
            print(f"[WARN] books_path '{books_path}' is not a directory. Skipping ingestion.", file=sys.stderr)
        return corpus
    for root, _, files in os.walk(books_path):
        for fn in files:
            fp = os.path.join(root, fn)
            ext = os.path.splitext(fn.lower())[1]
            try:
                if ext in SUPPORTED_TEXT:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        corpus.text_blobs.append(f.read())
                elif ext in SUPPORTED_PDF and PyPDF2 is not None:
                    try:
                        with open(fp, "rb") as f:
                            r = PyPDF2.PdfReader(f)
                            page_texts = []
                            for p in r.pages:
                                try: page_texts.append(p.extract_text() or "")
                                except Exception: continue
                            if any(page_texts):
                                corpus.text_blobs.append("\n".join(page_texts))
                    except Exception as e:
                        print(f"[WARN] PDF '{fp}' skipped: {e}", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Skipping '{fp}': {e}", file=sys.stderr)

    # very light bias extraction
    full = "\n".join(corpus.text_blobs).lower()
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

    corpus.tokens = re.findall(r"[A-Za-z#♯♭b0-9]+", full)
    return corpus

# --------------------- Training ---------------------
def train_models(corpus: Corpus, style_weights: Dict[str, float], seed: Optional[int]=None) -> StyleModels:
    if seed is not None: random.seed(seed)
    pitch_model  = MarkovModel(order=2)
    rhythm_model = MarkovModel(order=2)
    texture_model= MarkovModel(order=1)

    weights = style_weights.copy()
    for k,v in corpus.influence.items():
        if k in ["jazz","latin","classical","blues"]:
            weights[k] = weights.get(k,0.0)+v
    s = sum(weights.values()) or 1.0
    for k in weights: weights[k] /= s

    step = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,5),(5,4),(4,3),(3,2),(2,1),(1,0)]
    leaps_small = [(0,2),(2,4),(4,6),(6,4),(4,2),(2,0)]
    jazz_moves  = [(2,1),(4,3),(6,5),(1,2),(3,4),(5,6)]
    blues_degs  = [(2,7),(7,3),(7,4)]  # 7 = blue pool

    def add_pairs(pairs):
        for a,b in pairs:
            pitch_model.observe((a,), b)
            pitch_model.observe((a,b), b)
            for nxt in [b, max(0,min(6,b+1)), max(0,min(6,b-1))]:
                pitch_model.observe((a,b), nxt)
    add_pairs(step); add_pairs(leaps_small)
    if weights.get("jazz",0)>0:  add_pairs(jazz_moves)
    if weights.get("blues",0)>0: add_pairs(blues_degs)

    base_cells = [[1.0,1.0],[0.5]*4,[1.5,0.5],[0.5,1.0,0.5],[0.75,0.25,0.5,0.5]]
    latin_cells = [[0.5]*8,[1.0,0.5,0.5,1.0,1.0]]
    classical_cells = [[1.0]*4,[2.0,2.0]]
    jazz_cells = [[1.5,0.5,1.0],[0.5,1.0,0.5,1.0]]
    blues_cells = [[1.0,0.5,0.5,1.0]]
    def obs(cells):
        for cell in cells:
            for a,b in zip(cell, cell[1:]): rhythm_model.observe((a,), b)
            for a,b,c in zip(cell, cell[1:], cell[2:]): rhythm_model.observe((a,b), c)
    obs(base_cells); obs(classical_cells); obs(jazz_cells); obs(latin_cells); obs(blues_cells)

    for i,j in [(0,1),(1,2),(2,0),(0,3),(3,0),(0,4),(4,0)]:
        texture_model.observe((i,), j)

    return StyleModels(pitch_model, rhythm_model, texture_model)

# --------------------- Generation ---------------------
def blue_note_options(scale_pcset: List[int]) -> List[int]:
    return [pc for pc in [3,6,10] if pc not in scale_pcset]

def pick_degree_next(model: MarkovModel, ctx: List[int]) -> int:
    out = model.sample(tuple(ctx[-model.order:]))
    return out if out is not None else random.choice(list(range(7)))

def degree_to_pitch_midi(root: pitch.Pitch, mode_name: str, degree: int, octave_center: int) -> int:
    degs = degrees_for_mode(mode_name)
    base_pc = (root.midi % 12)
    if degree == 7:
        pcs = blue_note_options([(base_pc + d) % 12 for d in degs]) or [(base_pc + degs[2])%12]
        pc = random.choice(pcs)
    else:
        pc = (base_pc + degs[max(0, min(6, degree))]) % 12
    midi_guess = octave_center*12 + pc
    candidates = [midi_guess + k*12 for k in (-2,-1,0,1,2)]
    return min(candidates, key=lambda m: abs(m - (octave_center*12+base_pc)))

def voice_leading_cost(prev_midi: int, new_midi: int, prefer_step=2) -> float:
    dist = abs(new_midi - prev_midi)
    c = 0.05 * max(0, dist - prefer_step)
    if dist >= 12: c += 0.5
    return c

def avoid_parallels(prev_intervals: List[int], new_interval: int) -> float:
    penalty = 0.0
    if new_interval % 12 in (0,7):
        penalty += 1.0
        if prev_intervals and (prev_intervals[-1] % 12) in (0,7):
            penalty += 1.0
    return penalty

def range_cost(midi_num: int, hand: str) -> float:
    lo, hi = PREFERRED_RANGE["RH" if hand=="RH" else "LH"]
    if midi_num < lo.midi: return 0.02*(lo.midi - midi_num)
    if midi_num > hi.midi: return 0.02*(midi_num - hi.midi)
    return 0.0

def decode_measure(models: StyleModels, req: ScoreRequest, chord_sym: str, root_pitch: pitch.Pitch,
                   ctx_deg: List[int], ctx_rhy: List[float], hand: str, texture_id: int):
    octave_center = 6 if hand=="RH" else 3
    meter_num, meter_den = map(int, req.meter.split("/"))
    beats_left = meter_num * (4.0/ meter_den)
    out_notes: List[note.Note] = []
    prev_midi, prev_intervals = None, []

    while beats_left > 1e-6:
        dur = models.rhythm_model.sample(tuple(ctx_rhy[-models.rhythm_model.order:])) or 0.5
        if dur > beats_left: dur = max(0.25, beats_left)
        ctx_rhy.append(dur)

        candidates = [pick_degree_next(models.pitch_model, ctx_deg) for _ in range(3)]
        best_deg, best_score, best_midi = None, float("inf"), None
        for deg in candidates:
            midi_guess = degree_to_pitch_midi(root_pitch, req.mode, deg, octave_center)
            if texture_id == 1 and hand=="RH":
                try:
                    cch = chord_from_symbol(chord_sym)
                    cpcs = [p.midi % 12 for p in cch.pitches]
                    midi_guess = min([midi_guess+k for k in (-2,-1,0,1,2)],
                                     key=lambda m: min(abs((m%12)-pc) for pc in cpcs))
                except: pass
            elif texture_id == 2 and hand=="RH":
                midi_guess += random.choice([-4,-3,3,4])
            elif texture_id == 3 and hand=="LH":
                try:
                    cch = chord_from_symbol(chord_sym)
                    cps = sorted([p.midi % 12 for p in cch.pitches])
                    guide_pcs = [cps[1], cps[-1]] if len(cps)>=2 else cps
                    if guide_pcs:
                        midi_guess = (midi_guess // 12) * 12 + random.choice(guide_pcs)
                except: pass

            sc = range_cost(midi_guess, hand)
            if prev_midi is not None:
                sc += voice_leading_cost(prev_midi, midi_guess)
                sc += avoid_parallels(prev_intervals, midi_guess - prev_midi)
            if sc < best_score:
                best_score, best_deg, best_midi = sc, deg, midi_guess

        degree = best_deg if best_deg is not None else candidates[0]
        midi_num = best_midi if best_midi is not None else degree_to_pitch_midi(root_pitch, req.mode, degree, octave_center)
        n = note.Note()
        n.pitch = pitch.Pitch(midi=midi_num)
        n.duration = duration.Duration(dur)
        out_notes.append(n)
        if prev_midi is not None: prev_intervals.append(midi_num - prev_midi)
        prev_midi = midi_num
        ctx_deg.append(degree)
        beats_left -= dur

    return out_notes, ctx_deg, ctx_rhy

def generate_score(models: StyleModels, req: ScoreRequest) -> stream.Score:
    if req.seed is not None: random.seed(req.seed)
    s = stream.Score()
    partRH, partLH = stream.Part(), stream.Part()
    partRH.insert(0, instrument.Piano()); partLH.insert(0, instrument.Piano())
    partRH.insert(0, clef.TrebleClef());  partLH.insert(0, clef.BassClef())

    # meter/tempo: fresh instances for LH (no .flat.copy())
    m = meter.TimeSignature(req.meter)
    mm = tempo.MetronomeMark(number=req.tempo_bpm)
    partRH.insert(0, m); partRH.insert(0, mm)
    partLH.insert(0, meter.TimeSignature(req.meter))
    partLH.insert(0, tempo.MetronomeMark(number=req.tempo_bpm))

    try:
        ch = chord_from_symbol(req.chord_symbol); root_p = ch.root() or pitch.Pitch("C")
    except Exception: root_p = pitch.Pitch("C")

    ctx_deg_RH, ctx_rhy_RH = [0,1], [1.0,0.5]
    ctx_deg_LH, ctx_rhy_LH = [5,3], [1.0,1.0]
    tex_model = train_models(Corpus(), req.style_weights).texture_model  # reuse transitions
    tex_id_RH = tex_model.sample((0,)) or 0
    tex_id_LH = tex_model.sample((3,)) or 3

    for _ in range(max(32, min(108, req.staves))):
        measRH, measLH = stream.Measure(), stream.Measure()
        tex_id_RH = tex_model.sample((tex_id_RH,)) or tex_id_RH
        tex_id_LH = tex_model.sample((tex_id_LH,)) or tex_id_LH
        nRH, ctx_deg_RH, ctx_rhy_RH = decode_measure(models, req, req.chord_symbol, root_p, ctx_deg_RH, ctx_rhy_RH, "RH", tex_id_RH)
        nLH, ctx_deg_LH, ctx_rhy_LH = decode_measure(models, req, req.chord_symbol, root_p, ctx_deg_LH, ctx_rhy_LH, "LH", tex_id_LH)
        for n in nRH: measRH.append(n)
        for n in nLH: measLH.append(n)
        partRH.append(measRH); partLH.append(measLH)

    s.insert(0, partRH); s.insert(0, partLH)
    return s

# --------------------- Exporters ---------------------
def export_musicxml(s: stream.Score, out_path: str) -> Tuple[str, str]:
    s.write('musicxml', fp=out_path)
    # return text for UI display
    try:
        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            xml_text = f.read()
    except Exception:
        xml_text = "(Could not read MusicXML text—file saved to disk.)"
    return out_path, xml_text

def export_midi(s: stream.Score, out_path: str) -> str:
    s.write('midi', fp=out_path)
    return out_path

def export_lilypond_text(s: stream.Score, out_path: str) -> Tuple[str, str]:
    def duration_to_lily(q: float) -> str:
        mapping = {4.0:"1", 2.0:"2", 1.0:"4", 0.5:"8", 0.25:"16", 0.125:"32"}
        if q in mapping: return mapping[q]
        for base, token in [(2.0,"2"), (1.0,"4"), (0.5,"8"), (0.25,"16")]:
            if abs(q - base*1.5) < 1e-6: return token + "."
        denom_pow = max(0, min(6, round(math.log2(1.0/max(1e-6,q)) + 2)))
        return str(int(2**denom_pow))

    def part_to_lily(p: stream.Part):
        lines = []
        measures = [m for m in p.getElementsByClass(stream.Measure)]
        for mobj in measures:
            tokens = []
            for el in mobj.notesAndRests:
                if isinstance(el, note.Note):
                    tokens.append(f"{lily_from_pitch(el.pitch.midi)}{duration_to_lily(float(el.duration.quarterLength))}")
                elif isinstance(el, chord.Chord):
                    mids = [n.midi for n in el.notes]
                    tokens.append(f"<{' '.join(lily_from_pitch(m) for m in mids)}>{duration_to_lily(float(el.duration.quarterLength))}")
                else:
                    tokens.append(f"r{duration_to_lily(float(el.duration.quarterLength))}")
            lines.append(" ".join(tokens))
        return " |\n  ".join(lines)

    header = r"""
\version "2.24.0"
\paper { indent = 0\mm }
\header { title = "Generated Score" composer = "Markov Fusion Engine" }
"""
    parts = list(s.parts) if len(s.parts) > 0 else [s]
    rh = part_to_lily(parts[0])
    lh = part_to_lily(parts[1] if len(parts)>1 else parts[0])
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
    with open(out_path, "w", encoding="utf-8") as f: f.write(text)
    return out_path, text

# --------------------- Orchestration ---------------------
def run(req: ScoreRequest, out_dir: str) -> Dict[str, str | None]:
    ensure_dir(out_dir)
    corpus = ingest_books(req.books_path)
    models = train_models(corpus, req.style_weights, seed=req.seed)
    score = generate_score(models, req)

    base = f"{sanitize_filename(req.chord_symbol)}_{sanitize_mode(req.mode)}_{req.staves}staves"
    musicxml_fp = os.path.join(out_dir, base + ".musicxml")
    midi_fp     = os.path.join(out_dir, base + ".mid")
    lily_fp     = os.path.join(out_dir, base + ".ly")

    print("[INFO] Exporting MusicXML...")
    musicxml_fp, musicxml_text = export_musicxml(score, musicxml_fp)
    print("[INFO] Exporting MIDI...")
    export_midi(score, midi_fp)
    print("[INFO] Exporting LilyPond text...")
    lily_fp, lily_text = export_lilypond_text(score, lily_fp)

    return {
        "musicxml": musicxml_fp,
        "musicxml_text": musicxml_text,
        "midi": midi_fp,
        "lilypond": lily_fp,
        "lilypond_text": lily_text,
    }

# --------------------- Simple Tkinter UI ---------------------
def launch_ui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText

    root = tk.Tk()
    root.title("Markov Fusion Composer")
    root.geometry("900x680")

    # Vars
    v_books = tk.StringVar(value="")
    v_out   = tk.StringVar(value="")
    v_chord = tk.StringVar(value="Cmaj7")
    v_mode  = tk.StringVar(value="Dorian")
    v_staves= tk.IntVar(value=64)
    v_tempo = tk.IntVar(value=120)
    v_meter = tk.StringVar(value="4/4")
    v_jazz  = tk.DoubleVar(value=0.45)
    v_latin = tk.DoubleVar(value=0.25)
    v_class = tk.DoubleVar(value=0.20)
    v_blues = tk.DoubleVar(value=0.10)
    v_seed  = tk.StringVar(value="")
    status  = tk.StringVar(value="Ready.")

    # Top controls
    top = ttk.Frame(root, padding=10); top.pack(fill="x")
    def add_row(lbl, widget):
        r = ttk.Frame(top); r.pack(fill="x", pady=4)
        ttk.Label(r, text=lbl, width=16).pack(side="left")
        widget.pack(side="left", fill="x", expand=True)
        return r

    r1 = ttk.Frame(top); r1.pack(fill="x", pady=4)
    ttk.Label(r1, text="Books folder", width=16).pack(side="left")
    e_books = ttk.Entry(r1, textvariable=v_books); e_books.pack(side="left", fill="x", expand=True)
    ttk.Button(r1, text="Browse…", command=lambda: v_books.set(filedialog.askdirectory() or v_books.get())).pack(side="left", padx=6)

    r2 = ttk.Frame(top); r2.pack(fill="x", pady=4)
    ttk.Label(r2, text="Output folder", width=16).pack(side="left")
    e_out = ttk.Entry(r2, textvariable=v_out); e_out.pack(side="left", fill="x", expand=True)
    ttk.Button(r2, text="Browse…", command=lambda: v_out.set(filedialog.askdirectory() or v_out.get())).pack(side="left", padx=6)

    add_row("Chord", ttk.Entry(top, textvariable=v_chord))
    rmode = ttk.Frame(top); rmode.pack(fill="x", pady=4)
    ttk.Label(rmode, text="Mode", width=16).pack(side="left")
    ttk.OptionMenu(rmode, v_mode, v_mode.get(), *[m.capitalize() for m in MODES.keys()]).pack(side="left")

    rnum = ttk.Frame(top); rnum.pack(fill="x", pady=4)
    ttk.Label(rnum, text="Staves (32–108)", width=16).pack(side="left")
    ttk.Scale(rnum, from_=32, to=108, orient="horizontal", variable=v_staves).pack(side="left", fill="x", expand=True)

    rtempo = ttk.Frame(top); rtempo.pack(fill="x", pady=4)
    ttk.Label(rtempo, text="Tempo (BPM)", width=16).pack(side="left")
    ttk.Spinbox(rtempo, from_=40, to=240, textvariable=v_tempo, width=8).pack(side="left")
    ttk.Label(rtempo, text="Meter").pack(side="left", padx=8)
    ttk.Entry(rtempo, textvariable=v_meter, width=8).pack(side="left")

    rstyle = ttk.LabelFrame(top, text="Style Weights (auto-normalized)"); rstyle.pack(fill="x", pady=8)
    def style_row(name, var):
        fr = ttk.Frame(rstyle); fr.pack(fill="x", pady=2)
        ttk.Label(fr, text=name, width=10).pack(side="left")
        ttk.Spinbox(fr, from_=0.0, to=1.0, increment=0.05, textvariable=var, width=6).pack(side="left")
    style_row("Jazz", v_jazz); style_row("Latin", v_latin); style_row("Classical", v_class); style_row("Blues", v_blues)

    rseed = ttk.Frame(top); rseed.pack(fill="x", pady=4)
    ttk.Label(rseed, text="Seed (opt.)", width=16).pack(side="left")
    ttk.Entry(rseed, textvariable=v_seed, width=10).pack(side="left")

    # Action buttons & status
    actions = ttk.Frame(top); actions.pack(fill="x", pady=6)
    btn_generate = ttk.Button(actions, text="Generate")
    btn_open = ttk.Button(actions, text="Open Output Folder", command=lambda: (v_out.get() and webbrowser.open(v_out.get())))
    btn_generate.pack(side="left"); btn_open.pack(side="left", padx=8)
    ttk.Label(top, textvariable=status).pack(anchor="w", pady=4)

    # Notebook for text outputs
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=10, pady=10)

    tab_ly = ttk.Frame(nb); nb.add(tab_ly, text="LilyPond (.ly)")
    txt_ly = ScrolledText(tab_ly, wrap="none", height=20); txt_ly.pack(fill="both", expand=True, padx=6, pady=6)
    copy_ly = ttk.Button(tab_ly, text="Copy .ly to Clipboard", command=lambda: (root.clipboard_clear(), root.clipboard_append(txt_ly.get("1.0","end-1c"))))
    copy_ly.pack(anchor="e", padx=6, pady=(0,6))

    tab_xml = ttk.Frame(nb); nb.add(tab_xml, text="MusicXML (raw)")
    txt_xml = ScrolledText(tab_xml, wrap="none", height=20); txt_xml.pack(fill="both", expand=True, padx=6, pady=6)
    copy_xml = ttk.Button(tab_xml, text="Copy XML to Clipboard", command=lambda: (root.clipboard_clear(), root.clipboard_append(txt_xml.get("1.0","end-1c"))))
    copy_xml.pack(anchor="e", padx=6, pady=(0,6))

    def on_generate():
        try:
            out_dir = v_out.get().strip()
            if not out_dir:
                from tkinter import messagebox
                messagebox.showerror("Missing output folder", "Please select an output folder.")
                return
            ensure_dir(out_dir)
            weights = {
                "jazz": float(v_jazz.get()),
                "latin": float(v_latin.get()),
                "classical": float(v_class.get()),
                "blues": float(v_blues.get())
            }
            total = sum(weights.values()) or 1.0
            for k in weights: weights[k] = weights[k]/total

            req = ScoreRequest(
                chord_symbol=v_chord.get().strip() or "C",
                mode=v_mode.get().strip(),
                staves=max(32, min(108, int(v_staves.get()))),
                meter=v_meter.get().strip() or "4/4",
                tempo_bpm=int(v_tempo.get()),
                style_weights=weights,
                books_path=v_books.get().strip() or None,
                grand_staff=True,
                seed=int(v_seed.get()) if v_seed.get().strip() else None
            )
            status.set("Generating…")
            root.update_idletasks()
            outputs = run(req, out_dir)
            # Update text views
            txt_ly.delete("1.0","end");   txt_ly.insert("1.0", outputs.get("lilypond_text",""))
            txt_xml.delete("1.0","end");  txt_xml.insert("1.0", outputs.get("musicxml_text",""))
            status.set("Done. Files saved.")
        except Exception as e:
            status.set("Error.")
            from tkinter import messagebox
            messagebox.showerror("Generation failed", str(e))

    btn_generate.config(command=on_generate)

    root.mainloop()

# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser(description="Grand-staff generator (with UI showing text outputs).")
    ap.add_argument("--ui", action="store_true", help="Launch the Tkinter UI.")
    ap.add_argument("--books-path", type=str, default=None, help="Folder with theory books (PDF/TXT/MD).")
    ap.add_argument("--out-dir", type=str, help="Output directory.")
    ap.add_argument("--chord", type=str, default="C", help='Chord symbol, e.g. "Cmaj7".')
    ap.add_argument("--mode", type=str, default="Dorian", help='Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian.')
    ap.add_argument("--staves", type=int, default=64, help="32–108 systems.")
    ap.add_argument("--tempo", type=int, default=120, help="Tempo BPM.")
    ap.add_argument("--meter", type=str, default="4/4", help='Time signature, e.g., "4/4".')
    ap.add_argument("--styles", type=str, default="jazz=0.4,latin=0.3,classical=0.2,blues=0.1", help='e.g., "jazz=0.5,latin=0.2,classical=0.2,blues=0.1"')
    ap.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = ap.parse_args()

    if args.ui:
        launch_ui()
        return

    if not args.out_dir:
        print("ERROR: --out-dir is required in CLI mode (or use --ui).", file=sys.stderr)
        sys.exit(2)

    req = ScoreRequest(
        chord_symbol=args.chord,
        mode=args.mode,
        staves=max(32, min(108, int(args.staves))),
        meter=args.meter,
        tempo_bpm=int(args.tempo),
        style_weights=parse_styles(args.styles),
        books_path=args.books_path,
        grand_staff=True,
        seed=args.seed
    )
    outputs = run(req, args.out_dir)
    print("\nDone. Files:")
    for k,v in outputs.items():
        if k.endswith("_text"):  # skip printing the big blobs in CLI
            continue
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
