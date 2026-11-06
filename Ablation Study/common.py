# New: versioned ablation utilities; never edit original v3 or old artifacts.
from pathlib import Path
from contextlib import contextmanager
import hashlib, json, os, random, time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def fingerprints():
    return {p.relative_to(HERE).as_posix(): digest(p) for p in sorted(HERE.rglob('*.py'))
            if '__pycache__' not in p.parts and 'outputs' not in p.parts}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, value, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if replace:
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(text, encoding='utf-8'); os.replace(temp, path)
    else:
        with path.open('x', encoding='utf-8') as f: f.write(text)

def save_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: torch.save(value, f)

def load_torch(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_for(name):
    if name == 'auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)

def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)

@contextmanager
def lock(path):
    path = Path(path)
    try:
        with path.open('x', encoding='utf-8') as f:
            json.dump({'pid': os.getpid(), 'started': time.time()}, f)
    except FileExistsError:
        raise RuntimeError(f'Lock exists: {path}. Check the existing process; do not launch twice.')
    try: yield
    finally: path.unlink(missing_ok=True)

def load_plan(root):
    root = Path(root).resolve(); plan = read_json(root/'plan.json')
    if plan['format'] != 'ranking_ablation_v1': raise ValueError('Wrong plan format')
    if plan['source_hashes'] != fingerprints():
        raise ValueError('Source differs from the frozen plan. Create a new study/version.')
    for relative, expected in plan['artifacts'].items():
        if digest(root/relative) != expected: raise ValueError('Artifact changed: '+relative)
    return root, plan

