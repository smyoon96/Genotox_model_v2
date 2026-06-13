"""Minimal progress utilities."""
try:
    from tqdm import tqdm
    def pbar(iterable, **kwargs):
        return tqdm(iterable, **kwargs)
except ImportError:
    def pbar(iterable, **kwargs):
        return iterable

def step_header(text, width=60):
    return f"\n{'='*width}\n  {text}\n{'='*width}"

def task_done(text=""):
    return f"  ✓ {text}" if text else "  ✓"

def eta_str(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"
