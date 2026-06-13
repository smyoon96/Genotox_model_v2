"""
utils/progress.py — 파이프라인 전용 진행률 표기 유틸리티
=========================================================
tqdm이 없는 환경에서도 fallback으로 동작.

사용:
    from utils.progress import pbar, step_header, task_done

    for ep in pbar(ENDPOINTS, desc="Endpoint"):
        for sc in pbar(SCENARIOS, desc=f"  {ep} scenario", leave=False):
            ...
"""
import sys
import time
import logging
from typing import Iterable, Optional

logger = logging.getLogger("progress")

# tqdm 가용 여부 확인
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    logger.info("tqdm not installed — using fallback progress display")


class _FallbackTqdm:
    """
    tqdm 없을 때 사용하는 fallback.
    [██████░░░░] 39% (4/10) Endpoint [elapsed 12s]
    """
    BAR_WIDTH = 25

    def __init__(self, iterable=None, total=None, desc="", leave=True,
                 unit="it", ncols=None, **kwargs):
        self.iterable  = list(iterable) if iterable is not None else []
        self.total     = total if total is not None else len(self.iterable)
        self.desc      = desc
        self.leave     = leave
        self.n         = 0
        self._start    = time.time()
        self._last_print = -1.0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)
        if self.leave:
            self._print(force=True)
            print()          # 개행

    def update(self, n=1):
        self.n += n
        self._print()

    def set_postfix_str(self, s="", **kwargs):
        self._postfix = s

    def _print(self, force=False):
        now = time.time()
        if not force and now - self._last_print < 0.5:
            return
        self._last_print = now

        pct   = self.n / max(self.total, 1)
        done  = int(self.BAR_WIDTH * pct)
        bar   = "█" * done + "░" * (self.BAR_WIDTH - done)
        elapsed = now - self._start
        ps    = getattr(self, "_postfix", "")
        line  = (f"\r  [{bar}] {pct*100:5.1f}% ({self.n}/{self.total})"
                 f"  {self.desc}  [{elapsed:.0f}s]{' '+ps if ps else ''}")
        sys.stderr.write(line)
        sys.stderr.flush()

    def close(self):
        if self.leave:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def pbar(iterable: Iterable, desc: str = "", leave: bool = True,
         total: Optional[int] = None, unit: str = "it",
         colour: Optional[str] = None) -> Iterable:
    """
    tqdm 또는 fallback progress bar를 반환.

    파라미터
    --------
    iterable : 반복 대상
    desc     : 왼쪽 설명 레이블 (예: "Endpoint", "  ames scenario")
    leave    : 완료 후 bar 유지 여부 (서브 루프는 False 권장)
    total    : 전체 항목 수 (자동 추론 불가할 때 지정)
    colour   : tqdm 색상 (tqdm 설치 시만 적용)
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    if _HAS_TQDM:
        kwargs = dict(desc=desc, leave=leave, unit=unit, total=total,
                      dynamic_ncols=True, file=sys.stderr)
        if colour:
            kwargs["colour"] = colour
        return _tqdm(iterable, **kwargs)
    else:
        return _FallbackTqdm(iterable, total=total, desc=desc, leave=leave)


# ──────────────────────────────────────────────
#  Step 헤더 / 완료 출력 헬퍼
# ──────────────────────────────────────────────

def step_header(step_num: int, total_steps: int, description: str):
    """
    ──────────────────────────────────
    [Step 3/9] Leakage-free Training
    ──────────────────────────────────
    """
    bar = "─" * 50
    msg = f"\n{bar}\n[Step {step_num}/{total_steps}] {description}\n{bar}"
    print(msg, file=sys.stderr)
    logger.info(f"[Step {step_num}/{total_steps}] {description}")


def task_done(label: str, elapsed: float, extra: str = ""):
    """  ✓ Endpoint [ames] — 12.3s   MCC=0.624"""
    msg = f"  ✓ {label} [{elapsed:.1f}s]{' — ' + extra if extra else ''}"
    print(msg, file=sys.stderr)
    logger.info(msg)


def eta_str(elapsed: float, done: int, total: int) -> str:
    """ETA 문자열 반환."""
    if done == 0:
        return "ETA: ?"
    rate = elapsed / done
    remaining = rate * (total - done)
    if remaining < 60:
        return f"ETA: {remaining:.0f}s"
    return f"ETA: {remaining/60:.1f}min"
