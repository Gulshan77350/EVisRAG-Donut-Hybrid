import os
import sys
from pathlib import Path

# Windows consoles default to cp1252; model outputs contain characters it can't print.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def _disable_windows_throttling():
    """Opt this process out of Windows 11 'efficiency mode' (EcoQoS) and raise its priority.

    4-bit decoding is bound by how fast the CPU launches GPU kernels; when Windows throttles a
    background process, generation speed drops from ~13 to ~3 tokens/s.
    """
    if sys.platform != "win32" or os.environ.get("THESIS_NO_CPU_TUNING"):
        return
    try:
        import ctypes
        from ctypes import wintypes

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        k32 = ctypes.windll.kernel32
        proc = k32.GetCurrentProcess()
        # ProcessPowerThrottling = 4; EXECUTION_SPEED = 1; StateMask 0 => never throttle
        state = PowerThrottlingState(1, 1, 0)
        k32.SetProcessInformation(proc, 4, ctypes.byref(state), ctypes.sizeof(state))
        k32.SetPriorityClass(proc, 0x00000080)  # HIGH_PRIORITY_CLASS
        _pin_to_performance_cores(k32, proc)
    except Exception:
        pass


def _pin_to_performance_cores(k32, proc):
    """On hybrid Intel CPUs keep the process on P-cores (highest EfficiencyClass).

    When Windows moves the decode thread to an E-core, GPU utilisation drops to ~30%.
    """
    import ctypes

    needed = ctypes.c_ulong(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), proc, 0)
    buf = (ctypes.c_ubyte * needed.value)()
    if not k32.GetSystemCpuSetInformation(buf, needed, ctypes.byref(needed), proc, 0):
        return
    raw, off, sets = bytes(buf), 0, []
    while off < needed.value:
        size = int.from_bytes(raw[off:off + 4], "little")
        cpu_id = int.from_bytes(raw[off + 8:off + 12], "little")
        eff_class = raw[off + 18]
        sets.append((cpu_id, eff_class))
        off += size
    top = max(e for _, e in sets)
    if top == 0:  # not a hybrid CPU
        return
    ids = [i for i, e in sets if e == top]
    arr = (ctypes.c_ulong * len(ids))(*ids)
    k32.SetProcessDefaultCpuSets(proc, arr, len(ids))


_disable_windows_throttling()

# Keep all model weights on D: (C: is nearly full). Must be set before transformers is imported.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .donut_reader import DonutReader  # noqa: E402
from .evisrag_reasoner import EVisRAGReasoner  # noqa: E402
from .hybrid import HybridPipeline, build_evidence  # noqa: E402
