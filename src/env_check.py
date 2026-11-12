"""env_check.py – Runtime environment verification. Run at start of any phase."""
import os
import sys
import subprocess
import platform
from pathlib import Path


def check_sumo():
    sumo_home = os.environ.get("SUMO_HOME", "")
    sumo_bin = Path(sumo_home) / "bin" / "sumo.exe"
    netconvert = Path(sumo_home) / "bin" / "netconvert.exe"
    ok = sumo_bin.exists() and netconvert.exists()
    return {
        "SUMO_HOME": sumo_home,
        "sumo.exe": str(sumo_bin),
        "netconvert.exe": str(netconvert),
        "ok": ok,
    }


def check_traci():
    try:
        import traci
        return {"ok": True, "version": traci.__version__}
    except ImportError as e:
        return {"ok": False, "error": str(e)}


def check_torch():
    try:
        import torch
        return {
            "ok": True,
            "version": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except ImportError as e:
        return {"ok": False, "error": str(e)}


def check_pyg():
    try:
        import torch_geometric
        return {"ok": True, "version": torch_geometric.__version__}
    except ImportError as e:
        return {"ok": False, "error": str(e)}


def check_quantum():
    results = {}
    for pkg, name in [
        ("dimod", "dimod"),
        ("neal", "dwave-neal"),
        ("qiskit", "qiskit"),
        ("qiskit_aer", "qiskit-aer"),
        ("qiskit_algorithms", "qiskit-algorithms"),
    ]:
        try:
            mod = __import__(pkg)
            results[name] = {"ok": True, "version": getattr(mod, "__version__", "?")}
        except ImportError as e:
            results[name] = {"ok": False, "error": str(e)}
    # Token checks (never print actual tokens)
    results["DWAVE_API_TOKEN_set"] = bool(os.environ.get("DWAVE_API_TOKEN"))
    results["IBMQ_TOKEN_set"] = bool(
        os.environ.get("IBMQ_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
    )
    return results


def run_all():
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    s = check_sumo()
    print(f"\nSUMO: {'OK' if s['ok'] else 'FAIL'}")
    for k, v in s.items():
        print(f"  {k}: {v}")

    t = check_traci()
    print(f"\nTraCI: {'OK' if t['ok'] else 'FAIL'} – {t}")

    th = check_torch()
    print(f"\nPyTorch: {'OK' if th['ok'] else 'FAIL'} – {th}")

    pyg = check_pyg()
    print(f"\nPyG: {'OK' if pyg['ok'] else 'FAIL'} – {pyg}")

    q = check_quantum()
    print("\nQuantum stack:")
    for k, v in q.items():
        print(f"  {k}: {v}")

    # Overall pass/fail
    critical = [s["ok"], t["ok"], th["ok"]]
    if all(critical):
        print("\n[ENV CHECK] PASS – all critical components present")
        return True
    else:
        print("\n[ENV CHECK] PARTIAL – some packages missing (see above)")
        return False


if __name__ == "__main__":
    run_all()
