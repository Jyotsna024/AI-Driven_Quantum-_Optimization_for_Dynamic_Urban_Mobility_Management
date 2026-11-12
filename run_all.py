"""
run_all.py – Single entry point for the entire capstone pipeline.

Usage:
  python run_all.py --phase 1          # run Phase 1 only
  python run_all.py --phase 1 2 3      # run phases 1, 2, 3
  python run_all.py --all              # run all phases in order
  python run_all.py --smoke            # quick sanity run (short simulations)
"""
import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/run_all.log"),
    ]
)
log = logging.getLogger(__name__)

PHASES = {
    1: "network_builder",
    2: "scenario_runner",
    3: "stgnn",
    4: "cost_model",
    5: "qubo_builder",
    6: "solvers",
    7: "controller",
    8: "evaluator",
}


def run_phase(phase_num: int, smoke: bool):
    module_name = PHASES.get(phase_num)
    if module_name is None:
        raise ValueError(f"Unknown phase: {phase_num}")
    log.info(f"=== Starting Phase {phase_num}: {module_name} ===")
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    mod = __import__(module_name)
    if hasattr(mod, "main"):
        if smoke and "smoke" in mod.main.__code__.co_varnames:
            mod.main(smoke=smoke)
        else:
            mod.main()
    else:
        log.warning(f"Module {module_name} has no main() function")
    log.info(f"=== Phase {phase_num} complete ===")


def main():
    Path("logs").mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(description="Capstone pipeline entry point")
    parser.add_argument("--phase", nargs="+", type=int, help="Phase(s) to run")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--smoke", action="store_true", help="Quick test mode")
    args = parser.parse_args()

    if args.all:
        phases = sorted(PHASES.keys())
    elif args.phase:
        phases = args.phase
    else:
        parser.print_help()
        sys.exit(1)

    for p in phases:
        run_phase(p, args.smoke)

    log.info("All requested phases complete.")


if __name__ == "__main__":
    main()
