"""Daily tracker for the frozen min-variance and max-Sharpe Nifty 50 portfolios.

    python tracker.py init      once: freeze target weights, fit the VaR models
    python tracker.py update    each evening: add new trading days, check revisions

See the "Daily tracker" section of README.md.
"""

import argparse
import sys
from pathlib import Path

from daily_tracker.dashboard import build as build_dashboard
from daily_tracker.data import now_ist
from daily_tracker.init import run_init
from daily_tracker.reoptimise import run_weekly
from daily_tracker.store import Store
from daily_tracker.update import run_update


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="freeze targets and fit the VaR models (once)")
    init.add_argument("--benchmark", choices=["nsei", "tri"], default="nsei",
                      help="nsei = ^NSEI price index (default); tri = Nifty 50 TRI from "
                           "data/nifty50_tri.csv")
    init.add_argument("--force", action="store_true",
                      help="overwrite an existing setup and start a new record")
    sub.add_parser("update", help="add new trading days and correct revisions")
    sub.add_parser("dashboard", help="build the dashboard page into site/")
    args = parser.parse_args()

    store = Store(Path(__file__).resolve().parent)
    now = now_ist()
    if args.command == "dashboard":
        print(f"Wrote {build_dashboard(store, now, store.root / 'site')}")
        return 0
    if args.command == "init":
        models, targets = run_init(store, now, args.benchmark, args.force)
        for key, m in models.items():
            print(f"{m['name']}: {len(targets[key])} stocks, fitted {m['fit_start']} to "
                  f"{m['fit_end']} ({m['n_obs']} days), persistence {m['persistence']:.3f}, "
                  f"next-day 99% VaR {m['var_99_next']:.2%}")
        print(f"Inception {store.load_config()['inception_date']}, value 100.")
        return 0
    status = run_update(store, now, weekly=run_weekly)
    log = store.read("job_log.csv").iloc[-1]
    print(f"{status}: {log.dates_added} date(s) added")
    for field in ["revisions", "warnings", "errors"]:
        if isinstance(log[field], str) and log[field]:
            print(f"{field}:\n  " + log[field].replace(" | ", "\n  "))
    return 1 if status == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
