"""Export an interactive GitHub Pages replay from the trained FastAPI service."""

import json
from pathlib import Path

from src.api.main import STATIC_DIR, get_alert_queue, get_overview, get_transaction_dossier

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"


def main():
    overview = get_overview()
    selections = [
        get_alert_queue(limit=10, typology=None, risk_tier=None, camouflaged_only=True),
        get_alert_queue(limit=5, typology="Licit_Standard", risk_tier=None, camouflaged_only=False),
    ]
    for item in overview["typology_recall_comparison"]:
        for camouflaged_only in (False, True):
            selections.append(get_alert_queue(
                limit=5, typology=item["typology"], risk_tier=None, camouflaged_only=camouflaged_only
            ))

    alerts = {alert["tx_id"]: alert for selection in selections for alert in selection["alerts"]}
    cases = {tx_id: get_transaction_dossier(tx_id) for tx_id in alerts}
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8").replace(
        "<body ", '<body data-mode="replay" ', 1
    )
    SITE.mkdir(exist_ok=True)
    (SITE / ".nojekyll").touch()
    (SITE / "index.html").write_text(page, encoding="utf-8")
    payload = {"overview": overview, "alerts": list(alerts.values()), "cases": cases}
    (SITE / "replay.json").write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(cases)} held-out cases to {SITE}")


if __name__ == "__main__":
    main()
