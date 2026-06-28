"""
Synthetic 'internal communications + records' corpus for a due-diligence demo.

We plant signal on purpose so the engine has something true to find:
  - 4 normal themes (procurement, hiring, logistics, routine finance)
  - a small hidden cluster about an off-book payment (the anomaly)
  - recurring actors who co-occur (the relationship graph)

Deterministic: same seed -> same corpus -> reproducible findings.
"""

import random
from engine.core import Document

random.seed(7)

ACTORS = ["Marcus Vale", "Lena Roth", "David Okoye", "Priya Anand",
          "Tomas Berg", "Sara Klein", "Yuki Sato", "Omar Faris"]

# appended to normal docs so no two are near-identical (real corpora have variety)
_FILLER = [
    "Noted in the weekly sync.", "Will circulate the summary by Friday.",
    "Copying the wider team for visibility.", "Low priority, revisit next sprint.",
    "Flagged to the ops channel earlier today.", "No action needed for now.",
    "Adding to the tracker.", "Discussed briefly over lunch.",
    "Follow-up call booked for Thursday.", "Looping in the regional lead.",
    "Same as last month, nothing unusual.", "Routine, closing the ticket.",
    "Updated the shared sheet accordingly.", "Awaiting sign-off, should be quick.",
    "Standard process, all good.", "Quick one, handled.",
]

_TEMPLATES = {
    "procurement": [
        "Vendor {a} confirmed delivery of office hardware, invoice REF-{n} for $12,400.",
        "{a} approved the procurement request from {b} for new laptops, refurbished batch.",
        "Purchase order to {a} processed; standard 30 day terms, account ACC-{n}.",
        "{a} flagged a duplicate invoice REF-{n}; finance to reconcile with {b}.",
    ],
    "hiring": [
        "{a} interviewed candidates for the analyst role; {b} runs the final round.",
        "Offer extended to a new hire in the data team; onboarding planned with {a}.",
        "{a} asked {b} about headcount budget and the hiring plan next quarter.",
        "Reference check done for the {a} candidate; background clean, start date agreed.",
        "{a} shared the interview scorecards with {b}; strong shortlist this cycle.",
        "Recruiter update from {a}: pipeline healthy, two senior roles still open.",
    ],
    "logistics": [
        "Shipment from the Hamburg depot delayed; {a} coordinating with carrier on 2025-08-12.",
        "{a} reports container ACC-{n} cleared customs, onward route to Lyon confirmed.",
        "Route optimisation review with {a} and {b}; fuel costs up, margins tightening.",
        "{a} escalated a warehouse capacity issue affecting the EUR 80k order.",
    ],
    "finance_routine": [
        "Monthly close completed by {a}; all ledgers reconciled, no exceptions noted.",
        "{a} submitted the quarterly VAT filing; refund of EUR 4,200 expected.",
        "Payroll run approved by {a}; {b} double-checked the totals, account ACC-{n}.",
        "{a} reviewed the expense report from {b}; within policy, approved.",
    ],
}

# The planted anomaly: an off-book side channel. Distinct vocabulary, linked actors.
_ANOMALY = [
    "Off the record {a}: move the EUR 250,000 through the {b} intermediary, no invoice, no trace.",
    "{a} confirmed the consultancy fee to ACC-99812 is a placeholder; do not log it anywhere.",
    "Keep this between us {a} — the {b} payment skips approval entirely, settle in cash, no paperwork.",
    "{a}: the shell entity in Valletta receives the transfer, route REF-77310, stay quiet about it.",
    "Per {a}, the kickback to {b} gets coded as marketing; EUR 120,000, strictly off the books.",
    "{a} delete this thread after reading; the bribe clears through the Valletta shell, untraceable.",
    "Confidential {a}: launder the surplus via the {b} intermediary, cash only, no invoice trail.",
    "{a} the off-book transfer to ACC-99812 must never appear in the ledger; settle quietly offshore.",
]


def _fill(t: str, n: int) -> str:
    a, b = random.sample(ACTORS, 2)
    base = t.format(a=a, b=b, n=1000 + n)
    return base + " " + random.choice(_FILLER)


def make_corpus() -> list[Document]:
    docs: list[Document] = []
    n = 0
    # normal traffic
    for theme, templates in _TEMPLATES.items():
        for _ in range(26):
            t = random.choice(templates)
            docs.append(Document(
                id=f"DOC-{n:04d}",
                text=_fill(t, n),
                meta={"channel": theme, "date": f"2025-08-{(n % 28) + 1:02d}"},
            ))
            n += 1
    # planted anomalies — a few actors recur to create a graph cluster
    insiders = ["Marcus Vale", "Tomas Berg"]
    for _ in range(8):
        t = random.choice(_ANOMALY)
        a = random.choice(insiders)
        b = random.choice([x for x in ACTORS if x not in insiders])
        docs.append(Document(
            id=f"DOC-{n:04d}",
            text=t.format(a=a, b=b),
            meta={"channel": "comms_misc", "date": f"2025-08-{(n % 28) + 1:02d}"},
        ))
        n += 1

    random.shuffle(docs)
    return docs
