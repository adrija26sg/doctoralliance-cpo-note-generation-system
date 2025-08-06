#!/usr/bin/env python3
import os
import sys
import random
import requests
import calendar
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AzureOpenAI, APITimeoutError, OpenAIError

# ─── LOAD ENVIRONMENT VARIABLES ────────────────────────────────────────────────
load_dotenv()
BASE_URL                = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api"
DA_API_KEY              = os.getenv("DA_API_KEY")
AZURE_OPENAI_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_VERSION    = os.getenv("AZURE_OPENAI_VERSION", "2023-05-15")

if not (DA_API_KEY and AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT):
    raise RuntimeError(
        "❌ Please set DA_API_KEY, AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT in your .env"
    )

DRY_RUN = True  # if set to FALSE will POST

# ─── INIT AZURE OPENAI CLIENT ──────────────────────────────────────────────────
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version=AZURE_OPENAI_VERSION,
)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def _auth_hdrs():
    return {"Authorization": f"Bearer {DA_API_KEY}", "Content-Type": "application/json"}

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%m/%d/%Y")

def random_date(start: datetime, end: datetime) -> datetime:
    secs = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, secs))

# ─── FETCH EXISTING CPO MINUTES ────────────────────────────────────────────────
def get_existing_cpo_minutes(pid: str, month: str) -> int:
    ml   = " ".join(month.split()).title()
    mdt  = datetime.strptime(ml, "%B %Y")
    first = datetime(mdt.year, mdt.month, 1)
    last  = datetime(
        mdt.year, mdt.month,
        calendar.monthrange(mdt.year, mdt.month)[1],
        23, 59, 59
    )
    resp = requests.get(f"{BASE_URL}/CCNotes/patient/{pid}", headers=_auth_hdrs())
    resp.raise_for_status()
    total = 0
    for note in resp.json() or []:
        ts = note.get("updatedOn") or note.get("createdAt")
        if not ts:
            continue
        dt = datetime.fromisoformat(ts.rstrip("Z"))
        if first <= dt <= last:
            total += int(note.get("cpOmin") or 0)
    return total

# ─── FETCH ORDERS & CERTIFICATION ───────────────────────────────────────────────
def get_orders(pid: str) -> list[dict]:
    r = requests.get(f"{BASE_URL}/Order/patient/{pid}", headers=_auth_hdrs())
    r.raise_for_status()
    return r.json() or []

def find_cert_order(orders: list[dict]) -> dict | None:
    for o in orders:
        da = (o.get("daOrderType") or "").lower()
        dn = (o.get("documentName") or "").lower()
        if da.startswith("485") or "485" in dn or "recert" in da or "recert" in dn:
            return o
    return None

def get_cert_info(pid: str) -> dict:
    r = requests.get(f"{BASE_URL}/Patient/total/{pid}", headers=_auth_hdrs())
    r.raise_for_status()
    return r.json() or {}

# ─── FETCH ALL CCNOTES ───────────────────────────────────────────────────────────
def get_ccnotes(pid: str) -> list[dict]:
    r = requests.get(f"{BASE_URL}/CCNotes/patient/{pid}", headers=_auth_hdrs())
    r.raise_for_status()
    return r.json() or []

# ─── VALIDATION PROMPT ──────────────────────────────────────────────────────────
def validate_note(group: str, title: str, text: str,
                  icd: list[str], phys: str) -> str:
    prompt = (
        f"Validate this care-coordination note as a home-health physician.\n"
        f"Category: {group}\n\n"
        f"Diagnoses (no codes): {', '.join(icd[:5])}\n"
        f"Physician statement: {phys}\n\n"
        f"Note Title: {title}\n\n"
        f"Note Text: {text}\n\n"
        "1) Is the note TEXT medically sound and relevant to the certified conditions and category?\n"
        "2) Does the NOTE TITLE accurately summarize the text?\n\n"
        "Reply: VALID or INVALID: <reasons>"
    )
    try:
        res = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a home-health physician reviewer."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0,
            max_tokens=300,
            timeout=60
        )
        return res.choices[0].message.content.strip()
    except (APITimeoutError, OpenAIError) as e:
        return f"❌ Validation error: {e}"

# ─── MAIN WORKFLOW ──────────────────────────────────────────────────────────────
def main(patient_id: str, month_label: str):
    # 1) Check billed minutes
    existing = get_existing_cpo_minutes(patient_id, month_label)
    print(f"\n▶ Patient {patient_id} | {month_label} | {existing} min existing\n")
    if existing >= 30:
        print("✔ Already ≥30 min billed. Exiting.")
        return

    # 2) Fetch certification order
    orders = get_orders(patient_id)
    cert   = find_cert_order(orders)
    if not cert:
        print("⚠️ No 485/recert order found; cannot validate.")
        return

    soc      = parse_date(cert["startOfCare"])
    ep_start = parse_date(cert["episodeStartDate"])
    ep_end   = parse_date(cert["episodeEndDate"])

    # 3) Compute validation window
    ml   = " ".join(month_label.split()).title()
    mdt  = datetime.strptime(ml, "%B %Y")
    window_start = max(soc, datetime(mdt.year, mdt.month, 1))
    month_end    = datetime(
        mdt.year, mdt.month,
        calendar.monthrange(mdt.year, mdt.month)[1],
        23, 59, 59
    )
    window_end   = min(ep_end, month_end)

    # 4) Fetch ICDs & physician statement
    agency = get_cert_info(patient_id).get("agencyInfo", {})
    icd    = agency.get("icdCodes", [])
    phys   = agency.get("physicianCertification", "")

    # 5) Fetch & validate all CCNotes
    notes     = [n for n in get_ccnotes(patient_id) if n.get("entityType") == "CCNote"]
    processed = []

    if not notes:
        print("⚠️ No CCNotes found.")
        return

    print(f"Validating {len(notes)} existing notes...\n" + "="*60)
    for note in notes:
        group     = note.get("noteType", "Unknown")
        title     = note.get("noteTitle", "")
        text      = note.get("noteText", "")
        send_date = random_date(window_start, window_end).strftime("%m/%d/%Y")

        verdict = validate_note(group, title, text, icd, phys)

        print(f"SentToPhysicianDate: {send_date}")
        print(f"NoteType:           {group}")
        print(f"Note Title:         {title}")
        print(f"Note Text:          {text}\n")
        print("Validation Result:")
        print(verdict)
        print("\n" + "-"*60 + "\n")

        processed.append({"title": title, "text": text, "verdict": verdict})

    # ─── DUPLICATE FLAGGING ─────────────────────────────────────────────────
    seen_titles = {}
    seen_snips  = {}
    print("\n===== DUPLICATE CHECK =====")
    for idx, rec in enumerate(processed):
        t_lower = rec["title"].strip().lower()
        snippet = " ".join(rec["text"].split()[:10]).lower()
        if t_lower in seen_titles:
            print(f"❌ Duplicate TITLE between notes {seen_titles[t_lower]} and {idx}: “{rec['title']}”")
        else:
            seen_titles[t_lower] = idx

        if snippet in seen_snips:
            print(f"❌ Duplicate TEXT snippet between notes {seen_snips[snippet]} and {idx}: “{snippet}…”")
        else:
            seen_snips[snippet] = idx

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_cpo.py <patientId> <Month Year>")
        sys.exit(1)
    _, pid, month = sys.argv
    main(pid, month)
