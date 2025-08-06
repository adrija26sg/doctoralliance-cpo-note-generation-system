#!/usr/bin/env python3
import os
import sys
import math
import random
import requests
import calendar
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AzureOpenAI, APITimeoutError, OpenAIError

# ─── LOAD ENV ───────────────────────────────────────────────────────────────────
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

DRY_RUN = True  # set to False to actually POST

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
    ml    = " ".join(month.split()).title()
    m_dt  = datetime.strptime(ml, "%B %Y")
    first = datetime(m_dt.year, m_dt.month, 1)
    last  = datetime(
        m_dt.year,
        m_dt.month,
        calendar.monthrange(m_dt.year, m_dt.month)[1],
        23, 59, 59
    )
    r = requests.get(f"{BASE_URL}/CCNotes/patient/{pid}", headers=_auth_hdrs()); r.raise_for_status()
    total = 0
    for n in r.json() or []:
        ts = n.get("updatedOn") or n.get("createdAt")
        if not ts: continue
        dt = datetime.fromisoformat(ts.rstrip("Z"))
        if first <= dt <= last:
            total += int(n.get("cpOmin") or 0)
    return total

# ─── FETCH ORDERS & CERTIFICATION ───────────────────────────────────────────────
def get_orders(pid: str) -> list[dict]:
    r = requests.get(f"{BASE_URL}/Order/patient/{pid}", headers=_auth_hdrs()); r.raise_for_status()
    return r.json() or []

def find_cert_order(orders: list[dict]) -> dict | None:
    for o in orders:
        da = (o.get("daOrderType") or "").lower()
        dn = (o.get("documentName")  or "").lower()
        if da.startswith("485") or "485" in dn or "recert" in da or "recert" in dn:
            return o
    return None

def get_cert_info(pid: str) -> dict:
    r = requests.get(f"{BASE_URL}/Patient/total/{pid}", headers=_auth_hdrs()); r.raise_for_status()
    return r.json() or {}

# ─── FETCH EXISTING NOTES ───────────────────────────────────────────────────────
def get_ccnotes(pid: str) -> list[dict]:
    r = requests.get(f"{BASE_URL}/CCNotes/patient/{pid}", headers=_auth_hdrs()); r.raise_for_status()
    return r.json() or []

# ─── GENERATION ─────────────────────────────────────────────────────────────────
def build_gen_prompt(icd: list[str], phys: str, cnt: int) -> str:
    return (
        f"Patient Diagnoses (do NOT print numeric ICD-10 codes): {', '.join(icd[:5])}\n"
        f"Physician Certification Statement:\n{phys}\n\n"
        f"Please generate {cnt} well-formed care-coordination notes as a nurse, each ~3 minutes "
        "(about 100–150 words). For each note, output:\n"
        "NoteTitle: <concise title>\n"
        "NoteText: <narrative>\n\n"
        "Separate notes with two blank lines.\n"
    )

def generate_notes(icd: list[str], phys: str, cnt: int) -> list[tuple[str,str]]:
    prompt = build_gen_prompt(icd, phys, cnt)
    for _ in range(2):
        try:
            res = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role":"system","content":"You are a skilled nurse documentation assistant."},
                    {"role":"user",  "content":prompt}
                ],
                temperature=0.7,
                max_tokens=600,
                timeout=60
            )
            raw = res.choices[0].message.content.strip()
            notes = []
            for block in raw.split("\n\n"):
                if block.strip().startswith("NoteTitle:"):
                    lines = block.splitlines()
                    title = lines[0].split(":",1)[1].strip()
                    text  = lines[1].split(":",1)[1].strip() if len(lines)>1 else ""
                    notes.append((title, text))
            return notes
        except APITimeoutError:
            continue
        except OpenAIError as e:
            print("❌ Generation error:", e)
            return []
    print("❌ Generation timed out twice; aborting.")
    return []

# ─── VALIDATION ─────────────────────────────────────────────────────────────────
def validate_note(group: str, title: str, text: str,
                  icd: list[str], phys: str) -> str:
    prompt = (
        f"Validate this care-coordination note as a home-health physician.\n"
        f"Category: {group}\n\n"
        f"Diagnoses (no codes): {', '.join(icd[:5])}\n"
        f"Physician stmt: {phys}\n\n"
        f"Title: {title}\n\n"
        f"Text: {text}\n\n"
        "1) Does the title reflect the text?\n"
        "2) Is it medically sound?\n"
        "3) Is it relevant to the category?\n\n"
        "Reply: VALID or INVALID: <reasons>"
    )
    try:
        res = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role":"system","content":"You are a home-health physician reviewer."},
                {"role":"user",  "content":prompt}
            ],
            temperature=0,
            max_tokens=300,
            timeout=60
        )
        return res.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"❌ Validation error: {e}"

# ─── POST (DRY RUN) ─────────────────────────────────────────────────────────────
def post_note(pid, soc, eps, epe, note_type, title, text, send_date):
    if DRY_RUN:
        return True
    body = {
        "patientId": pid,
        "startOfCare": soc,
        "startOfEpisode": eps,
        "endOfEpisode": epe,
        "noteType": note_type,
        "noteTitle": title,
        "noteText": text,
        "cpOmin": 3,
        "sentToPhysicianDate": send_date,
        "sentToPhysicianStatus": False
    }
    r = requests.post(f"{BASE_URL}/CCNotes/patient/{pid}", json=body, headers=_auth_hdrs())
    return r.status_code < 300

# ─── MAIN WORKFLOW ──────────────────────────────────────────────────────────────
def main(pid: str, month: str):
    existing = get_existing_cpo_minutes(pid, month)
    print(f"\n▶ Patient {pid} | {month} | {existing} min existing\n")
    if existing >= 30:
        print("✔ Already ≥30 min billed. Done.")
        return

    orders = get_orders(pid)
    cert   = find_cert_order(orders)
    if not cert:
        print("⚠️ No 485/recert found. Aborting.")
        return

    soc = parse_date(cert["startOfCare"])
    eps = parse_date(cert["episodeStartDate"])
    epe = parse_date(cert["episodeEndDate"])
    mdt = datetime.strptime(" ".join(month.split()).title(), "%B %Y")
    month_end = datetime(mdt.year, mdt.month, calendar.monthrange(mdt.year,mdt.month)[1],23,59,59)
    window_start = soc
    window_end   = min(epe, month_end)

    agency = get_cert_info(pid).get("agencyInfo",{})
    icd    = agency.get("icdCodes",[])
    phys   = agency.get("physicianCertification","")

    existing_notes = get_ccnotes(pid)
    seen_titles    = {n.get("noteTitle","").lower() for n in existing_notes}
    seen_snips     = {
        " ".join(n.get("noteText","").split()[:10]).lower()
        for n in existing_notes
    }

    created = 0
    while existing + created*3 < 30:
        needed    = math.ceil((30 - existing - created*3)/3)
        batch     = min(3, needed)
        gen_notes = generate_notes(icd, phys, batch)
        if not gen_notes:
            print("❌ Generation failed; aborting.")
            break

        # Dedupe
        unique = []
        for title, text in gen_notes:
            kt = title.lower()
            ks = " ".join(text.split()[:10]).lower()
            if kt in seen_titles or ks in seen_snips:
                continue
            seen_titles.add(kt)
            seen_snips.add(ks)
            unique.append((title, text))

        if not unique:
            print("❌ No unique generated notes; aborting.")
            break

        # Show deduped
        print("\n===== GENERATED NOTES (deduped) =====")
        for i, (t, tx) in enumerate(unique, 1):
            print(f"[{i}] Title: {t}\n    Text:  {tx}\n")

        # Only post as many as needed
        to_post = unique[:needed]
        print(f"\n⚡ Will attempt to POST {len(to_post)} note(s) (need {needed})…")

        # Validate & (dry-run) post
        for idx, (t, tx) in enumerate(to_post, 1):
            verdict = validate_note("CPO", t, tx, icd, phys)
            print(f"\n🔎 Validating note #{idx}: “{t}” → {verdict}")
            if verdict.startswith("VALID"):
                send_dt  = random_date(window_start, window_end)
                send_str = send_dt.strftime("%m/%d/%Y")
                print(f"📬 [dry-run] Would POST note #{idx}:")
                print(f"    NoteTitle: {t}")
                print(f"    NoteText:  {tx}")
                print(f"    cpOmin:    3")
                print(f"    SendToPhysicianDate: {send_str}\n")
                created += 1
            else:
                print("⚠️ Invalid; skipping post.")

        print(f"\n⏳ {existing + created*3} min so far\n")

    total = existing + created*3
    print(f"\n🏁 Done: {created} new notes → {total} min total.")
    print("✔ Reached 30 min." if total >= 30 else "⚠️ Did NOT reach 30 min.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_cpo_with_groupname.py <patientId> <Month Year>")
        sys.exit(1)
    _, pid, mon = sys.argv
    main(pid, mon)
