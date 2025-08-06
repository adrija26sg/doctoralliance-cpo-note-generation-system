#!/usr/bin/env python3
import os
import sys
import math
import requests
import calendar
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI, APITimeoutError, OpenAIError

# ─── Load environment variables ──────────────────────────────────────────────────
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

# ─── INIT AZURE OPENAI CLIENT ──────────────────────────────────────────────────
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version=AZURE_OPENAI_VERSION,
)

def _auth_hdrs():
    return {"Authorization": f"Bearer {DA_API_KEY}"} if DA_API_KEY else {}

def parse_cert_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%m/%d/%Y")

def get_existing_cpo_minutes(patient_id: str, month_label: str) -> int:
    m_label = " ".join(month_label.split()).title()
    m_dt    = datetime.strptime(m_label, "%B %Y")
    first   = datetime(m_dt.year, m_dt.month, 1)
    last_day= calendar.monthrange(m_dt.year, m_dt.month)[1]
    last    = datetime(m_dt.year, m_dt.month, last_day, 23, 59, 59)

    resp = requests.get(f"{BASE_URL}/CCNotes/patient/{patient_id}", headers=_auth_hdrs())
    resp.raise_for_status()
    total = 0

    for note in resp.json() or []:
        ts_str = note.get("updatedOn") or note.get("createdAt")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.rstrip("Z"))
        except ValueError:
            continue
        if first <= ts <= last:
            total += int(note.get("cpOmin") or 0)

    return total

def get_orders(patient_id: str) -> list[dict]:
    resp = requests.get(f"{BASE_URL}/Order/patient/{patient_id}", headers=_auth_hdrs())
    resp.raise_for_status()
    return resp.json() or []

def get_cert_info(patient_id: str) -> dict:
    resp = requests.get(f"{BASE_URL}/Patient/total/{patient_id}", headers=_auth_hdrs())
    resp.raise_for_status()
    return resp.json() or {}

def find_485_cert_order(orders: list[dict]) -> dict | None:
    for o in orders:
        da = (o.get("daOrderType") or "").lower()
        dn = (o.get("documentName") or "").lower()
        if da.startswith("485") or "485" in dn:
            return o
    return None

# ─── Prompt builder  ───────────────────────────────────────────────────────────
def build_prompt(icd_codes: list[str], physician_text: str, count: int) -> str:
    return (
        f"Patient has the following diagnoses (ICD-10 codes not to be mentioned directly): "
        f"{', '.join(icd_codes[:5])}\n"
        f"Physician Certification Statement:\n{physician_text}\n\n"
        f"Generate {count} distinct, human-sounding care-coordination notes "
        "(each ~3 minutes), referencing the patient’s conditions and physician statement, "
        "but do NOT print any numeric ICD-10 codes."
    )

def generate_notes(prompt: str) -> list[str]:
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a skilled nurse documentation assistant."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600,
                timeout=60
            )
            text = resp.choices[0].message.content.strip()
            return [n.strip() for n in text.split("\n\n") if n.strip()]
        except APITimeoutError:
            if attempt == 0:
                print("⚠️  OpenAI request timed out, retrying once…")
                continue
            print("❌ OpenAI timed out again; aborting this batch.")
            return []
        except OpenAIError as e:
            print(f"❌ OpenAI API error: {e}")
            return []

    return []

def ensure_thirty_minutes_cpo(patient_id: str, month_label: str):
    existing     = get_existing_cpo_minutes(patient_id, month_label)
    print(f"Existing CPO minutes in {month_label}: {existing}")
    if existing >= 30:
        print("✔ Already at or above 30 minutes. Billing complete.")
        return

    orders       = get_orders(patient_id)
    cert_order   = find_485_cert_order(orders)
    if not cert_order:
        print(f"⚠️  No 485 cert order found for patient {patient_id}.")
        return

    soc_date     = parse_cert_date(cert_order["startOfCare"])
    ep_start     = parse_cert_date(cert_order["episodeStartDate"])
    ep_end       = parse_cert_date(cert_order["episodeEndDate"])

    agency        = get_cert_info(patient_id).get("agencyInfo", {})
    icd_list      = agency.get("icdCodes", [])
    physician_txt = agency.get("physicianCertification", "")

    notes_needed  = math.ceil((30 - existing) / 3)
    print(f"Will generate {notes_needed} note(s)…")

    for idx in range(0, notes_needed, 3):
        batch_count = min(3, notes_needed - idx)
        prompt      = build_prompt(icd_list, physician_txt, batch_count)
        new_notes   = generate_notes(prompt)

        for note in new_notes:
            print("── Generated care-coordination note (+3 min) ──")
            print(note)
            print("──────────────────────────────────────────────\n")

    total_mins = notes_needed * 3
    print(f"✔ Completed generation of {total_mins} minutes worth of notes for {month_label}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_cpo.py <patientId> <monthLabel>")
        sys.exit(1)

    ensure_thirty_minutes_cpo(sys.argv[1], sys.argv[2])


# openai.api_key = OPENAI_API_KEY

# def _auth_hdrs():
#     return {"Authorization": f"Bearer {DA_API_KEY}"} if DA_API_KEY else {}

# # ─── Helpers ────────────────────────────────────────────────────────────────────

# def get_existing_cpo_minutes(patient_id: str) -> int:
#     resp = requests.get(f"{BASE_URL}/CCNotes/patient/{patient_id}", headers=_auth_hdrs())
#     resp.raise_for_status()
#     notes = resp.json() or []
#     return sum(int(n.get("cpOmin", 0)) for n in notes)

# def get_orders(patient_id: str) -> list[dict]:
#     resp = requests.get(f"{BASE_URL}/Order/patient/{patient_id}", headers=_auth_hdrs())
#     resp.raise_for_status()
#     return resp.json() or []

# def get_cert_info(patient_id: str) -> dict:
#     resp = requests.get(f"{BASE_URL}/Patient/total/{patient_id}", headers=_auth_hdrs())
#     resp.raise_for_status()
#     return resp.json() or {}

# def find_485_cert_order(orders: list[dict]) -> dict | None:
#     """
#     Try to locate the 485 cert by checking `daOrderType` or `documentName`.
#     Returns the order dict if found, or None otherwise.
#     """
#     for o in orders:
#         # 1) check daOrderType field
#         if o.get("daOrderType", "").lower().startswith("485"):
#             return o
#         # 2) fallback to documentName
#         if "485" in o.get("documentName", "").lower():
#             return o
#     return None

# def build_prompt(icd_codes: list[str], physician_text: str, count: int) -> str:
#     return (
#         f"ICD-10 Codes (first 5): {', '.join(icd_codes[:5])}\n"
#         f"Physician Certification Statement:\n{physician_text}\n\n"
#         f"Generate {count} distinct, human-sounding care-coordination notes "
#         "(each ~3 minutes), referencing only the above codes and statement."
#     )

# def generate_notes(prompt: str) -> list[str]:
#     resp = openai.ChatCompletion.create(
#         model=OPENAI_MODEL,
#         messages=[
#             {"role": "system", "content": "You are a skilled nurse documentation assistant."},
#             {"role": "user",   "content": prompt}
#         ],
#         temperature=0.7,
#         max_tokens=600
#     )
#     text = resp.choices[0].message.content.strip()
#     return [n.strip() for n in text.split("\n\n") if n.strip()]

# def post_note(patient_id: str, text: str):
#     payload = {"noteText": text, "cpOmin": 3}
#     resp = requests.post(
#         f"{BASE_URL}/CCNotes/patient/{patient_id}",
#         json=payload,
#         headers={**_auth_hdrs(), "Content-Type":"application/json"}
#     )
#     resp.raise_for_status()
#     return resp.json()

# # ─── Orchestration with New Flow ───────────────────────────────────────────────

# def ensure_thirty_minutes_cpo(patient_id: str, month_label: str):
#     # 1) Check existing CPO minutes
#     existing = get_existing_cpo_minutes(patient_id)
#     print(f"Existing CPO minutes: {existing}")
#     if existing >= 30:
#         print("✔ Already at or above 30 minutes. Billing complete.")
#         return

#     # 2) Fetch orders & locate the 485 cert ( skip if none)
#     orders     = get_orders(patient_id)
#     cert_order = find_485_cert_order(orders)
#     if cert_order is None:
#         print(f"⚠️  No 485 certification order found for patient {patient_id}.")
#         print("    Cannot generate CPO notes without a valid cert.")
#         return

#     # 3) Parse the cert’s start/end dates
#     soc_date = datetime.fromisoformat(cert_order["startOfCare"])
#     ep_start = datetime.fromisoformat(cert_order["episodeStartDate"])
#     ep_end   = datetime.fromisoformat(cert_order["episodeEndDate"])

#     # 4) Compute first/last instants of the target month
#     m_dt     = datetime.strptime(month_label.title(), "%B %Y")  # e.g. "June 2025"
#     year, mon= m_dt.year, m_dt.month
#     first    = datetime(year, mon, 1)
#     last_day = calendar.monthrange(year, mon)[1]
#     last     = datetime(year, mon, last_day, 23, 59, 59)

#     # 5) Ensure the cert covers the entire target month
#     if not (ep_start <= first and ep_end >= last):
#         print(f"⚠️  Cert period {ep_start.date()}–{ep_end.date()} does not span {month_label.title()}.")
#         return

#     # 6) Pull ICD codes & physician text from cert-info
#     agency        = get_cert_info(patient_id).get("agencyInfo", {})
#     icd_list      = agency.get("icdCodes", [])
#     physician_txt = agency.get("physicianCertification", "")

#     # 7) Calculate remaining notes to hit 30 minutes
#     to_go         = 30 - existing
#     notes_needed  = math.ceil(to_go / 3)
#     print(f"Will generate {notes_needed} note(s) to reach 30 minutes…")

#     # 8) Build prompt, generate via OpenAI, and post each note
#     prompt    = build_prompt(icd_list, physician_txt, notes_needed)
#     new_notes = generate_notes(prompt)
#     for note in new_notes:
#         posted = post_note(patient_id, note)
#         print(f"  • Posted note ID {posted.get('id')} (+3 min)")

#     print("✔ Completed 30 minutes of CPO.")

# # ─── CLI Entry ─────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 3:
#         print("Usage: python cpo_workflow.py <patientId> <monthLabel>")
#         print("Example: python cpo_workflow.py 07ba7a72-f6e9-451c-afa5-569286705c62 \"june 2025\"")
#         sys.exit(1)

#     pid, month = sys.argv[1], sys.argv[2]
#     ensure_thirty_minutes_cpo(pid, month)