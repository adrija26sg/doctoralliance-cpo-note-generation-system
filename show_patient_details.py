#!/usr/bin/env python3
import os
import sys
import requests
import json

BASE_URL   = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api"
DA_API_KEY = os.getenv("etWxbbDCpGumaFi4JjESeb716HLWhKmXBfH9N0ACVz_X0MzFoDPssSo_zJspbUmPcrfPK85OMm1bO6bJWfLbNgPAl_zGC2KsQPCb2Kjo_JZEbU_0TfhF-BAFc0nfj-pGSq6teW-YaDga_P5UJrANPO-GPvqX64lJKtDeQcnJOZs1hlR6-jJsKWhHMsCCXmZbj_klAUKtIzQeRrLXtGF03WEBTrmUdiguZeMkYKxWHQz73muJcFXnAqX4lJZyFFoVBFHMcvVF9Wh-McsYwwqNYQ")

HEADERS = {
    "Accept": "application/json",
    **({"Authorization": f"Bearer {DA_API_KEY}"} if DA_API_KEY else {})
}

def fetch_orders(patient_id: str):
    url = f"{BASE_URL}/Order/patient/{patient_id}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json() or []

def list_documents(patient_id: str):
    orders = fetch_orders(patient_id)
    if not orders:
        print(f"No orders found for patient {patient_id}")
        return

    # Group by documentName
    docs = {}
    for o in orders:
        name = o.get("documentName") or "(untitled)"
        docs.setdefault(name, []).append(o)

    print(f"\nFound {len(docs)} document type(s) for patient {patient_id}:\n")
    for doc_name, entries in docs.items():
        sample = entries[0]
        print(f"▶ Document Name: {doc_name!r}  ( {len(entries)} entr{'y' if len(entries)==1 else 'ies'} )")
        print("  Fields in a sample record:")
        for key in sample.keys():
            print(f"   - {key}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_patient_details.py <patientId>")
        sys.exit(1)
    pid = sys.argv[1]
    list_documents(pid)
