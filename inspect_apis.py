import requests

BASE_URL   = "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api"
patient_id = "9694a0c4-2894-46c7-9e2d-6267ec0cbe11"

endpoints = {
    "Certification Info": f"/Patient/total/{patient_id}",
    "Orders List":        f"/Order/patient/{patient_id}",
    "CCNotes List":       f"/CCNotes/patient/{patient_id}"
}

def fetch_fields(name, path):
    url = BASE_URL + path
    print(f"\n=== {name} ===")
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    sample = data[0] if isinstance(data, list) and data else data
    if not sample:
        print("  (no data returned)")
        return
    print("  Fields:")
    for key in sample.keys():
        print(f"   - {key}")

if __name__ == "__main__":
    for name, path in endpoints.items():
        fetch_fields(name, path)
