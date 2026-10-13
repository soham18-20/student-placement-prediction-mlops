import urllib.request
import json

__test__ = False  # Mark as manual script, exclude from automatic pytest discovery

def post(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))

if __name__ == "__main__":
    print("=== 1. Health ===")
    print(get("http://127.0.0.1:8000/health"))

    print("\n=== 2. High Achiever Prediction ===")
    high = post("http://127.0.0.1:8000/api/v1/predict", {
        "cgpa": 9.2, "technical_score": 92.0, "aptitude_score": 88.0, "communication_score": 85.0,
        "internships": 2, "projects": 4, "attendance": 92.0
    })
    print("Result:", high["prediction"], "| Prob:", high["probability_percentage"], "| Conf:", high["confidence"])
    print("Key Factors:", high["key_factors"])

    print("\n=== 3. At Risk Prediction ===")
    risk = post("http://127.0.0.1:8000/api/v1/predict", {
        "cgpa": 5.8, "technical_score": 45.0, "aptitude_score": 48.0, "communication_score": 52.0,
        "internships": 0, "projects": 1, "attendance": 64.0
    })
    print("Result:", risk["prediction"], "| Prob:", risk["probability_percentage"], "| Conf:", risk["confidence"])
    print("Recommendations:", risk["recommendations"])

    print("\n=== 4. Retrain Trigger ===")
    retrain = post("http://127.0.0.1:8000/api/v1/retrain", {})
    print("Status:", retrain["message"], "| Version:", retrain["new_version"], "| Accuracy:", retrain["metrics"]["accuracy"])

    print("\n=== 5. Monitoring Metrics ===")
    metrics = get("http://127.0.0.1:8000/api/v1/monitoring/metrics")
    print("Metrics:", metrics)

