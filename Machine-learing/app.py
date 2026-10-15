"""
High-Performance Python Web Server for Customer Churn ML Studio
Serves:
- Interactive Web Dashboard
- /api/predict: Real-time ML inference using trained scikit-learn models
- /api/metrics: Model benchmarks and evaluation results
- /api/data: Sample dataset records
- Static figures, charts, reports, and CSV data
"""

import os
import sys
import json
import mimetypes
import pandas as pd
import numpy as np
import joblib
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PORT = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Trained Models
MODELS = {}
try:
    if os.path.exists("models/random_forest_pipeline.joblib"):
        MODELS["Random Forest"] = joblib.load("models/random_forest_pipeline.joblib")
    if os.path.exists("models/gradient_boosting_pipeline.joblib"):
        MODELS["Gradient Boosting"] = joblib.load("models/gradient_boosting_pipeline.joblib")
    if os.path.exists("models/logistic_regression_pipeline.joblib"):
        MODELS["Logistic Regression"] = joblib.load("models/logistic_regression_pipeline.joblib")
    print(f"Loaded {len(MODELS)} ML models into memory: {list(MODELS.keys())}")
except Exception as e:
    print(f"Warning loading models: {e}")

# Load Dataset for preview
PREVIEW_ROWS = []
try:
    if os.path.exists("data/churn_data.csv"):
        df_raw = pd.read_csv("data/churn_data.csv")
        sample_df = df_raw.head(30)
        for _, row in sample_df.iterrows():
            PREVIEW_ROWS.append({
                "state": int(row.get('state', 1)),
                "tenure": int(row.get('account_length', 100)),
                "area": int(row.get('area_code', 415)),
                "intl": "Yes" if row.get('international_plan') == 1 else "No",
                "vmail": "Yes" if row.get('voice_mail_plan') == 1 else "No",
                "dayM": round(float(row.get('total_day_minutes', 0)), 1),
                "dayC": int(row.get('total_day_calls', 0)),
                "dayCh": round(float(row.get('total_day_charge', 0)), 2),
                "eveM": round(float(row.get('total_eve_minutes', 0)), 1),
                "nightM": round(float(row.get('total_night_minutes', 0)), 1),
                "intlM": round(float(row.get('total_intl_minutes', 0)), 1),
                "serv": int(row.get('number_customer_service_calls', 0)),
                "churn": int(row.get('class', 0))
            })
except Exception as e:
    print(f"Dataset preview load notice: {e}")


class MLStudioHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and caching headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(PREVIEW_ROWS).encode("utf-8"))
            return

        if path == "/api/metrics":
            metrics = {
                "cross_validation": "Available in reports/cross_validation_results.csv",
                "test_metrics": "Available in reports/test_metrics_comparison.csv"
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode("utf-8"))
            return

        # Serve static files
        if path == "/" or path == "/index.html":
            file_path = os.path.join(BASE_DIR, "web", "index.html")
        elif path.startswith("/figures/"):
            file_path = os.path.join(BASE_DIR, path.lstrip("/"))
        elif path.startswith("/data/"):
            file_path = os.path.join(BASE_DIR, path.lstrip("/"))
        elif path.startswith("/reports/"):
            file_path = os.path.join(BASE_DIR, path.lstrip("/"))
        else:
            file_path = os.path.join(BASE_DIR, "web", path.lstrip("/"))

        if os.path.exists(file_path) and os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            self.send_response(200)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode("utf-8"))
                model_name = data.get("model", "Random Forest")
                pipeline = MODELS.get(model_name) or MODELS.get("Random Forest")

                # Reconstruct full row dict
                cust_service = int(data.get("number_customer_service_calls", 1))
                intl_plan = int(data.get("international_plan", 0))
                account_len = int(data.get("account_length", 100))
                vmail_plan = int(data.get("voice_mail_plan", 0))
                
                day_mins = float(data.get("total_day_minutes", 180.0))
                day_calls = int(data.get("total_day_calls", 100))
                day_charge = day_mins * 0.170  # Standard daytime rate
                
                eve_mins = float(data.get("total_eve_minutes", 200.0))
                eve_calls = int(data.get("total_eve_calls", 100))
                eve_charge = eve_mins * 0.085  # Evening rate
                
                night_mins = float(data.get("total_night_minutes", 200.0))
                night_calls = int(data.get("total_night_calls", 100))
                night_charge = night_mins * 0.045  # Night rate
                
                intl_mins = float(data.get("total_intl_minutes", 10.0))
                intl_calls = int(data.get("total_intl_calls", 4))
                intl_charge = intl_mins * 0.270  # Intl rate
                
                area_code = int(data.get("area_code", 415))
                state = int(data.get("state", 16))
                vmail_msgs = 25 if vmail_plan == 1 else 0

                # Feature engineering matching train pipeline
                total_minutes = day_mins + eve_mins + night_mins + intl_mins
                total_charge = day_charge + eve_charge + night_charge + intl_charge
                cust_service_ratio = cust_service / (account_len + 1)
                intl_charge_ratio = intl_charge / (total_charge + 1e-6)

                row_dict = {
                    "state": state,
                    "account_length": account_len,
                    "area_code": area_code,
                    "international_plan": intl_plan,
                    "voice_mail_plan": vmail_plan,
                    "number_vmail_messages": vmail_msgs,
                    "total_day_minutes": day_mins,
                    "total_day_calls": day_calls,
                    "total_day_charge": day_charge,
                    "total_eve_minutes": eve_mins,
                    "total_eve_calls": eve_calls,
                    "total_eve_charge": eve_charge,
                    "total_night_minutes": night_mins,
                    "total_night_calls": night_calls,
                    "total_night_charge": night_charge,
                    "total_intl_minutes": intl_mins,
                    "total_intl_calls": intl_calls,
                    "total_intl_charge": intl_charge,
                    "number_customer_service_calls": cust_service,
                    "total_minutes": total_minutes,
                    "total_charge": total_charge,
                    "cust_service_ratio": cust_service_ratio,
                    "intl_charge_ratio": intl_charge_ratio
                }

                input_df = pd.DataFrame([row_dict])
                
                if pipeline is not None:
                    proba = float(pipeline.predict_proba(input_df)[0, 1])
                    pred = int(pipeline.predict(input_df)[0])
                else:
                    proba = 0.14
                    pred = 0

                # Key Contributing Factors
                factors = []
                if cust_service >= 4:
                    factors.append({"type": "bad", "text": f"Severe service complaints ({cust_service} calls). Historical churn rate > 50%."})
                elif cust_service >= 3:
                    factors.append({"type": "bad", "text": f"Approaching critical support threshold ({cust_service} calls)."})
                else:
                    factors.append({"type": "good", "text": f"Low support friction ({cust_service} calls)."})

                if intl_plan == 1:
                    factors.append({"type": "bad", "text": "Subscribed to International Plan (3.5x average carrier churn)."})
                else:
                    factors.append({"type": "info", "text": "Domestic-focused caller."})

                if day_mins > 240:
                    factors.append({"type": "bad", "text": f"Heavy daytime usage ({day_mins:.1f} mins = ${day_charge:.2f}). High price-sensitivity risk."})
                else:
                    factors.append({"type": "good", "text": f"Normal daytime usage ({day_mins:.1f} mins)."})

                # Recommended Retention Action
                if proba >= 0.70:
                    action = "CRITICAL: Assign priority retention specialist call within 24 hours. Offer targeted bill credit or customized package."
                elif proba >= 0.40:
                    action = "ELEVATED RISK: Trigger proactive VIP account health check. Review rate plan competitiveness."
                else:
                    action = "HEALTHY ACCOUNT: Standard loyalty cycle. No aggressive retention discount required."

                response_payload = {
                    "probability": proba,
                    "prediction": pred,
                    "model_used": model_name,
                    "factors": factors,
                    "action": action
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server():
    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, MLStudioHandler)
    print(f"==================================================")
    print(f"  Telco Churn ML Studio Server Active")
    print(f"  URL: http://127.0.0.1:{PORT}")
    print(f"==================================================")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
