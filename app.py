from dotenv import load_dotenv
load_dotenv()

"""
app.py
Mobile-first Flask app for field inspectors.
Designed to be used on a phone while on site.
"""
 
import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
 
from vision_analyzer import analyze_photo, generate_report_narrative, format_deficiency_summary
 
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
 
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
 
# ── Routes ────────────────────────────────────────────────────────────────────
 
@app.route("/")
def index():
    """Start a new inspection or resume one."""
    return render_template("index.html",
                           inspection=session.get("inspection"),
                           item_count=len(session.get("analyses", [])))
 
 
@app.route("/inspection/new", methods=["POST"])
def new_inspection():
    """Start a new inspection session."""
    session["inspection"] = {
        "id": str(uuid.uuid4())[:8].upper(),
        "building_name": request.form.get("building_name", ""),
        "building_address": request.form.get("building_address", ""),
        "inspector_name": request.form.get("inspector_name", ""),
        "inspection_date": request.form.get("inspection_date", datetime.today().strftime("%Y-%m-%d")),
        "started_at": datetime.now().isoformat()
    }
    session["analyses"] = []
    return redirect(url_for("capture"))
 
 
@app.route("/capture")
def capture():
    """Main capture screen — inspector takes or uploads a photo."""
    if not session.get("inspection"):
        return redirect(url_for("index"))
    analyses = session.get("analyses", [])
    return render_template("capture.html",
                           inspection=session["inspection"],
                           item_count=len(analyses),
                           recent=analyses[-3:] if analyses else [])
 
 
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Receive a photo from the inspector's phone, run Claude vision analysis,
    return results as JSON for the UI to display.
    """
    if "photo" not in request.files:
        return jsonify({"error": "No photo received"}), 400
 
    photo = request.files["photo"]
    location_hint = request.form.get("location", "")
 
    # Read image bytes directly — no need to save to disk first
    image_bytes = photo.read()
 
    if len(image_bytes) == 0:
        return jsonify({"error": "Empty file received"}), 400
 
    try:
        result = analyze_photo(image_bytes, location_hint)
    except json.JSONDecodeError:
        return jsonify({"error": "AI returned unexpected format. Try retaking the photo with better lighting."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
    # Save photo to disk and attach path to result
    filename = f"{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    result["_photo_path"] = filepath
    result["_captured_at"] = datetime.now().isoformat()
    result["_item_id"] = str(uuid.uuid4())[:8]
 
    # Add to inspection session
    analyses = session.get("analyses", [])
    analyses.append(result)
    session["analyses"] = analyses
    session.modified = True
 
    return jsonify({
        "ok": True,
        "item_id": result["_item_id"],
        "equipment_type": result.get("equipment_type"),
        "overall_status": result.get("overall_status"),
        "deficiency_count": len(result.get("deficiencies", [])),
        "critical_count": sum(1 for d in result.get("deficiencies", []) if d["severity"] == "CRITICAL"),
        "deficiencies": result.get("deficiencies", []),
        "equipment_details": result.get("equipment_details", {}),
        "inspection_notes": result.get("inspection_notes", ""),
        "photo_quality": result.get("photo_quality"),
        "photo_quality_notes": result.get("photo_quality_notes"),
        "nfpa_standard": result.get("nfpa_standard"),
    })
 
 
@app.route("/inspection/summary")
def summary():
    """Summary screen — all items inspected, deficiency counts, ready to generate report."""
    analyses = session.get("analyses", [])
    if not analyses:
        return redirect(url_for("capture"))
 
    total_deficiencies = sum(len(a.get("deficiencies", [])) for a in analyses)
    critical = sum(
        sum(1 for d in a.get("deficiencies", []) if d["severity"] == "CRITICAL")
        for a in analyses
    )
    major = sum(
        sum(1 for d in a.get("deficiencies", []) if d["severity"] == "MAJOR")
        for a in analyses
    )
    minor = sum(
        sum(1 for d in a.get("deficiencies", []) if d["severity"] == "MINOR")
        for a in analyses
    )
 
    overall = "FAIL" if critical > 0 else ("NEEDS_ATTENTION" if major > 0 else "PASS")
 
    return render_template("summary.html",
                           inspection=session.get("inspection", {}),
                           analyses=analyses,
                           total=total_deficiencies,
                           critical=critical,
                           major=major,
                           minor=minor,
                           overall=overall)
 
 
@app.route("/report/generate", methods=["POST"])
def generate_report():
    """Generate the full NFPA-compliant report narrative."""
    analyses = session.get("analyses", [])
    inspection = session.get("inspection", {})
 
    if not analyses:
        return jsonify({"error": "No inspections to report on"}), 400
 
    try:
        narrative = generate_report_narrative(analyses, {
            "name": inspection.get("building_name", ""),
            "address": inspection.get("building_address", ""),
            "inspector_name": inspection.get("inspector_name", ""),
            "inspection_date": inspection.get("inspection_date", ""),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
    session["report_narrative"] = narrative
    session.modified = True
 
    return jsonify({"ok": True, "narrative": narrative})
 
 
@app.route("/report/view")
def view_report():
    """View the generated report."""
    return render_template("report.html",
                           inspection=session.get("inspection", {}),
                           analyses=session.get("analyses", []),
                           narrative=session.get("report_narrative", ""))
 
 
# ── Dev helper — test with a sample image path ────────────────────────────────
 
@app.route("/dev/test-analyze")
def dev_test():
    """Quick test endpoint for development — analyze a local test image."""
    test_path = request.args.get("path", "test_extinguisher.jpg")
    location = request.args.get("location", "Test location")
 
    if not os.path.exists(test_path):
        return jsonify({"error": f"Test image not found: {test_path}. "
                                  "Pass ?path=your_image.jpg to test with a real photo."})
 
    result = analyze_photo(test_path, location)
    return jsonify(result)
 
 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
