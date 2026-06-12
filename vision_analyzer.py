"""
vision_analyzer.py
 
The core AI module. Takes a photo of fire protection equipment,
sends it to Claude's vision API, and returns structured deficiency data
ready to populate an NFPA-compliant inspection report.
 
Supports:
  - Fire extinguishers    (NFPA 10)
  - Sprinkler heads       (NFPA 25)
  - Smoke/heat detectors  (NFPA 72)
  - Pull stations         (NFPA 72)
  - Exit signs            (NFPA 101)
  - Emergency lighting    (NFPA 101)
  - Fire doors            (NFPA 80)
  - Kitchen hoods         (NFPA 96)
"""
 
import base64
import json
import anthropic
from pathlib import Path
 
client = anthropic.Anthropic()
 
# ── System prompt ─────────────────────────────────────────────────────────────
# This is the secret weapon. Deep fire inspection domain knowledge
# baked into every analysis.
 
SYSTEM_PROMPT = """You are an expert fire protection inspector with 20+ years of experience 
and deep knowledge of NFPA codes. You analyze photos of fire protection equipment and 
identify deficiencies with the precision of a certified inspector.
 
For every photo you receive, you must:
 
1. IDENTIFY the equipment type (be specific — e.g. "ABC dry chemical stored pressure 
   extinguisher, appears to be 5lb Amerex B500")
 
2. READ all visible information:
   - Inspection tags/labels (dates, inspector name, service company)
   - Pressure gauge readings (PSI)
   - Manufacture/hydro test dates
   - Model numbers, ratings, certifications
   - Any visible text on labels
 
3. ASSESS condition against the relevant NFPA standard:
   - Look for: corrosion, physical damage, missing components, 
     obstructions, improper mounting, expired dates, low/no pressure,
     tamper seal condition, pin/pull tab condition, hose condition,
     bracket condition, signage/visibility
 
4. FLAG deficiencies with:
   - The exact NFPA code section violated
   - Severity: CRITICAL (immediate hazard), MAJOR (non-compliant), MINOR (advisory)
   - Clear plain-English description of the problem
   - Recommended corrective action
 
5. POPULATE standard inspection fields appropriate to the equipment type
 
Return ONLY valid JSON. No preamble, no explanation outside the JSON.
 
JSON structure:
{
  "equipment_type": string,           // e.g. "Fire Extinguisher - ABC Dry Chemical"
  "equipment_details": {
    "manufacturer": string or null,
    "model": string or null,
    "size_rating": string or null,     // e.g. "5 lb / 3A:40B:C"
    "manufacture_date": string or null,
    "last_inspection_date": string or null,
    "last_service_date": string or null,
    "next_inspection_due": string or null,
    "hydro_test_date": string or null,
    "pressure_reading": string or null, // e.g. "In the green / 195 PSI"
    "serial_number": string or null,
    "location_notes": string or null    // e.g. "wall-mounted, lobby area"
  },
  "nfpa_standard": string,             // primary standard, e.g. "NFPA 10"
  "overall_status": "PASS" | "FAIL" | "NEEDS_ATTENTION",
  "deficiencies": [
    {
      "id": number,
      "severity": "CRITICAL" | "MAJOR" | "MINOR",
      "nfpa_code": string,             // e.g. "NFPA 10 7.3.2"
      "description": string,           // plain English, what is wrong
      "corrective_action": string,     // what needs to be done
      "estimated_repair_type": string  // "replace", "repair", "recharge", "retest", "relocate", "service"
    }
  ],
  "inspection_notes": string,          // professional narrative for the report
  "photo_quality": "GOOD" | "PARTIAL" | "POOR",  // confidence in the analysis
  "photo_quality_notes": string or null // if PARTIAL/POOR, what was hard to see
}
 
NFPA code reference for common deficiencies:
- NFPA 10 7.3.2: Annual inspection requirements for extinguishers
- NFPA 10 6.2.1: Extinguisher must be fully charged
- NFPA 10 7.3.3: 6-year maintenance requirement
- NFPA 10 5.2.1: Extinguisher must be accessible and unobstructed
- NFPA 10 6.1.3.2: Tamper seal must be intact
- NFPA 25 5.2.1.1: Sprinklers free from corrosion
- NFPA 25 5.2.1.2: Sprinklers free from paint/damage
- NFPA 25 5.2.2: Spare sprinkler cabinet requirements
- NFPA 72 14.3.1: Detector annual inspection
- NFPA 72 17.7.1: Initiating devices must be unobstructed
- NFPA 101 7.10.1: Exit signs must be illuminated
- NFPA 101 7.9.3: Emergency lighting must function
- NFPA 80 5.2.1: Fire door annual inspection
- NFPA 96 11.4: Hood inspection and cleaning
 
Be conservative — if you cannot clearly see something, note it as "unable to verify 
from photo" rather than guessing. Flag items that need closer physical inspection."""
 
 
# ── Main analysis function ─────────────────────────────────────────────────────
 
def analyze_photo(image_source, location_hint: str = "") -> dict:
    """
    Analyze a photo of fire protection equipment.
 
    Args:
        image_source: Either a file path (str/Path) or raw bytes of the image
        location_hint: Optional context like "main lobby", "kitchen", "stairwell B"
 
    Returns:
        dict with equipment details, deficiencies, and inspection data
    """
    # Load and encode the image
    if isinstance(image_source, (str, Path)):
        with open(image_source, "rb") as f:
            image_bytes = f.read()
    else:
        image_bytes = image_source
 
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
 
    # Detect media type from bytes header
    media_type = _detect_media_type(image_bytes)
 
    # Build the user message
    user_text = "Analyze this fire protection equipment photo and return the JSON inspection data."
    if location_hint:
        user_text += f"\n\nLocation context: {location_hint}"
 
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_text
                    }
                ],
            }
        ],
    )
 
    raw = response.content[0].text.strip()
 
    # Strip any accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
 
    result = json.loads(raw)
 
    # Add metadata
    result["_api_input_tokens"] = response.usage.input_tokens
    result["_api_output_tokens"] = response.usage.output_tokens
    result["_location_hint"] = location_hint
 
    return result
 
 
def analyze_multiple_photos(photos: list[dict]) -> list[dict]:
    """
    Analyze multiple photos from a single inspection stop.
    Useful when an inspector takes 2-3 photos of the same item from different angles.
 
    Args:
        photos: List of dicts with keys:
                  "source" (file path or bytes)
                  "location_hint" (optional str)
 
    Returns:
        List of analysis results
    """
    results = []
    for photo in photos:
        result = analyze_photo(
            photo["source"],
            photo.get("location_hint", "")
        )
        results.append(result)
    return results
 
 
def generate_report_narrative(analyses: list[dict], building_info: dict) -> str:
    """
    Takes a list of photo analyses from a full inspection and generates
    a professional NFPA-compliant inspection report narrative.
 
    Args:
        analyses: List of results from analyze_photo()
        building_info: Dict with name, address, system_types, etc.
 
    Returns:
        Full report narrative as a string
    """
    all_deficiencies = []
    equipment_summary = []
 
    for a in analyses:
        equipment_summary.append({
            "type": a.get("equipment_type"),
            "location": a.get("_location_hint", ""),
            "status": a.get("overall_status"),
            "deficiency_count": len(a.get("deficiencies", []))
        })
        for d in a.get("deficiencies", []):
            d["equipment_type"] = a.get("equipment_type")
            d["location"] = a.get("_location_hint", "")
            all_deficiencies.append(d)
 
    critical = [d for d in all_deficiencies if d["severity"] == "CRITICAL"]
    major = [d for d in all_deficiencies if d["severity"] == "MAJOR"]
    minor = [d for d in all_deficiencies if d["severity"] == "MINOR"]
 
    prompt = f"""You are writing a professional fire inspection report for:
 
Building: {building_info.get('name', 'Unknown')}
Address: {building_info.get('address', 'Unknown')}
Inspector: {building_info.get('inspector_name', 'Unknown')}
Inspection Date: {building_info.get('inspection_date', 'Unknown')}
 
Equipment inspected: {len(analyses)} items
Deficiencies found: {len(critical)} critical, {len(major)} major, {len(minor)} minor
 
Deficiency details:
{json.dumps(all_deficiencies, indent=2)}
 
Write a professional NFPA-compliant inspection report narrative that:
1. Opens with a brief executive summary
2. Lists all deficiencies grouped by severity (critical first)
3. References specific NFPA code sections
4. Uses professional, precise language appropriate for AHJ submission
5. Ends with a compliance determination and recommended timeline for corrections
 
Format as plain text with clear section headers."""
 
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )
 
    return response.content[0].text
 
 
# ── Helper functions ──────────────────────────────────────────────────────────
 
def _detect_media_type(image_bytes: bytes) -> str:
    """Detect image format from file header bytes."""
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    elif image_bytes[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    elif image_bytes[:3] == b'GIF':
        return "image/gif"
    else:
        return "image/jpeg"  # safe default
 
 
def severity_emoji(severity: str) -> str:
    return {"CRITICAL": "🔴", "MAJOR": "🟠", "MINOR": "🟡"}.get(severity, "⚪")
 
 
def format_deficiency_summary(analysis: dict) -> str:
    """Quick human-readable summary for logging/debugging."""
    lines = [
        f"Equipment: {analysis.get('equipment_type', 'Unknown')}",
        f"Status: {analysis.get('overall_status', 'Unknown')}",
        f"NFPA Standard: {analysis.get('nfpa_standard', 'Unknown')}",
        f"Deficiencies: {len(analysis.get('deficiencies', []))}",
    ]
    for d in analysis.get("deficiencies", []):
        emoji = severity_emoji(d["severity"])
        lines.append(f"  {emoji} [{d['nfpa_code']}] {d['description']}")
    return "\n".join(lines)
