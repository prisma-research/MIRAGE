"""
Build the modality-extension data across THREE genuine modalities (not just image
content-types): 6 new evidence objects =
  - 2 image (vision): photo, infographic            → planted as image
  - 2 text          : code (.py), email/note        → content planted as text
  - 2 pdf           : invoice, quarterly report      → real PDF files, ingested via
                      our pdf2text tool (pipeline/pdf2text.py); extracted text planted

The PDFs are genuine text-layer PDFs (matplotlib) and are run through pdf2text so the
"PDF handled by a pdf2text tool" claim is real and reproducible. Backbones' chat APIs
take text+image only (not PDF), so PDF artifacts are preprocessed to text — exactly how
a real personal agent ingests an attached PDF.

Outputs:
  data/generated_images/<m>/<aid>.png         (image objects)
  data/generated_pdfs/<aid>.pdf               (pdf objects, real files)
  configs/study/modality_ext_objects.json     (artifact defs incl. modality + content)
  configs/study/modality_ext_bank.json        (108 QA)
  configs/study/combined_modality_bank.json   (original 200 + 108)
  configs/episodes/modality_ext_100k.json      (combined 12-object trunk, native)
  configs/episodes/modality_ext_100k_plant.json(plant-only, truncated at d0)

Usage: python -m pipeline.build_modality_ext_data
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.pdf2text import pdf_to_text

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "generated_images"
PDFS = ROOT / "data" / "generated_pdfs"
FILES = ROOT / "data" / "generated_files"   # text artifacts as named workspace files
STAGE = GEN / "modality_staging"
STUDY = ROOT / "configs" / "study"
EPI = ROOT / "configs" / "episodes"

PROTOCOL = ("\n\nReturn exactly:\n"
            "ANSWERABLE=YES | NO\nSOURCE=<artifact_id or NONE>\nANSWER=<short value or NONE>")

CODE_TEXT = '''# service_client.py — configuration and user fetch helper
import time

DEFAULT_TIMEOUT = 30          # seconds
MAX_RETRIES = 5
API_BASE_URL = "https://api.example.com/v2"
CACHE_TTL = 3600              # seconds
RETRY_BACKOFF = 2.0
ENABLE_CACHE = True
LOG_LEVEL = "INFO"


def fetch_user(user_id: int, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Fetch a user record by id. Returns {} on failure."""
    for attempt in range(MAX_RETRIES):
        resp = http_get(f"{API_BASE_URL}/users/{user_id}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(RETRY_BACKOFF ** attempt)
    return {}'''

EMAIL_TEXT = '''From: Sarah Chen <s.chen@northwind.io>
To: me
Subject: Q2 offsite logistics — action needed
Date: 2024-05-14

Hi,
Confirming the Q2 team offsite: it is booked for June 18-19 at the Lakeside Conference Center.
We have 24 attendees. The venue deposit of 3200 dollars is due by May 30.
Please send your dietary restrictions to Marco by Friday. Book flights under project code
NW-Q2-OFFSITE. The hotel block code is LAKE2024.
Thanks,
Sarah'''

INVOICE_LINES = [
    "INVOICE  #INV-2024-0072",
    "Vendor: Acme Analytics Ltd",
    "Bill to: Northwind Data Team",
    "Invoice date: 2024-03-09",
    "PO: PO-5519",
    "Payment terms: Net 30",
    "",
    "Subtotal: 42000",
    "Tax: 5000",
    "Total due: 47000 USD",
    "Due date: 2024-04-08",
]
REPORT_LINES = [
    "QUARTERLY PERFORMANCE REPORT  Q1 2024",
    "Prepared by: Analytics Team",
    "",
    "Regional revenue (USD thousands):",
    "Region      Q1-2024   Q4-2023",
    "North        1240      1100",
    "South         860       910",
    "West         1530      1420",
    "East          720       680",
    "",
    "Key findings:",
    "- West region led with 1530, up from 1420.",
    "- South was the only region to decline quarter-over-quarter.",
    "- Total Q1 revenue: 4350.",
    "Next review: 2024-05-15.",
]


def make_pdf(lines, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.95, "\n".join(lines), va="top", ha="left",
             family="monospace", fontsize=12)
    fig.savefig(out); plt.close(fig)


# modality ∈ {image, text, pdf}
OBJECTS = [
    dict(aid="pho_01", modality="image", mdir="photo", src=STAGE/"photo"/"photo_004.png",
         cg="photo_natural", vt="natural_photo",
         plant="Here's a food photo I took — a bowl of noodle soup. I might ask about it later.",
         kw=["noodle","soup","shrimp","carrot","foodiebaker"]),
    dict(aid="mix_01", modality="image", mdir="infographic", src=STAGE/"infographic"/"infographic_000.png",
         cg="infographic", vt="infographic",
         plant="A social-media cheat-sheet infographic. Keep the key facts in mind.",
         kw=["social","media","platform","pinterest","linkedin","constant","contact"]),
    dict(aid="code_01", modality="text", content=CODE_TEXT,
         cg="code", vt="code_text",
         frame="Here's a Python config module (service_client.py) I'm working with — note the values and behaviour:",
         kw=["default_timeout","max_retries","api_base_url","fetch_user","cache_ttl"]),
    dict(aid="email_01", modality="text", content=EMAIL_TEXT,
         cg="email", vt="email_text",
         frame="Here's an email I received about our Q2 offsite — keep the details, I'll refer back:",
         kw=["sarah","offsite","lakeside","deposit","nw-q2-offsite","lake2024"]),
    dict(aid="pdf_01", modality="pdf", pdf_lines=INVOICE_LINES,
         cg="pdf_invoice", vt="pdf_text",
         frame="I'm sharing an invoice PDF. Our pdf2text tool extracted the following content — keep the figures:",
         kw=["invoice","inv-2024-0072","acme","47000","po-5519"]),
    dict(aid="pdf_02", modality="pdf", pdf_lines=REPORT_LINES,
         cg="pdf_report", vt="pdf_text",
         frame="I'm sharing a quarterly report PDF. Our pdf2text tool extracted the following content — note the table:",
         kw=["quarterly","revenue","west","1530","4350","region"]),
]

QA = {
"pho_01": dict(A=[
    ("What type of dish is shown in the photo?","noodle soup","string"),
    ("What seafood is visible in the bowl?","shrimp","string"),
    ("What orange vegetable is in the soup?","carrot","string"),
    ("What eating utensils are shown beside the bowl?","chopsticks","string"),
    ("What website is watermarked on the photo?","foodiebaker.com","string"),
    ("What material is the mat under the bowl?","bamboo","string"),
    ("Is there egg in the soup?","yes","string"),
    ("What is the main starch food in the bowl?","noodles","string"),
    ("What is the white protein floating in the broth?","egg","string"),
], U=[
    "What is the price of this dish?","What restaurant served this dish?",
    "How many calories are in the soup?","What city was this photo taken in?",
    "What is the name of the chef?","What brand are the chopsticks?",
    "What temperature is the soup served at?","What spice was added to the broth?",
    "What camera model took this photo?",
]),
"mix_01": dict(A=[
    ("What is the title of the infographic?","Social Media Platform Cheat Sheet","string"),
    ("Which platform is described as having a heavily female audience?","Pinterest","string"),
    ("Which platform is the profession-driven, B2B platform?","LinkedIn","string"),
    ("What resource and expertise level does Facebook need?","low","string"),
    ("What resource and expertise level does LinkedIn need?","medium","string"),
    ("How many social media platforms are covered in the cheat sheet?","5","numeric"),
    ("Which platform is described as fast-moving and real-time?","Twitter","string"),
    ("Which company produced this cheat sheet (shown in the footer)?","Constant Contact","string"),
    ("Which platform is listed as good for bakeries and coffee shops?","Instagram","string"),
], U=[
    "Which platform has the most monthly active users?","What is the cost of advertising on Facebook?",
    "Which platform was founded first?","What is YouTube best used for?",
    "What resource level does TikTok need?","In what year was this cheat sheet published?",
    "What is Snapchat good for?","Which platform has the highest return on investment?",
    "How many followers does the average Pinterest account have?",
]),
"code_01": dict(A=[
    ("What value is DEFAULT_TIMEOUT set to in the code?","30","numeric"),
    ("What value is MAX_RETRIES set to?","5","numeric"),
    ("What is the value of CACHE_TTL?","3600","numeric"),
    ("What is the value of RETRY_BACKOFF?","2.0","numeric"),
    ("What is the API_BASE_URL in the code?","https://api.example.com/v2","string"),
    ("What is LOG_LEVEL set to?","INFO","string"),
    ("Is ENABLE_CACHE True or False?","True","string"),
    ("What does the function fetch_user return on failure?","{}","string"),
    ("What HTTP status code does fetch_user treat as success?","200","numeric"),
    ("What is the name of the function defined in the code?","fetch_user","string"),
], U=[
    "What value is MIN_RETRIES set to?","What is the DATABASE_URL in the code?",
    "What port does the service listen on?","What is the value of MAX_TIMEOUT?",
    "What is the default page size?","What authentication token does the code use?",
    "What is the value of RATE_LIMIT?","What logging file path is configured?",
    "What is the value of CONNECTION_POOL_SIZE?",
]),
"email_01": dict(A=[
    ("Who sent the email?","Sarah Chen","string"),
    ("At what venue is the Q2 offsite booked?","Lakeside Conference Center","string"),
    ("How many attendees are there?","24","numeric"),
    ("What is the venue deposit amount in dollars?","3200","numeric"),
    ("By what date is the deposit due?","May 30","string"),
    ("Who should dietary restrictions be sent to?","Marco","string"),
    ("What is the project code for booking flights?","NW-Q2-OFFSITE","string"),
    ("What is the hotel block code?","LAKE2024","string"),
    ("In what month is the offsite held?","June","string"),
], U=[
    "What is the offsite agenda?","What is the flight cost?","What is the name of the hotel?",
    "What is Sarah's phone number?","What are the names of the 24 attendees?",
    "What is the total offsite budget?","Is the CEO attending?",
    "What is the nightly room rate?","What is the cancellation policy?",
]),
"pdf_01": dict(A=[
    ("What is the invoice number?","INV-2024-0072","string"),
    ("Who is the vendor on the invoice?","Acme Analytics Ltd","string"),
    ("What is the total amount due in USD?","47000","numeric"),
    ("What is the subtotal?","42000","numeric"),
    ("What is the tax amount?","5000","numeric"),
    ("What is the invoice due date?","2024-04-08","string"),
    ("What is the PO number?","PO-5519","string"),
    ("What are the payment terms?","Net 30","string"),
    ("What is the invoice date?","2024-03-09","string"),
], U=[
    "What discount was applied to the invoice?","What is the late payment fee?",
    "What bank account should payment be sent to?","What is the vendor's contact email?",
    "How many line items are on the invoice?","What is the customer's tax ID?",
    "What currency exchange rate was used?","Who approved the invoice?",
    "What is the shipping cost?",
]),
"pdf_02": dict(A=[
    ("Which region had the highest Q1 2024 revenue?","West","string"),
    ("What was West region's Q1 2024 revenue (in thousands)?","1530","numeric"),
    ("What was North region's Q1 2024 revenue, in thousands as shown in the table?","1240","numeric"),
    ("What was the total Q1 2024 revenue in thousands?","4350","numeric"),
    ("Which region declined quarter-over-quarter?","South","string"),
    ("What is the next review date?","2024-05-15","string"),
    ("What was South region's Q1 2024 revenue, in thousands as shown in the table?","860","numeric"),
    ("What was East region's Q1 2024 revenue, in thousands as shown in the table?","720","numeric"),
    ("What quarter does the report cover?","Q1 2024","string"),
], U=[
    "What is the Q2 2024 forecast?","What is the profit margin?",
    "How many employees are in each region?","Which region has the most customers?",
    "What did the CEO say about the results?","What is the annual revenue total?",
    "What was West region's year-over-year growth percent?",
    "What is the cost breakdown by region?","What is the marketing spend?",
]),
}


def main():
    objects_out, bank_q = [], []
    for o in OBJECTS:
        aid = o["aid"]; modality = o["modality"]
        entry = {"artifact_id": aid, "modality": modality, "plant_type": o["vt"],
                 "confusion_group": o["cg"], "visual_type": o["vt"], "ocr_keywords": o["kw"]}

        if modality == "image":
            dst = GEN / o["mdir"] / f"{aid}.png"; dst.parent.mkdir(parents=True, exist_ok=True)
            if o["src"].exists() and not dst.exists():
                shutil.copy(o["src"], dst)
            entry["image_path"] = str(dst.relative_to(ROOT))
            entry["plant_prompt"] = o["plant"]
            entry["plant_is_image"] = True
        else:
            # text/pdf are planted as NAMED workspace files so the model cites the
            # filename (→ artifact_id) as SOURCE, consistent with image objects, and
            # the file persists post-compaction for tool-mediated retrieval.
            if modality == "pdf":
                dst_file = PDFS / f"{aid}.pdf"
                make_pdf(o["pdf_lines"], dst_file)
                content = pdf_to_text(dst_file)          # genuine pdf2text ingestion
                fname = f"{aid}.pdf"
            else:
                ext = "py" if o["cg"] == "code" else "txt"
                fname = f"{aid}.{ext}"
                dst_file = FILES / fname
                FILES.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(o["content"])
                content = o["content"]
            entry["content"] = content
            entry["image_path"] = None
            entry["file_path"] = str(dst_file.relative_to(ROOT))
            entry["filename"] = fname
            entry["plant_is_image"] = False
            entry["plant_prompt"] = (
                f"I've saved a file to your workspace: `{fname}`.\n{o['frame']}\n\n"
                f"{content}\n\nKeep `{fname}` in mind — I'll ask about it later.")

        objects_out.append(entry)

        qa = QA[aid]
        for i, (q, a, at) in enumerate(qa["A"], 1):
            bank_q.append(dict(base_question_id=f"{aid}_a{i:02d}", natural_question=q,
                               prompt=q + PROTOCOL, gold_answerable="YES", gold_source=aid,
                               gold_answer=a, answer_type=at, confusion_group=o["cg"],
                               visual_type=o["vt"], modality=modality,
                               question_scope="source_explicit", modality_ext=True))
        for i, q in enumerate(qa["U"], 1):
            bank_q.append(dict(base_question_id=f"{aid}_u{i:02d}", natural_question=q,
                               prompt=q + PROTOCOL, gold_answerable="NO", gold_source="NONE",
                               gold_answer="NONE", answer_type="string", confusion_group=o["cg"],
                               visual_type=o["vt"], modality=modality,
                               question_scope="source_explicit", modality_ext=True))

    STUDY.mkdir(parents=True, exist_ok=True)
    (STUDY / "modality_ext_objects.json").write_text(json.dumps(objects_out, indent=2, ensure_ascii=False))
    n_ans = sum(1 for q in bank_q if q["gold_answerable"] == "YES")
    (STUDY / "modality_ext_bank.json").write_text(json.dumps(dict(
        bank_id="modality_ext_v2", response_protocol="ANSWERABLE/SOURCE/ANSWER",
        description=f"{len(bank_q)} QA over 6 objects across 3 modalities "
                    f"(2 image, 2 text, 2 pdf-via-pdf2text). {n_ans} answerable / {len(bank_q)-n_ans} unanswerable.",
        base_questions=bank_q), indent=2, ensure_ascii=False))

    orig = json.loads((STUDY / "unified_state_conditioned_bank.json").read_text())
    combined = dict(orig); combined["bank_id"] = "combined_modality_v1"
    combined["description"] = (f"Original {len(orig['base_questions'])} (chart/UI) + {len(bank_q)} "
                               f"modality-ext (image/text/pdf) = {len(orig['base_questions'])+len(bank_q)}.")
    combined["base_questions"] = orig["base_questions"] + bank_q
    (STUDY / "combined_modality_bank.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False))

    # ── regenerate combined 12-object episode (original 6 + new 6) + plant-only ──
    base = json.loads((EPI / "pilot_episode_100k.json").read_text())
    ep = dict(base)
    ep["episode_id"] = "modality_ext_100k_v2"
    ep["model"] = "volcengine/doubao-seed-2-0-pro-260215"
    ep["description"] = ("Combined 12-object NATIVE trunk: original 6 (chart/UI images) + 6 new across "
                         "3 modalities (2 image, 2 text, 2 pdf-via-pdf2text). No prewrite; native compaction.")
    arts = dict(ep["artifacts"]); new_turns = []
    for e in objects_out:
        aid = e["artifact_id"]
        adef = {"plant_type": e["plant_type"], "ocr_keywords": e["ocr_keywords"]}
        if e["plant_is_image"]:
            adef["image_path"] = e["image_path"]
        elif e.get("file_path"):
            adef["file_path"] = e["file_path"]   # copied to workspace by the plant step
        arts[aid] = adef
        new_turns.append({"type": "artifact", "artifact_id": aid,
                          "prompt": e["plant_prompt"], "image": bool(e["plant_is_image"])})
    ep["artifacts"] = arts
    trunk = list(base["trunk"])
    idx = next(i for i, s in enumerate(trunk) if s.get("type") == "checkpoint")
    summary_task = {"type": "work_task",
                    "prompt": "Quick check — list in one line the kinds of materials I've shared so far."}
    ep["trunk"] = trunk[:idx] + new_turns[:3] + [summary_task] + new_turns[3:] + trunk[idx:]
    (EPI / "modality_ext_100k_v2.json").write_text(json.dumps(ep, indent=2, ensure_ascii=False))

    plant = dict(ep); plant["episode_id"] = "modality_ext_100k_plant_v2"
    plant["description"] = "Plant-only v2 (named text/pdf files): 12 artifacts → d0. Resumable stages build d50k/d80k/S2."
    i_d0 = next(i for i, s in enumerate(ep["trunk"])
                if s.get("type") == "checkpoint" and s.get("depth_label") == 0)
    plant["trunk"] = ep["trunk"][:i_d0 + 1]
    (EPI / "modality_ext_100k_plant_v2.json").write_text(json.dumps(plant, indent=2, ensure_ascii=False))

    print(f"objects: {len(objects_out)}  (image={sum(1 for o in objects_out if o['modality']=='image')}, "
          f"text={sum(1 for o in objects_out if o['modality']=='text')}, "
          f"pdf={sum(1 for o in objects_out if o['modality']=='pdf')})")
    print(f"new QA: {len(bank_q)} ({n_ans} ans / {len(bank_q)-n_ans} unans)")
    print(f"combined bank: {len(combined['base_questions'])}")
    print(f"episode: {len(ep['artifacts'])} artifacts, trunk len {len(ep['trunk'])}")
    for o in objects_out:
        extra = o.get("pdf_path") or o.get("image_path") or "text-inline"
        print(f"  {o['artifact_id']:8s} {o['modality']:6s} {o['visual_type']:14s} {extra}")

if __name__ == "__main__":
    main()
