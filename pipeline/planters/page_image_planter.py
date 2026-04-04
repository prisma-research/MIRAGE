"""
Page image artifact planter.

Session A: 用 Pillow 渲染一张文档页面图片（模拟 PDF/报告截图），
作为图片附件发送给 agent，要求 agent 提取关键信息并保存到 memory。
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

from pipeline.planters.base_planter import BasePlanter, PlantSpec

IMAGES_DIR = Path(__file__).resolve().parent.parent / "generated_images" / "page"

PAGE_EXCERPTS = [
    {
        "title": "Q4 2025 Earnings Report — Executive Summary",
        "content": (
            "Total revenue for Q4 2025 reached $847.3 million, representing a 23.4% "
            "year-over-year increase. Operating income was $142.6 million (16.8% margin). "
            "The North America segment contributed 58% of total revenue ($491.4M), "
            "while International operations grew 31% to $355.9M. "
            "Free cash flow improved to $198.2M, up from $143.1M in Q4 2024. "
            "The board approved a $0.42 per share quarterly dividend."
        ),
        "keywords": ["Q4 2025", "847.3 million", "23.4%", "operating income", "dividend"],
    },
    {
        "title": "Project Phoenix — Technical Specification v2.1",
        "content": (
            "System architecture: Microservices deployment on Kubernetes 1.28. "
            "Primary database: PostgreSQL 16 with read replicas in 3 availability zones. "
            "API gateway: Kong 3.4 with rate limiting at 10,000 req/min per tenant. "
            "Expected throughput: 50,000 concurrent users. SLA target: 99.95% uptime. "
            "Data retention: 7 years for audit logs, 90 days for session data. "
            "Encryption: AES-256 at rest, TLS 1.3 in transit."
        ),
        "keywords": ["PostgreSQL", "Kubernetes", "50,000 concurrent", "99.95%", "AES-256"],
    },
    {
        "title": "Clinical Trial Results — Drug XR-471 Phase 2",
        "content": (
            "Primary endpoint: Reduction in systolic blood pressure at 12 weeks. "
            "Treatment group (n=312): Mean reduction 18.7 mmHg (±4.2). "
            "Placebo group (n=308): Mean reduction 6.3 mmHg (±3.8). "
            "P-value: <0.001. Adverse events: 12.8% treatment vs 9.4% placebo. "
            "Most common AE: headache (6.1%), dizziness (4.3%). "
            "No serious adverse events in treatment group."
        ),
        "keywords": ["XR-471", "18.7 mmHg", "placebo", "p-value", "adverse events"],
    },
    {
        "title": "2025 Urban Mobility Report — City of Portland",
        "content": (
            "Daily transit ridership: 387,400 (up 14.2% from 2024). "
            "Bus network: 72 routes, 2,847 stops, average headway 8.3 minutes. "
            "Light rail: 5 lines, 97 stations, on-time performance 91.2%. "
            "Cycling infrastructure: 284 miles protected lanes, 12,400 daily cyclists. "
            "EV charging: 847 public stations installed in 2025, target 2,000 by 2027. "
            "Carbon reduction: 31,400 tonnes CO2 avoided vs. personal vehicle baseline."
        ),
        "keywords": ["Portland", "387,400", "transit ridership", "on-time", "CO2"],
    },
    {
        "title": "Attention Is All You Need — Abstract & Results",
        "content": (
            "We propose a new simple network architecture, the Transformer, based solely on "
            "attention mechanisms, dispensing with recurrence and convolutions entirely. "
            "On the WMT 2014 English-to-German translation task, the Transformer achieves "
            "28.4 BLEU, improving over the existing best results by over 2 BLEU. "
            "On the WMT 2014 English-to-French translation task, the model establishes a new "
            "single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days "
            "on eight GPUs. The base model uses d_model=512, d_ff=2048, h=8 attention heads, "
            "and 6 encoder/decoder layers with a total of 65M parameters."
        ),
        "keywords": ["Transformer", "attention", "BLEU", "28.4", "41.0", "65M parameters"],
    },
    {
        "title": "SaaS Platform — Terms of Service (Section 7: Data Retention)",
        "content": (
            "7.1 Data Retention Period. Customer Data will be retained for the duration of "
            "the subscription term plus 90 calendar days following termination. "
            "7.2 Backup Copies. Automated backups are created every 6 hours and retained "
            "for 30 days. Point-in-time recovery is available for the most recent 72 hours. "
            "7.3 Data Deletion. Upon written request, all Customer Data including backups will "
            "be permanently deleted within 14 business days. A deletion certificate will be "
            "provided. Cost: $0.00 for accounts under 500GB; $250.00 per TB for larger datasets. "
            "7.4 Legal Holds. Data subject to legal hold will be preserved indefinitely "
            "regardless of retention policy. Customer must notify within 5 business days."
        ),
        "keywords": ["retention", "90 days", "backup", "deletion", "14 business days"],
    },
    {
        "title": "REST API Rate Limits — Developer Documentation v3.2",
        "content": (
            "Rate limits apply per API key. Free tier: 100 requests/minute, 5,000/day. "
            "Pro tier: 1,000 requests/minute, 100,000/day, burst allowance 50 req/sec. "
            "Enterprise tier: 10,000 requests/minute, unlimited daily, burst 500 req/sec. "
            "Rate limit headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset. "
            "HTTP 429 responses include Retry-After header (seconds). "
            "Webhook endpoints: 30 second timeout, 3 automatic retries with exponential backoff. "
            "Batch endpoints: Maximum 100 items per request, 50MB payload limit. "
            "WebSocket connections: 5 concurrent per key (Free), 50 (Pro), 500 (Enterprise)."
        ),
        "keywords": ["rate limit", "1,000 requests", "429", "webhook", "API"],
    },
    {
        "title": "Board Meeting Minutes — March 7, 2026",
        "content": (
            "Meeting called to order at 10:00 AM. Present: 8 of 9 board members (quorum met). "
            "Q4 2025 financial results approved unanimously: Revenue $847.3M (+23.4% YoY), "
            "Net income $112.8M (13.3% margin). "
            "Resolution 2026-03: Approved $45M capital expenditure for new data center in "
            "Frankfurt, Germany. Expected operational by Q3 2026, capacity 12MW. "
            "Resolution 2026-04: Authorized 2M share buyback program, not to exceed $180M. "
            "CEO search committee update: 3 finalist candidates, final interviews by March 28. "
            "Next meeting: April 11, 2026 at 10:00 AM. Meeting adjourned at 12:47 PM."
        ),
        "keywords": ["board meeting", "$847.3M", "Frankfurt", "buyback", "$45M"],
    },
    {
        "title": "Incident Post-Mortem — Database Outage #INC-20260301",
        "content": (
            "Duration: March 1, 2026 03:14 UTC to 05:42 UTC (2 hours 28 minutes). "
            "Impact: 100% of write operations failed; read operations degraded to ~40% success. "
            "Affected users: approximately 34,000 active sessions. Revenue impact: ~$127,000. "
            "Root cause: Automated failover triggered by disk latency spike (>200ms p99) on "
            "primary node pg-prod-01. Failover to pg-prod-02 failed due to replication lag "
            "of 47 seconds exceeding the 30-second threshold. "
            "Resolution: Manual promotion of pg-prod-03 (sync replica, 0 lag). "
            "Action items: (1) Increase replication lag threshold to 60s. "
            "(2) Add automated pg-prod-03 as priority failover target. "
            "(3) Deploy disk I/O monitoring with 100ms alert threshold."
        ),
        "keywords": ["outage", "2 hours 28 minutes", "pg-prod-01", "failover", "$127,000"],
    },
]


def _render_document_page(page: dict, output_path: Path) -> None:
    """用 Pillow 渲染一张仿文档页面的图片"""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 800, 1000
    margin = 60
    text_width = W - 2 * margin

    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # 尝试加载字体，降级到默认
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 22)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 15)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        header_font = ImageFont.load_default()

    # 页眉线
    draw.line([(margin, 40), (W - margin, 40)], fill="#333333", width=2)
    draw.text((margin, 20), "CONFIDENTIAL", fill="#999999", font=header_font)
    draw.text((W - margin - 60, 20), "Page 1", fill="#999999", font=header_font)

    # 标题
    y = 60
    title = page["title"]
    wrapped_title = textwrap.fill(title, width=50)
    draw.text((margin, y), wrapped_title, fill="#1a1a1a", font=title_font)
    y += 30 * wrapped_title.count("\n") + 40

    # 分隔线
    draw.line([(margin, y), (W - margin, y)], fill="#cccccc", width=1)
    y += 20

    # 正文内容——逐段渲染
    content = page["content"]
    sentences = content.split(". ")
    char_per_line = 70
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not sentence.endswith("."):
            sentence += "."
        wrapped = textwrap.fill(sentence, width=char_per_line)
        for line in wrapped.split("\n"):
            draw.text((margin, y), line, fill="#333333", font=body_font)
            y += 22
        y += 8
        if y > H - 80:
            break

    # 页脚线
    draw.line([(margin, H - 50), (W - margin, H - 50)], fill="#333333", width=1)
    draw.text((margin, H - 40), "© 2025 Confidential Report", fill="#999999", font=header_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG")


class PageImagePlanter(BasePlanter):
    plant_type = "page_image"

    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        rng = random.Random(seed)
        page = PAGE_EXCERPTS[seed % len(PAGE_EXCERPTS)]
        artifact_id = self.make_artifact_id("page")

        image_filename = f"{artifact_id}.png"
        image_path = IMAGES_DIR / image_filename
        _render_document_page(page, image_path)

        plant_prompt = (
            f"I'm sharing a document page image with you. "
            f"Please read and extract the key information from it. "
            f"Save the important figures and findings to memory using the identifier "
            f"'{artifact_id}'. I'll need to reference this document in our next session."
        )

        artifact_repr = (
            f"Document: {page['title']}\n"
            f"Artifact ID: {artifact_id}\n"
            f"Content:\n{page['content']}"
        )

        return PlantSpec(
            artifact_id=artifact_id,
            plant_type=self.plant_type,
            plant_prompt=plant_prompt,
            artifact_ocr_keywords=page["keywords"],
            artifact_repr=artifact_repr,
            image_path=str(image_path),
            metadata={"title": page["title"]},
        )
