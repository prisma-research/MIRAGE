"""
UI State image artifact planter.

Session A: 用 Pillow 渲染一张 dashboard/监控面板风格的 UI 截图，
作为图片附件发送给 agent，要求 agent 记录 UI 状态到 memory。
"""

from __future__ import annotations

import random
from pathlib import Path

from pipeline.planters.base_planter import BasePlanter, PlantSpec

IMAGES_DIR = Path(__file__).resolve().parent.parent / "generated_images" / "ui"

UI_STATES = [
    {
        "name": "Analytics Dashboard — March 13 2026 14:32",
        "cards": [
            ("Active Users", "14,283", "↑ 8.2%", "#27ae60"),
            ("Sessions Today", "47,891", "", "#2980b9"),
            ("Avg Session", "4m 23s", "", "#8e44ad"),
            ("Conversion Rate", "3.84%", "↑ 0.3%", "#27ae60"),
            ("Revenue Today", "$28,473", "↑ 12.1%", "#27ae60"),
            ("Response Time", "142ms p95", "", "#2980b9"),
        ],
        "footer": "Top page: /products/ai-tools (3,421 views) | Top source: Organic Search (41.2%)",
        "keywords": ["14,283", "analytics", "conversion", "3.84%", "dashboard"],
    },
    {
        "name": "CI/CD Pipeline Status — Build #4521",
        "cards": [
            ("Build", "#4521", "merge to main", "#2980b9"),
            ("Lint", "✓ Passed", "", "#27ae60"),
            ("Unit Tests", "312 passed", "0 failed", "#27ae60"),
            ("Integration", "89 passed", "", "#27ae60"),
            ("Security Scan", "0 critical", "", "#27ae60"),
            ("Deploy", "Running...", "2m 14s", "#f39c12"),
        ],
        "footer": "Target: us-east-1 ECS | Image: app:4521 (847MB) | Commit: a3f8b92",
        "keywords": ["build #4521", "312 passed", "deploy", "ECS", "production"],
    },
    {
        "name": "Inventory Management — Warehouse B Status",
        "cards": [
            ("Capacity", "78.3%", "15,660 / 20,000", "#f39c12"),
            ("Orders Shipped", "847", "today", "#2980b9"),
            ("Pending Orders", "234", "", "#e74c3c"),
            ("Temperature", "18.4°C", "within spec", "#27ae60"),
            ("Low Stock Alerts", "2 items", "SKU-8821, SKU-3301", "#e74c3c"),
            ("Inbound", "PO-44521", "Mar 15, 2,400 units", "#2980b9"),
        ],
        "footer": "Last audit: March 10 | Reorder alerts: SKU-8821 (14 units), SKU-3301 (8 units)",
        "keywords": ["Warehouse B", "78.3%", "SKU-8821", "PO-44521", "inventory"],
    },
    {
        "name": "Kubernetes Cluster — Production Namespace",
        "cards": [
            ("Nodes", "12 running", "2 pending", "#27ae60"),
            ("CPU Avg", "64.2%", "peak 91.3%", "#f39c12"),
            ("Memory", "71.8%", "234 / 326 GB", "#f39c12"),
            ("Pods", "847 running", "3 pending", "#27ae60"),
            ("HPA Event", "8 → 12", "api-deployment", "#2980b9"),
            ("Alert", "memory-pressure", "prod-node-07", "#e74c3c"),
        ],
        "footer": "Cluster: prod-us-east-1 | Last HPA scale: 14:18 UTC | Threshold: 85%",
        "keywords": ["Kubernetes", "847 pods", "64.2% CPU", "prod-node-07", "HPA"],
    },
    {
        "name": "PostgreSQL Database Monitor — prod-db-primary",
        "cards": [
            ("Connections", "247 / 500", "49.4% used", "#f39c12"),
            ("TPS", "3,842", "↑ 12.3%", "#27ae60"),
            ("Cache Hit", "99.2%", "shared_buffers", "#27ae60"),
            ("Repl Lag", "0.8 sec", "2 replicas", "#27ae60"),
            ("Disk Usage", "1.42 TB", "73% of 2 TB", "#f39c12"),
            ("Deadlocks", "3 today", "last: 09:47 UTC", "#e74c3c"),
        ],
        "footer": "PG 16.2 | uptime: 47d 13h | last vacuum: 2h ago | WAL rate: 28 MB/s",
        "keywords": ["PostgreSQL", "3,842 TPS", "247 connections", "1.42 TB", "deadlocks"],
    },
    {
        "name": "Social Media Campaign — #LaunchDay Analytics",
        "cards": [
            ("Impressions", "2.4M", "↑ 340%", "#27ae60"),
            ("Engagement", "187K", "7.8% rate", "#27ae60"),
            ("Clicks", "34,521", "CTR 1.44%", "#2980b9"),
            ("Followers", "+8,420", "since launch", "#27ae60"),
            ("Sentiment", "78% positive", "12% negative", "#f39c12"),
            ("Top Post", "42K likes", "video reel", "#2980b9"),
        ],
        "footer": "Campaign: Mar 10-14 | Budget spent: $12,400 / $15,000 | CPM: $5.17",
        "keywords": ["2.4M impressions", "engagement", "CTR", "sentiment", "#LaunchDay"],
    },
    {
        "name": "E-Commerce Order Dashboard — March 14, 2026",
        "cards": [
            ("Orders Today", "1,247", "↑ 8.4%", "#27ae60"),
            ("GMV", "$89,340", "avg $71.64", "#2980b9"),
            ("Cart Abandon", "68.2%", "↓ 2.1%", "#f39c12"),
            ("Returns", "47 pending", "$3,218 value", "#e74c3c"),
            ("Fulfillment", "94.1%", "same-day", "#27ae60"),
            ("Top SKU", "SKU-4471", "312 units", "#2980b9"),
        ],
        "footer": "Warehouse: US-East | Avg ship time: 1.8 days | CS tickets: 23 open",
        "keywords": ["1,247 orders", "$89,340 GMV", "cart abandon", "SKU-4471", "fulfillment"],
    },
    {
        "name": "Network Security Operations Center — Live",
        "cards": [
            ("Threats", "14 blocked", "last hour", "#e74c3c"),
            ("Firewall", "2.1M rules", "99.97% pass", "#27ae60"),
            ("DDoS", "0 active", "clean", "#27ae60"),
            ("SSL Certs", "3 expiring", "within 7 days", "#f39c12"),
            ("VPN Users", "342 active", "peak: 518", "#2980b9"),
            ("Anomalies", "2 detected", "ML score 0.87", "#e74c3c"),
        ],
        "footer": "SOC Level: DEFCON 5 (normal) | Last incident: 6h ago | Uptime: 99.994%",
        "keywords": ["14 threats blocked", "firewall", "SSL certs", "VPN", "anomalies"],
    },
]


def _render_dashboard(state: dict, output_path: Path) -> None:
    """用 Pillow 渲染一张 dashboard 风格的 UI 截图"""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1000, 600
    card_cols, card_rows = 3, 2
    card_w, card_h = 280, 180
    pad_x, pad_y = 30, 20
    header_h = 70
    footer_h = 50

    img = Image.new("RGB", (W, H), "#1e2330")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        card_title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        card_value_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        card_sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        footer_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        card_title_font = card_value_font = card_sub_font = footer_font = title_font

    # 标题栏
    draw.rectangle([(0, 0), (W, header_h)], fill="#2c3e50")
    draw.text((pad_x, 22), state["name"], fill="#ecf0f1", font=title_font)

    # 状态指示灯
    draw.ellipse([(W - 50, 28), (W - 36, 42)], fill="#27ae60")

    # 卡片网格
    cards = state["cards"]
    start_y = header_h + pad_y
    for idx, (label, value, sub, color) in enumerate(cards):
        row = idx // card_cols
        col = idx % card_cols
        x = pad_x + col * (card_w + pad_x)
        y = start_y + row * (card_h + pad_y)

        # 卡片背景（圆角矩形模拟）
        draw.rounded_rectangle([(x, y), (x + card_w, y + card_h)], radius=10, fill="#2c3e50")

        # 左侧色条
        draw.rectangle([(x, y + 8), (x + 4, y + card_h - 8)], fill=color)

        # 卡片标题
        draw.text((x + 16, y + 14), label, fill="#95a5a6", font=card_title_font)

        # 卡片主值
        draw.text((x + 16, y + 50), value, fill="#ecf0f1", font=card_value_font)

        # 卡片副文本
        if sub:
            sub_color = color if "↑" in sub or "passed" in sub.lower() else "#bdc3c7"
            draw.text((x + 16, y + card_h - 40), sub, fill=sub_color, font=card_sub_font)

    # 页脚
    footer_y = H - footer_h
    draw.line([(pad_x, footer_y), (W - pad_x, footer_y)], fill="#34495e", width=1)
    footer_text = state.get("footer", "")
    if len(footer_text) > 100:
        footer_text = footer_text[:100] + "..."
    draw.text((pad_x, footer_y + 12), footer_text, fill="#7f8c8d", font=footer_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG")


class UIStatePlanter(BasePlanter):
    plant_type = "ui_state_image"

    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        rng = random.Random(seed)
        state = UI_STATES[seed % len(UI_STATES)]
        artifact_id = self.make_artifact_id("ui")

        image_filename = f"{artifact_id}.png"
        image_path = IMAGES_DIR / image_filename
        _render_dashboard(state, image_path)

        plant_prompt = (
            f"I'm sharing a screenshot of my dashboard with you. "
            f"Please document the current UI state shown in the image. "
            f"Save this UI state snapshot to memory with the identifier "
            f"'{artifact_id}'. I'll need to refer back to these specific values later."
        )

        artifact_repr = (
            f"UI State: {state['name']}\n"
            f"Artifact ID: {artifact_id}\n"
            f"Cards: {', '.join(f'{c[0]}={c[1]}' for c in state['cards'])}"
        )

        return PlantSpec(
            artifact_id=artifact_id,
            plant_type=self.plant_type,
            plant_prompt=plant_prompt,
            artifact_ocr_keywords=state["keywords"],
            artifact_repr=artifact_repr,
            image_path=str(image_path),
            metadata={"ui_name": state["name"]},
        )
