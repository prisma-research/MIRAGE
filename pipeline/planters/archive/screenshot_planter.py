"""
Screenshot artifact planter.

Session A: 用 Playwright 截取真实网页的屏幕截图，
作为图片附件发送给 agent，要求 agent 记录关键信息到 memory。

使用稳定的公开网页（Wikipedia、文档站点等）确保可复现。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

from pipeline.planters.base_planter import BasePlanter, PlantSpec

IMAGES_DIR = Path(__file__).resolve().parent.parent / "generated_images" / "screenshot"
# 缓存目录，存储已截图页面的文本内容
CACHE_DIR = IMAGES_DIR / ".cache"

# 8 个真实网页截图目标 —— 选择内容稳定、数据丰富的公开页面
SCREENSHOT_TARGETS = [
    {
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "description": "Wikipedia article about Python programming language",
        "scroll_to": None,
        "keywords": ["Python", "programming language", "Guido van Rossum",
                      "interpreted", "high-level"],
        "extract_selector": "#mw-content-text .mw-parser-output > p",
        "viewport": {"width": 1280, "height": 900},
    },
    {
        "url": "https://en.wikipedia.org/wiki/Large_language_model",
        "description": "Wikipedia article about Large Language Models",
        "scroll_to": None,
        "keywords": ["large language model", "LLM", "transformer",
                      "neural network", "parameters"],
        "extract_selector": "#mw-content-text .mw-parser-output > p",
        "viewport": {"width": 1280, "height": 900},
    },
    {
        "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        "description": "Wikipedia article about Transformer architecture",
        "scroll_to": None,
        "keywords": ["Transformer", "attention", "self-attention",
                      "encoder", "decoder"],
        "extract_selector": "#mw-content-text .mw-parser-output > p",
        "viewport": {"width": 1280, "height": 900},
    },
    {
        "url": "https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue",
        "description": "Wikipedia list of largest companies by revenue with data table",
        "scroll_to": None,
        "keywords": ["Walmart", "revenue", "billion", "Amazon",
                      "largest companies"],
        "extract_selector": "#mw-content-text .mw-parser-output > p, #mw-content-text .wikitable",
        "viewport": {"width": 1280, "height": 900},
    },
    {
        "url": "https://en.wikipedia.org/wiki/World_population",
        "description": "Wikipedia article about world population statistics",
        "scroll_to": None,
        "keywords": ["world population", "billion", "growth rate",
                      "fertility", "United Nations"],
        "extract_selector": "#mw-content-text .mw-parser-output > p",
        "viewport": {"width": 1280, "height": 900},
    },
    {
        "url": "https://en.wikipedia.org/wiki/Graphics_processing_unit",
        "description": "Wikipedia article about GPU technology",
        "scroll_to": None,
        "keywords": ["GPU", "graphics processing unit", "NVIDIA",
                      "parallel", "CUDA"],
        "extract_selector": "#mw-content-text .mw-parser-output > p",
        "viewport": {"width": 1280, "height": 900},
    },
    {
        "url": "https://en.wikipedia.org/wiki/Electric_vehicle",
        "description": "Wikipedia article about electric vehicles",
        "scroll_to": None,
        "keywords": ["electric vehicle", "battery", "Tesla",
                      "BEV", "charging"],
        "extract_selector": "#mw-content-text .mw-parser-output > p",
        "viewport": {"width": 1280, "height": 900},
    },
    {
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "description": "Wikipedia article about Artificial Intelligence",
        "scroll_to": None,
        "keywords": ["artificial intelligence", "machine learning",
                      "neural network", "deep learning", "AI"],
        "extract_selector": "#mw-content-text .mw-parser-output > p",
        "viewport": {"width": 1280, "height": 900},
    },
]


def _url_hash(url: str) -> str:
    """为 URL 生成短 hash 用于缓存文件名"""
    return hashlib.md5(url.encode()).hexdigest()[:10]


def _take_screenshot(target: dict, output_path: Path) -> str:
    """
    用 Playwright 截取真实网页截图，返回提取的页面文本。
    截图保存到 output_path，文本缓存到 CACHE_DIR。
    如果截图已存在且缓存有效，直接返回缓存文本。
    """
    from playwright.sync_api import sync_playwright

    url = target["url"]
    url_id = _url_hash(url)
    cache_file = CACHE_DIR / f"{url_id}.json"

    # 如果截图和缓存都存在，跳过重新截取
    if output_path.exists() and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return cached.get("extracted_text", "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    vp = target.get("viewport", {"width": 1280, "height": 900})

    # 检测系统代理配置并传递给 Chromium
    proxy_url = (
        os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
    )
    launch_args: dict = {"headless": True}
    if proxy_url:
        launch_args["proxy"] = {"server": proxy_url}

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_args)
        context = browser.new_context(
            viewport=vp,
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)

        # 关闭 Wikipedia 的 cookie/consent 弹窗（如果存在）
        try:
            consent_btn = page.query_selector(
                ".vector-settings-icon, .uls-settings-close, "
                ".mw-dismissable-notice .mw-dismiss"
            )
            if consent_btn:
                consent_btn.click()
                page.wait_for_timeout(500)
        except Exception:
            pass

        # 可选：滚动到指定元素
        scroll_to = target.get("scroll_to")
        if scroll_to:
            try:
                page.locator(scroll_to).scroll_into_view_if_needed()
                page.wait_for_timeout(300)
            except Exception:
                pass

        # 截图
        page.screenshot(path=str(output_path), full_page=False)

        # 提取页面文本内容
        extract_selector = target.get(
            "extract_selector",
            "#mw-content-text .mw-parser-output > p",
        )
        extracted_text = ""
        try:
            elements = page.query_selector_all(extract_selector)
            texts = []
            for el in elements[:15]:
                t = el.inner_text()
                if t and len(t.strip()) > 20:
                    texts.append(t.strip())
            extracted_text = "\n\n".join(texts)
            # 截断过长文本
            if len(extracted_text) > 3000:
                extracted_text = extracted_text[:3000] + "..."
        except Exception:
            extracted_text = f"Screenshot of {url}"

        browser.close()

    # 缓存提取的文本
    cache_data = {
        "url": url,
        "extracted_text": extracted_text,
        "description": target["description"],
    }
    cache_file.write_text(
        json.dumps(cache_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return extracted_text


class ScreenshotPlanter(BasePlanter):
    plant_type = "screenshot"

    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        target = SCREENSHOT_TARGETS[seed % len(SCREENSHOT_TARGETS)]
        artifact_id = self.make_artifact_id("screenshot")

        image_filename = f"{artifact_id}.png"
        image_path = IMAGES_DIR / image_filename

        extracted_text = _take_screenshot(target, image_path)

        plant_prompt = (
            f"I'm sharing a screenshot of a web page: {target['description']}. "
            f"Please read the content shown in the screenshot and extract the "
            f"key facts, statistics, and information. Save a detailed summary "
            f"to memory with the identifier '{artifact_id}' so I can reference "
            f"it in a future session."
        )

        artifact_repr = (
            f"Web screenshot: {target['description']}\n"
            f"URL: {target['url']}\n"
            f"Artifact ID: {artifact_id}\n"
            f"Content:\n{extracted_text[:2000]}"
        )

        return PlantSpec(
            artifact_id=artifact_id,
            plant_type=self.plant_type,
            plant_prompt=plant_prompt,
            artifact_ocr_keywords=target["keywords"],
            artifact_repr=artifact_repr,
            image_path=str(image_path),
            metadata={
                "url": target["url"],
                "description": target["description"],
            },
        )
