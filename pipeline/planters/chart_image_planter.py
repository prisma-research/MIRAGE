"""
Chart image artifact planter.

Session A: 生成一张真实的 matplotlib 图表（从 CSV 数据），
作为图片附件发送给 agent，要求 agent 分析并保存到 memory。
"""

from __future__ import annotations

import csv
import io
import os
import random
from pathlib import Path

from pipeline.planters.base_planter import BasePlanter, PlantSpec

# 图片输出目录
IMAGES_DIR = Path(__file__).resolve().parent.parent / "generated_images" / "chart"

# 每个 dataset 定义 CSV 数据、图表类型、关键词
DATASETS = [
    {
        "name": "Monthly Revenue 2025",
        "csv": (
            "Month,Revenue_USD,Growth_pct\n"
            "Jan,1200000,5.2\nFeb,1350000,12.5\nMar,1420000,5.2\n"
            "Apr,1380000,-2.8\nMay,1510000,9.4\nJun,1680000,11.3\n"
            "Jul,1590000,-5.4\nAug,1720000,8.2\nSep,1850000,7.6\n"
            "Oct,1930000,4.3\nNov,2100000,8.8\nDec,2350000,11.9"
        ),
        "chart_type": "bar_line",
        "keywords": ["revenue", "monthly", "growth", "USD", "1.2M", "2.35M"],
        "question": "What were the monthly revenue figures?",
    },
    {
        "name": "Product Category Sales Q3 2025",
        "csv": (
            "Category,Units_Sold,Revenue_USD,Return_Rate_pct\n"
            "Electronics,45200,3200000,3.2\n"
            "Clothing,89400,1800000,8.7\n"
            "Home & Garden,23100,920000,2.1\n"
            "Sports,31500,1450000,4.8\n"
            "Books,67800,340000,1.2"
        ),
        "chart_type": "grouped_bar",
        "keywords": ["Electronics", "Clothing", "units sold", "return rate", "Q3"],
        "question": "What were the sales figures by product category?",
    },
    {
        "name": "Website Traffic Sources Jan 2026",
        "csv": (
            "Source,Sessions,Bounce_Rate_pct,Avg_Session_Min\n"
            "Organic Search,125000,42.3,4.2\n"
            "Direct,87500,38.1,5.1\n"
            "Social Media,43200,61.8,2.3\n"
            "Email,28900,29.4,6.8\n"
            "Paid Search,19600,55.2,3.1\n"
            "Referral,12400,44.7,3.9"
        ),
        "chart_type": "pie_bar",
        "keywords": ["organic search", "sessions", "bounce rate", "social media", "traffic"],
        "question": "What were the traffic source breakdown figures?",
    },
    {
        "name": "Employee Satisfaction Survey 2025",
        "csv": (
            "Department,Score_out_of_10,Response_Rate_pct,Headcount\n"
            "Engineering,7.8,92,145\n"
            "Sales,6.9,87,89\n"
            "Marketing,7.4,95,56\n"
            "Operations,7.1,83,112\n"
            "HR,8.2,98,23\n"
            "Finance,7.6,91,34"
        ),
        "chart_type": "horizontal_bar",
        "keywords": ["satisfaction", "score", "department", "response rate", "Engineering"],
        "question": "What were the employee satisfaction scores by department?",
    },
    {
        "name": "Cloud Market Share Q2 2025",
        "csv": (
            "Provider,Market_Share_pct,Revenue_B_USD,YoY_Growth_pct\n"
            "AWS,31.0,26.3,17.2\n"
            "Azure,25.0,21.2,29.1\n"
            "Google Cloud,11.0,9.3,28.5\n"
            "Alibaba Cloud,4.0,3.4,6.8\n"
            "IBM Cloud,3.0,2.5,-2.1\n"
            "Others,26.0,22.1,12.4"
        ),
        "chart_type": "pie_bar",
        "keywords": ["AWS", "Azure", "cloud", "market share", "31%"],
        "question": "What was the cloud infrastructure market share breakdown?",
    },
    {
        "name": "Global Temperature Anomaly 2015-2025",
        "csv": (
            "Year,Anomaly_C,CO2_ppm\n"
            "2015,0.87,400.8\n"
            "2016,1.01,404.2\n"
            "2017,0.92,406.6\n"
            "2018,0.83,408.5\n"
            "2019,0.98,411.4\n"
            "2020,1.02,414.2\n"
            "2021,0.84,416.4\n"
            "2022,0.89,418.6\n"
            "2023,1.17,421.1\n"
            "2024,1.29,424.0\n"
            "2025,1.35,427.3"
        ),
        "chart_type": "bar_line",
        "keywords": ["temperature", "anomaly", "CO2", "2025", "1.35°C"],
        "question": "What was the global temperature anomaly trend over the decade?",
    },
    {
        "name": "Startup Funding by Stage H1 2025",
        "csv": (
            "Stage,Deal_Count,Total_B_USD,Median_M_USD\n"
            "Pre-Seed,4210,2.8,0.5\n"
            "Seed,3150,8.4,2.1\n"
            "Series A,1820,24.7,11.5\n"
            "Series B,890,31.2,28.4\n"
            "Series C,420,22.8,45.2\n"
            "Series D+,180,38.6,142.0\n"
            "Growth,95,52.3,410.0"
        ),
        "chart_type": "grouped_bar",
        "keywords": ["startup", "funding", "Series A", "Pre-Seed", "billion"],
        "question": "What was the startup funding distribution by stage?",
    },
    {
        "name": "GPU Performance Benchmark 2025",
        "csv": (
            "GPU_Model,FP32_TFLOPS,Memory_GB,TDP_W,Price_USD\n"
            "NVIDIA H100,67.0,80,700,30000\n"
            "NVIDIA A100,19.5,80,400,15000\n"
            "NVIDIA RTX 4090,82.6,24,450,1599\n"
            "AMD MI300X,163.4,192,750,15000\n"
            "Intel Gaudi 3,64.0,128,600,12500\n"
            "NVIDIA B200,144.0,192,1000,40000"
        ),
        "chart_type": "horizontal_bar",
        "keywords": ["H100", "TFLOPS", "GPU", "NVIDIA", "B200"],
        "question": "What were the GPU performance benchmark results?",
    },
    {
        "name": "Global Smartphone Shipments Q3 2025",
        "csv": (
            "Brand,Units_M,Market_Share_pct,YoY_Change_pct\n"
            "Samsung,58.2,19.4,-2.1\n"
            "Apple,52.1,17.3,6.8\n"
            "Xiaomi,43.7,14.5,12.3\n"
            "OPPO,28.4,9.4,-1.5\n"
            "vivo,25.8,8.6,3.2\n"
            "Others,92.6,30.8,4.7"
        ),
        "chart_type": "pie_bar",
        "keywords": ["Samsung", "Apple", "smartphone", "shipments", "million units"],
        "question": "What were the smartphone shipment figures by brand?",
    },
]


def _parse_csv(csv_text: str) -> list[dict]:
    """将 CSV 字符串解析为 dict 列表"""
    reader = csv.DictReader(io.StringIO(csv_text))
    return list(reader)


def _generate_chart(dataset: dict, output_path: Path) -> None:
    """
    根据 dataset 配置，用 matplotlib 生成图表并保存为 PNG。
    通用列名逻辑：第一列=标签，后续列=数值，按 chart_type 选择展示方式。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _parse_csv(dataset["csv"])
    chart_type = dataset["chart_type"]
    cols = list(rows[0].keys())
    label_col = cols[0]
    labels = [r[label_col] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#fafafa")

    if chart_type == "bar_line":
        val_col, line_col = cols[1], cols[2]
        values = [float(r[val_col]) for r in rows]
        line_values = [float(r[line_col]) for r in rows]

        # 如果数值很大，自动缩放到百万
        scale = 1e6 if max(values) > 100000 else 1
        scaled = [v / scale for v in values]
        val_label = val_col.replace("_", " ")
        if scale == 1e6:
            val_label += " (M)"

        x = np.arange(len(labels))
        ax.bar(x, scaled, color="#4C72B0", alpha=0.8, label=val_label)
        ax.set_ylabel(val_label, fontsize=11)
        ax.set_xlabel(label_col.replace("_", " "), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")

        ax2 = ax.twinx()
        line_label = line_col.replace("_", " ")
        ax2.plot(x, line_values, color="#DD8452", marker="o", linewidth=2, label=line_label)
        ax2.set_ylabel(line_label, fontsize=11, color="#DD8452")
        ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax.set_title(dataset["name"], fontsize=14, fontweight="bold")

    elif chart_type == "grouped_bar":
        col_a, col_b = cols[1], cols[2]
        vals_a = [float(r[col_a]) for r in rows]
        vals_b = [float(r[col_b]) for r in rows]

        scale_a = 1000 if max(vals_a) > 5000 else 1
        scale_b = 1e6 if max(vals_b) > 100000 else 1
        a_scaled = [v / scale_a for v in vals_a]
        b_scaled = [v / scale_b for v in vals_b]
        a_label = col_a.replace("_", " ") + (" (K)" if scale_a == 1000 else "")
        b_label = col_b.replace("_", " ") + (" (M)" if scale_b == 1e6 else "")

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, a_scaled, width, label=a_label, color="#4C72B0")
        ax.bar(x + width / 2, b_scaled, width, label=b_label, color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend()
        ax.set_title(dataset["name"], fontsize=14, fontweight="bold")
        ax.set_ylabel("Value", fontsize=11)

    elif chart_type == "pie_bar":
        pie_col, bar_col = cols[1], cols[2]
        pie_vals = [float(r[pie_col]) for r in rows]
        bar_vals = [float(r[bar_col]) for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#fafafa")
        palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD",
                    "#DA8BC3", "#8C8C8C"]
        colors = palette[:len(labels)]

        ax1.pie(pie_vals, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        ax1.set_title(pie_col.replace("_", " "), fontsize=12, fontweight="bold")

        x = np.arange(len(labels))
        ax2.barh(x, bar_vals, color=colors)
        ax2.set_yticks(x)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel(bar_col.replace("_", " "), fontsize=11)
        ax2.set_title(bar_col.replace("_", " "), fontsize=12, fontweight="bold")

        fig.suptitle(dataset["name"], fontsize=14, fontweight="bold")
        ax = ax1

    elif chart_type == "horizontal_bar":
        val_col = cols[1]
        values = [float(r[val_col]) for r in rows]
        # 取第三列（如果有）作为注释
        anno_col = cols[2] if len(cols) > 2 else None

        y = np.arange(len(labels))
        vmin, vmax = min(values), max(values)
        mid = vmin + (vmax - vmin) * 0.6
        low = vmin + (vmax - vmin) * 0.3
        colors = ["#55A868" if v >= mid else "#CCB974" if v >= low else "#C44E52" for v in values]
        ax.barh(y, values, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(val_col.replace("_", " "), fontsize=11)
        ax.set_title(dataset["name"], fontsize=14, fontweight="bold")

        if anno_col:
            for i, (v, r) in enumerate(zip(values, rows)):
                anno = r[anno_col]
                ax.text(v + (vmax * 0.01), i, f"{v}  ({anno})", va="center", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


class ChartImagePlanter(BasePlanter):
    plant_type = "chart_image"

    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        rng = random.Random(seed)
        dataset = DATASETS[seed % len(DATASETS)]
        artifact_id = self.make_artifact_id("chart")

        # 生成图表图片
        image_filename = f"{artifact_id}.png"
        image_path = IMAGES_DIR / image_filename
        _generate_chart(dataset, image_path)

        plant_prompt = (
            f"I'm sharing a chart image with you. It shows: {dataset['name']}.\n\n"
            f"Please analyze the chart and extract the key data points and trends. "
            f"Save a detailed summary to memory with the identifier '{artifact_id}'. "
            f"Include the specific numbers visible in the chart."
        )

        artifact_repr = (
            f"Chart: {dataset['name']}\n"
            f"Artifact ID: {artifact_id}\n"
            f"Data:\n{dataset['csv']}"
        )

        return PlantSpec(
            artifact_id=artifact_id,
            plant_type=self.plant_type,
            plant_prompt=plant_prompt,
            artifact_ocr_keywords=dataset["keywords"],
            artifact_repr=artifact_repr,
            image_path=str(image_path),
            metadata={"dataset_name": dataset["name"], "chart_type": dataset["chart_type"]},
        )
