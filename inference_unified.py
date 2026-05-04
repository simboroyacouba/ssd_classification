"""
Inference SSD unifié — un seul modèle, toutes classes

Usage :
  python inference_unified.py --input ./images/
  python inference_unified.py --input photo.jpg --display
  python inference_unified.py --model runs/detect/train/ssd_unified_.../best_model.pth --input ./images/
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    ssd300_vgg16, ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.ssd import SSDClassificationHead
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONSTANTES
# =============================================================================

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))

CLASS_THRESHOLDS = {
    "panneau_solaire":       0.50,
    "batiment_peint":        0.30,
    "batiment_non_enduit":   0.35,
    "batiment_enduit":       0.35,
    "menuiserie_metallique": 0.40,
}

COLORS = {
    "panneau_solaire":       (255, 215,   0),
    "batiment_peint":        (  0, 200,  83),
    "batiment_non_enduit":   ( 33, 150, 243),
    "batiment_enduit":       (255, 152,   0),
    "menuiserie_metallique": (156,  39, 176),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# =============================================================================
# AUTODÉCOUVERTE
# =============================================================================

def find_unified_model(output_base=None):
    if output_base is None:
        output_base = os.getenv("OUTPUT_DIR", "./runs/detect/train")

    info_path = os.path.join(output_base, "model_info_unified.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        candidate = info.get("best_model")
        if candidate and os.path.exists(candidate):
            print(f"   model_info_unified.json -> {candidate}")
            return candidate, info.get("classes", [])

    for base in [output_base, "./runs/detect/train"]:
        if not os.path.exists(base):
            continue
        dirs = sorted(
            [d for d in os.listdir(base)
             if os.path.isdir(os.path.join(base, d)) and d.startswith("ssd_unified_")],
            reverse=True,
        )
        for d in dirs:
            for fname in ["best_model.pth", "best.pth"]:
                candidate = os.path.join(base, d, fname)
                if os.path.exists(candidate):
                    print(f"   Trouve -> {candidate}")
                    return candidate, []

    raise FileNotFoundError(
        "\n[ERREUR] Modele unifie SSD introuvable.\n"
        "Lancez d'abord: python train_unified.py\n"
        "Ou: --model chemin/best_model.pth"
    )


# =============================================================================
# MODÈLE
# =============================================================================

def build_model(model_name, num_classes):
    if model_name == "ssd300_vgg16":
        model = ssd300_vgg16(weights=None)
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes)
    elif model_name == "ssdlite320_mobilenet_v3_large":
        model = ssdlite320_mobilenet_v3_large(weights=None)
        try:
            in_channels = [layer.in_channels
                           for layer in model.head.classification_head.module_list]
        except Exception:
            in_channels = [672, 480, 512, 256, 256, 128]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes)
    else:
        raise ValueError(f"Modele inconnu: {model_name}")
    return model


def load_model(model_path, device):
    print(f"   Chargement: {model_path}")
    ckpt        = torch.load(model_path, map_location=device)
    num_classes = ckpt.get("num_classes", 6)
    classes     = ckpt.get("classes", [])
    model_name  = ckpt.get("model_name", "ssd300_vgg16")
    image_size  = ckpt.get("image_size", 300)

    model = build_model(model_name, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    map50 = ckpt.get("map50", 0)
    print(f"   Epoch {epoch} | mAP@50 (val) = {map50:.4f} | {model_name} | {image_size}px")
    return model, classes, model_name, image_size


# =============================================================================
# INFÉRENCE
# =============================================================================

@torch.no_grad()
def predict(model, image_path, classes, image_size, device, threshold=SCORE_THRESHOLD):
    image   = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized = image.resize((image_size, image_size))
    scale_x = orig_w / image_size
    scale_y = orig_h / image_size

    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    t0      = time.time()
    outputs = model([tensor.to(device)])
    inf_time = time.time() - t0

    output     = outputs[0]
    all_boxes  = output["boxes"].cpu().numpy()
    all_labels = output["labels"].cpu().numpy()
    all_scores = output["scores"].cpu().numpy()

    kept = []
    for i, (lbl, score) in enumerate(zip(all_labels, all_scores)):
        cname = classes[int(lbl)] if int(lbl) < len(classes) else "unknown"
        thr   = CLASS_THRESHOLDS.get(cname, threshold)
        if score >= thr:
            kept.append(i)

    if kept:
        boxes  = all_boxes[kept]
        labels = all_labels[kept]
        scores = all_scores[kept]
    else:
        boxes  = np.zeros((0, 4), dtype=np.float32)
        labels = np.array([], dtype=np.int64)
        scores = np.array([], dtype=np.float32)

    if len(boxes) > 0:
        boxes[:, 0] *= scale_x; boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y; boxes[:, 3] *= scale_y

    class_names = [classes[int(l)] if int(l) < len(classes) else "unknown" for l in labels]

    return image, {
        "boxes":          boxes,
        "labels":         labels,
        "scores":         scores,
        "class_names":    class_names,
        "inference_time": inf_time,
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def visualize(image, preds, output_path=None, show=True, image_name=""):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(image)

    for box, class_name, score in zip(preds["boxes"], preds["class_names"], preds["scores"]):
        color = [c / 255 for c in COLORS.get(class_name, (128, 128, 128))]
        x1, y1, x2, y2 = box
        axes[1].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor="none"))
        axes[1].text(
            x1, max(0, y1 - 5), f"{class_name}\n{score:.2f}",
            fontsize=8, color="white",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.8))

    n  = len(preds["boxes"])
    ms = preds["inference_time"] * 1000
    axes[1].set_title(f"SSD Unifie | {n} objet(s) | {ms:.0f} ms")
    axes[1].axis("off")

    legend = [patches.Patch(facecolor=[c/255 for c in col], label=name)
              for name, col in COLORS.items()]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9)
    if image_name:
        fig.suptitle(image_name, fontsize=10, y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# RAPPORT
# =============================================================================

def generate_report(preds, image_name):
    by_class = {}
    for cname in set(preds["class_names"]):
        by_class[cname] = sum(1 for n in preds["class_names"] if n == cname)
    return {
        "image":             image_name,
        "model":             "SSD_unified",
        "timestamp":         datetime.now().isoformat(),
        "inference_time_ms": preds["inference_time"] * 1000,
        "total_objects":     len(preds["boxes"]),
        "by_class":          by_class,
        "detections": [
            {"id": i, "class": cls, "confidence": float(score), "bbox": box.tolist()}
            for i, (box, cls, score) in enumerate(
                zip(preds["boxes"], preds["class_names"], preds["scores"]))
        ],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference SSD unifie — toutes classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      default=None, help="Chemin modele (.pth)")
    parser.add_argument("--input",      default=os.getenv("DETECTION_INFERENCE_IMAGES_DIR", "./test"))
    parser.add_argument("--output",     default=os.getenv("PREDICTIONS_DIR", "./predictions_unified"))
    parser.add_argument("--threshold",  type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--display",    action="store_true")
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_base = os.getenv("OUTPUT_DIR", "./runs/detect/train")

    print("=" * 65)
    print("   INFERENCE SSD — Modele Unifie")
    print("=" * 65)
    print(f"   Device: {device}")

    if args.model:
        model_path = args.model
        classes    = []
    else:
        try:
            model_path, classes = find_unified_model(output_base)
        except FileNotFoundError as e:
            print(e); return

    if not os.path.exists(model_path):
        print(f"   Modele introuvable: {model_path}"); return

    model, model_classes, model_name, image_size = load_model(model_path, device)
    if model_classes:
        classes = model_classes
    if not classes:
        classes = ["__background__", "panneau_solaire", "batiment_peint",
                   "batiment_non_enduit", "batiment_enduit", "menuiserie_metallique"]

    print(f"\n   Classes: {classes}")

    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTS)
    else:
        images = [input_path]

    print(f"\n   {len(images)} image(s)\n")
    os.makedirs(args.output, exist_ok=True)

    reports     = []
    start_total = time.time()

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}")
        image, preds = predict(model, str(img_path), classes, image_size, device, args.threshold)
        out_img      = os.path.join(args.output, f"{img_path.stem}_unified.png")
        visualize(image, preds, out_img, show=args.display, image_name=img_path.name)
        report = generate_report(preds, img_path.name)
        reports.append(report)
        ms = preds["inference_time"] * 1000
        print(f"   {report['total_objects']} objet(s) | {ms:.0f} ms | {report['by_class']}")

    reports_path = os.path.join(args.output, "reports_unified.json")
    with open(reports_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_total

    if len(reports) > 1:
        total_objs = sum(r["total_objects"] for r in reports)
        avg_ms     = sum(r["inference_time_ms"] for r in reports) / len(reports)
        by_class   = {}
        for r in reports:
            for cls, cnt in r["by_class"].items():
                by_class[cls] = by_class.get(cls, 0) + cnt

        summary = {
            "model":            "SSD_unified",
            "timestamp":        datetime.now().isoformat(),
            "total_images":     len(reports),
            "total_time_s":     total_time,
            "avg_inference_ms": avg_ms,
            "total_objects":    total_objs,
            "by_class":         by_class,
        }
        with open(os.path.join(args.output, "summary_unified.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n   Resume:")
        print(f"   Images: {len(reports)}")
        print(f"   Objets: {total_objs}")
        print(f"   Temps:  {avg_ms:.0f} ms/image")
        for cls, cnt in by_class.items():
            if cnt > 0:
                print(f"   {cls}: {cnt}")

    print(f"\n   Resultats: {args.output}")


if __name__ == "__main__":
    main()
