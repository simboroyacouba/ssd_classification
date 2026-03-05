"""
Inférence SSD - Détection des toitures cadastrales
Configuration: .env + classes.yaml
"""

import os
import json
import yaml
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
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONFIG
# =============================================================================

def load_classes(yaml_path="classes.yaml"):
    if not os.path.exists(yaml_path):
        return ['__background__', 'panneau_solaire', 'batiment_peint',
                'batiment_non_enduit', 'batiment_enduit']
    with open(yaml_path, 'r', encoding='utf-8') as f:
        classes = yaml.safe_load(f).get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    return classes


def load_colors(yaml_path="classes.yaml"):
    default = {
        'panneau_solaire':     (255, 0,   0),
        'batiment_peint':      (0,   255, 0),
        'batiment_non_enduit': (0,   0,   255),
        'batiment_enduit':     (255, 165, 0),
    }
    if not os.path.exists(yaml_path):
        return default
    with open(yaml_path, 'r', encoding='utf-8') as f:
        colors = yaml.safe_load(f).get('colors', {})
    return {k: tuple(v) for k, v in colors.items()} if colors else default


CLASSES_FILE = os.getenv("CLASSES_FILE", "classes.yaml")
CLASSES      = load_classes(CLASSES_FILE)
COLORS       = load_colors(CLASSES_FILE)


def format_time(seconds):
    return f"{seconds*1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


# =============================================================================
# MODÈLE
# =============================================================================

def find_best_model():
    path = os.getenv("MODEL_PATH", None)
    if path and os.path.exists(path):
        return path
    runs_base = os.path.join("runs", "detect", "train")
    if os.path.exists(runs_base):
        for filter_fn in [lambda d: 'ssd' in d, lambda d: True]:
            subdirs = sorted(
                [d for d in os.listdir(runs_base)
                 if os.path.isdir(os.path.join(runs_base, d)) and filter_fn(d)],
                reverse=True
            )
            for subdir in subdirs:
                for fname in ["best_model.pth", "best.pth"]:
                    c = os.path.join(runs_base, subdir, fname)
                    if os.path.exists(c):
                        print(f"📁 Modèle trouvé: {c}"); return c
    for root, dirs, files in os.walk("output"):
        for fname in ["best_model.pth", "best.pth"]:
            if fname in files:
                found = os.path.join(root, fname)
                print(f"📁 Modèle trouvé: {found}"); return found
    return None


def build_model_skeleton(model_name, num_classes):
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
        raise ValueError(f"Modèle inconnu: {model_name}")
    return model


def load_model(model_path, device):
    checkpoint  = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', len(CLASSES))
    classes     = checkpoint.get('classes', CLASSES)
    model_name  = checkpoint.get('model_name', os.getenv("SSD_MODEL", "ssd300_vgg16"))
    image_size  = checkpoint.get('image_size', 300)

    model = build_model_skeleton(model_name, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, classes, model_name, image_size


# =============================================================================
# INFÉRENCE
# =============================================================================

@torch.no_grad()
def predict(model, image_path, classes, device, threshold=0.3, image_size=300):
    image  = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized = image.resize((image_size, image_size))
    tensor  = TF.to_tensor(resized)
    tensor  = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor  = tensor.unsqueeze(0).to(device)

    scale_x = orig_w / image_size
    scale_y = orig_h / image_size

    start   = time.time()
    outputs = model(tensor)
    inf_time = time.time() - start

    output = outputs[0]
    keep   = output['scores'] >= threshold
    boxes  = output['boxes'][keep].cpu().numpy()
    labels = output['labels'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()

    # Rescaler vers dimensions originales
    boxes[:, 0] *= scale_x; boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y; boxes[:, 3] *= scale_y

    class_names = [classes[int(l)] if int(l) < len(classes) else 'unknown' for l in labels]

    return image, {
        'boxes':          boxes,
        'labels':         labels,
        'scores':         scores,
        'class_names':    class_names,
        'inference_time': inf_time,
    }


def visualize(image, preds, model_name="SSD", output_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(image)
    for box, class_name, score in zip(preds['boxes'], preds['class_names'], preds['scores']):
        color = [c / 255 for c in COLORS.get(class_name, (128, 128, 128))]
        x1, y1, x2, y2 = box
        axes[1].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none'))
        axes[1].text(x1, y1-5, f"{class_name}\n{score:.2f}", fontsize=8, color='white',
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[1].set_title(f"{model_name}: {len(preds['boxes'])} objets | ⏱️ {format_time(preds['inference_time'])}")
    axes[1].axis('off')
    legend = [patches.Patch(facecolor=[c/255 for c in col], label=name)
              for name, col in COLORS.items()]
    fig.legend(handles=legend, loc='lower center', ncol=4)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.1)
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def generate_report(preds, image_name, classes):
    class_names_no_bg = [c for c in classes if c != '__background__']
    report = {
        'image': image_name, 'model': 'SSD',
        'timestamp': datetime.now().isoformat(),
        'inference_time_ms': preds['inference_time'] * 1000,
        'total_objects': len(preds['boxes']),
        'by_class': {c: {'count': 0} for c in class_names_no_bg},
        'detections': [],
    }
    for i, (box, class_name, score) in enumerate(
            zip(preds['boxes'], preds['class_names'], preds['scores'])):
        if class_name in report['by_class']:
            report['by_class'][class_name]['count'] += 1
        report['detections'].append({
            'id': i, 'class': class_name,
            'confidence': float(score), 'bbox': box.tolist()
        })
    return report


def generate_summary(reports, output_dir, total_time, classes):
    class_names_no_bg = [c for c in classes if c != '__background__']
    summary = {
        'model': 'SSD', 'timestamp': datetime.now().isoformat(),
        'total_images': len(reports), 'total_time_s': total_time,
        'avg_inference_ms': sum(r['inference_time_ms'] for r in reports) / len(reports) if reports else 0,
        'total_objects': sum(r['total_objects'] for r in reports),
        'by_class': {c: sum(r['by_class'].get(c, {}).get('count', 0) for r in reports)
                     for c in class_names_no_bg},
        'per_image': [{'image': r['image'], 'objects': r['total_objects'],
                       'time_ms': r['inference_time_ms']} for r in reports],
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"RÉSUMÉ SSD\n{'='*50}\n")
        f.write(f"Images: {summary['total_images']} | Temps: {total_time:.1f}s\n")
        f.write(f"Temps moyen: {summary['avg_inference_ms']:.1f} ms | Objets: {summary['total_objects']}\n")
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inférence SSD")
    parser.add_argument("--model",      default=None)
    parser.add_argument("--input",      default=os.getenv("DETECTION_INFERENCE_IMAGES_DIR", None))
    parser.add_argument("--output",     default=os.getenv("PREDICTIONS_DIR", "./predictions"))
    parser.add_argument("--threshold",  type=float, default=float(os.getenv("SCORE_THRESHOLD", "0.3")))
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    if args.model is None:
        args.model = find_best_model()
    if args.model is None or not os.path.exists(args.model):
        print(f"❌ Modèle non trouvé: {args.model}"); return

    model, classes, model_name, image_size = load_model(args.model, device)
    print(f"🧠 Modèle: {args.model} ({model_name}, {image_size}px)")
    print(f"   Classes: {classes}")

    if args.input is None:
        print("⚠️  Aucun dossier d'entrée spécifié.")
        print("   Utilisation: python inference.py --input /chemin/vers/images")
        print("   Ou définir DETECTION_INFERENCE_IMAGES_DIR dans .env")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Chemin introuvable: {input_path}"); return
    if input_path.is_dir():
        images = sorted([p for p in input_path.iterdir()
                         if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}])
        if not images:
            print(f"❌ Aucune image trouvée dans: {input_path}"); return
    else:
        images = [input_path]

    os.makedirs(args.output, exist_ok=True)
    print(f"🖼️  {len(images)} image(s)\n")

    reports     = []
    start_total = time.time()

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] 🔍 {img_path.name}")
        image, preds = predict(model, str(img_path), classes, device,
                               args.threshold, image_size)
        out_img = os.path.join(args.output, f"{img_path.stem}_ssd.png")
        visualize(image, preds, model_name, out_img, show=not args.no_display)
        report = generate_report(preds, img_path.name, classes)
        reports.append(report)
        print(f"   ✅ {report['total_objects']} objets | ⏱️ {report['inference_time_ms']:.1f} ms")

    with open(os.path.join(args.output, 'reports.json'), 'w') as f:
        json.dump(reports, f, indent=2)

    if len(images) > 1:
        summary = generate_summary(reports, args.output, time.time() - start_total, classes)
        print(f"\n📊 Résumé: {summary['total_objects']} objets | {summary['avg_inference_ms']:.1f} ms/image")

    print(f"\n📁 Résultats: {args.output}")


if __name__ == "__main__":
    main()
