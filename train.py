"""
Entraînement SSD pour détection des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: Chargées depuis classes.yaml
Configuration: Chargée depuis .env

Backbone: VGG16 ou MobileNetV3 via torchvision
Variantes disponibles:
  ssd300_vgg16                    -> 300px, backbone VGG16, précision maximale
  ssdlite320_mobilenet_v3_large   -> 320px, backbone MobileNetV3, léger/rapide
"""

import os
import json
import yaml
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    ssd300_vgg16, SSD300_VGG16_Weights,
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.ssd import SSDClassificationHead
from pycocotools.coco import COCO
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CHARGEMENT DES CLASSES
# =============================================================================

def load_classes(yaml_path="classes.yaml"):
    """
    Charger les classes depuis YAML.
    SSD (torchvision) indexe à partir de 1, 0 = background.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    classes = data.get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    print(f"📋 Classes chargées depuis {yaml_path}:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "images_dir":        os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
    "annotations_file":  os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"),
    "output_dir":        os.getenv("OUTPUT_DIR", "./output"),
    "classes_file":      os.getenv("CLASSES_FILE", "classes.yaml"),
    "classes":           None,

    # Modèle SSD
    "model_name":        os.getenv("SSD_MODEL", "ssd300_vgg16"),

    # Hyperparamètres
    "num_epochs":        int(os.getenv("NUM_EPOCHS", "50")),
    "batch_size":        int(os.getenv("BATCH_SIZE", "4")),
    "learning_rate":     float(os.getenv("LEARNING_RATE", "0.01")),
    "momentum":          float(os.getenv("MOMENTUM", "0.9")),
    "weight_decay":      float(os.getenv("WEIGHT_DECAY", "5e-4")),
    "train_split":       float(os.getenv("TRAIN_SPLIT", "0.70")),
    "val_split":         float(os.getenv("VAL_SPLIT", "0.20")),
    "test_split":        float(os.getenv("TEST_SPLIT", "0.10")),
    "save_every":        int(os.getenv("SAVE_EVERY", "5")),
    "score_threshold":   float(os.getenv("SCORE_THRESHOLD", "0.3")),
    "pretrained":        os.getenv("PRETRAINED", "true").lower() == "true",
    "grad_clip":         float(os.getenv("GRAD_CLIP", "1.0")),
}

# Tailles d'image fixes par variante SSD
SSD_IMAGE_SIZES = {
    "ssd300_vgg16":                   300,
    "ssdlite320_mobilenet_v3_large":  320,
}


# =============================================================================
# UTILITAIRES
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"


def stratified_split(coco, train_split, val_split, test_split, seed=42):
    np.random.seed(seed)
    all_image_ids = [img_id for img_id in coco.imgs if coco.getAnnIds(imgIds=img_id)]
    np.random.shuffle(all_image_ids)
    n_total = len(all_image_ids)
    n_train = int(n_total * train_split)
    n_val   = int(n_total * val_split)
    n_test  = n_total - n_train - n_val
    if n_test < 1 and n_total > 2:
        n_test  = max(1, int(n_total * 0.10))
        n_train = n_total - n_val - n_test
    print(f"\n   📊 Split des IMAGES (total: {n_total}):")
    print(f"      Train: {n_train} ({n_train/n_total*100:.1f}%)")
    print(f"      Val:   {n_val}   ({n_val/n_total*100:.1f}%)")
    print(f"      Test:  {n_test}  ({n_test/n_total*100:.1f}%)")
    train_ids = all_image_ids[:n_train]
    val_ids   = all_image_ids[n_train:n_train + n_val]
    test_ids  = all_image_ids[n_train + n_val:]
    stats = {'train': {}, 'val': {}, 'test': {}}
    for cat_id in coco.getCatIds():
        stats['train'][cat_id] = 0
        stats['val'][cat_id]   = 0
        stats['test'][cat_id]  = 0
    for img_id in train_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['train'][ann['category_id']] += 1
    for img_id in val_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['val'][ann['category_id']] += 1
    for img_id in test_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['test'][ann['category_id']] += 1
    return train_ids, val_ids, test_ids, stats


def print_split_stats(coco, stats):
    print("\n   📊 Distribution des classes (split 70/20/10):")
    print(f"   {'Classe':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"   {'-'*70}")
    for cat_id in coco.getCatIds():
        name  = coco.cats[cat_id]['name']
        train = stats['train'].get(cat_id, 0)
        val   = stats['val'].get(cat_id, 0)
        test  = stats['test'].get(cat_id, 0)
        total = train + val + test
        ok    = "⚠️" if val == 0 or test == 0 else "✅"
        print(f"   {name:<30} {train:>8} {val:>8} {test:>8} {total:>8} {ok}")
    print(f"   {'-'*70}")


# =============================================================================
# DATASET PYTORCH
# =============================================================================

class SSDDataset(Dataset):
    """
    Dataset COCO pour SSD torchvision.
    SSD attend les boxes au format [x1, y1, x2, y2] en pixels.
    Les labels commencent à 1 (0 = background).
    """

    def __init__(self, images_dir, annotations_file, image_ids,
                 cat_mapping, image_size=300, augment=False):
        self.images_dir  = images_dir
        self.coco        = COCO(annotations_file)
        self.image_ids   = image_ids
        self.cat_mapping = cat_mapping
        self.image_size  = image_size
        self.augment     = augment

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size))
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        # Annotations SSD: [x1, y1, x2, y2]
        anns   = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes  = []
        labels = []

        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            class_id = self.cat_mapping.get(ann['category_id'])
            if class_id is None:
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            x1 = max(0.0, x * scale_x)
            y1 = max(0.0, y * scale_y)
            x2 = min(float(self.image_size), (x + w) * scale_x)
            y2 = min(float(self.image_size), (y + h) * scale_y)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        target = {
            'boxes':    torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4), dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),   dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =============================================================================
# MODÈLE
# =============================================================================

def build_model(model_name, num_classes, pretrained=True):
    """
    Construire SSD avec torchvision.
    num_classes: nombre de classes AVEC background.
    """
    if model_name == "ssd300_vgg16":
        weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
        model   = ssd300_vgg16(weights=weights)
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

    elif model_name == "ssdlite320_mobilenet_v3_large":
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model   = ssdlite320_mobilenet_v3_large(weights=weights)
        # Récupérer les in_channels réels depuis la tête existante
        try:
            in_channels = [
                layer.in_channels
                for layer in model.head.classification_head.module_list
            ]
        except Exception:
            in_channels = [672, 480, 512, 256, 256, 128]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Modèle inconnu: {model_name}. "
                         f"Choisir: ssd300_vgg16 | ssdlite320_mobilenet_v3_large")

    return model


# =============================================================================
# MÉTRIQUES
# =============================================================================

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0


def compute_map(predictions, ground_truths, class_names, iou_threshold=0.5):
    aps = {}
    for cls_id, name in enumerate(class_names, start=1):
        tps, fps, scores_list = [], [], []
        n_gt = sum((gt['labels'] == cls_id).sum().item() for gt in ground_truths)
        if n_gt == 0:
            continue
        for pred, gt in zip(predictions, ground_truths):
            mask_p   = pred['labels'] == cls_id
            mask_g   = gt['labels']   == cls_id
            p_boxes  = pred['boxes'][mask_p].cpu().numpy()
            p_scores = pred['scores'][mask_p].cpu().numpy()
            g_boxes  = gt['boxes'][mask_g].cpu().numpy()
            matched  = set()
            for i in np.argsort(-p_scores):
                scores_list.append(p_scores[i])
                if len(g_boxes) == 0:
                    tps.append(0); fps.append(1); continue
                ious   = [calculate_iou(p_boxes[i], g) for g in g_boxes]
                best_j = int(np.argmax(ious))
                if ious[best_j] >= iou_threshold and best_j not in matched:
                    matched.add(best_j); tps.append(1); fps.append(0)
                else:
                    tps.append(0); fps.append(1)
        if not scores_list:
            aps[name] = 0.0; continue
        order  = np.argsort(-np.array(scores_list))
        tp_cum = np.cumsum(np.array(tps)[order])
        fp_cum = np.cumsum(np.array(fps)[order])
        prec   = tp_cum / (tp_cum + fp_cum + 1e-10)
        rec    = tp_cum / (n_gt + 1e-10)
        ap = sum(np.max(prec[rec >= t]) if (rec >= t).any() else 0
                 for t in np.arange(0, 1.1, 0.1)) / 11
        aps[name] = float(ap)
    return aps


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_one_epoch(model, optimizer, dataloader, device, grad_clip=1.0):
    model.train()
    total_loss = 0; total_cls = 0; total_bbox = 0; num_batches = 0

    for images, targets in dataloader:
        images  = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()
                    if isinstance(v, torch.Tensor)} for t in targets]
        if all(len(t['boxes']) == 0 for t in targets):
            continue
        try:
            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss  += losses.item()
            total_cls   += loss_dict.get('classification', torch.tensor(0)).item()
            total_bbox  += loss_dict.get('bbox_regression', torch.tensor(0)).item()
            num_batches += 1
        except Exception as e:
            print(f"   ⚠️ Erreur batch: {e}"); continue

    n = max(num_batches, 1)
    return total_loss / n, {'cls_loss': total_cls / n, 'bbox_loss': total_bbox / n}


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, class_names, score_threshold=0.3):
    model.eval()
    all_preds = []; all_gts = []
    for images, targets in dataloader:
        images  = list(img.to(device) for img in images)
        outputs = model(images)
        for output, target in zip(outputs, targets):
            keep = output['scores'] >= score_threshold
            all_preds.append({
                'boxes':  output['boxes'][keep].cpu(),
                'labels': output['labels'][keep].cpu(),
                'scores': output['scores'][keep].cpu(),
            })
            all_gts.append({'boxes': target['boxes'], 'labels': target['labels']})
    aps   = compute_map(all_preds, all_gts, class_names, iou_threshold=0.5)
    map50 = float(np.mean(list(aps.values()))) if aps else 0.0
    return map50, aps


# =============================================================================
# MAIN
# =============================================================================

def train_ssd():
    CONFIG["classes"] = load_classes(CONFIG["classes_file"])
    num_classes       = len(CONFIG["classes"])  # inclut __background__
    class_names_no_bg = [c for c in CONFIG["classes"] if c != '__background__']
    image_size        = SSD_IMAGE_SIZES.get(CONFIG["model_name"], 300)

    print("=" * 70)
    print(f"   SSD ({CONFIG['model_name']}) - Détection des Toitures")
    print("=" * 70)
    print(f"\n📋 CONFIG (.env)")
    print(f"   Images:      {CONFIG['images_dir']}")
    print(f"   Annotations: {CONFIG['annotations_file']}")
    print(f"   Modèle:      {CONFIG['model_name']}")
    print(f"   Classes:     {num_classes} (avec __background__)")
    print(f"   Epochs:      {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['learning_rate']}")
    print(f"   Image size:  {image_size}px (fixe pour SSD)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device:      {device}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir   = os.path.join("runs", "detect", "train", f"ssd_{timestamp}")
    weights_dir = os.path.join(train_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Split
    coco        = COCO(CONFIG["annotations_file"])
    cat_ids     = coco.getCatIds()
    cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
    train_ids, val_ids, test_ids, split_stats = stratified_split(
        coco, CONFIG["train_split"], CONFIG["val_split"], CONFIG["test_split"], seed=42
    )
    print_split_stats(coco, split_stats)

    test_info = {
        'test_image_ids':   test_ids,
        'cat_mapping':      {str(k): v for k, v in cat_mapping.items()},
        'images_dir':       os.path.abspath(CONFIG["images_dir"]),
        'annotations_file': os.path.abspath(CONFIG["annotations_file"]),
        'num_test_images':  len(test_ids),
        'classes':          CONFIG["classes"],
        'model_name':       CONFIG["model_name"],
        'image_size':       image_size,
    }
    with open(os.path.join(train_dir, "test_info.json"), 'w') as f:
        json.dump(test_info, f, indent=2)

    # Datasets
    train_dataset = SSDDataset(CONFIG["images_dir"], CONFIG["annotations_file"],
                               train_ids, cat_mapping, image_size, augment=True)
    val_dataset   = SSDDataset(CONFIG["images_dir"], CONFIG["annotations_file"],
                               val_ids,   cat_mapping, image_size)
    train_loader  = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                               shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=1,
                               shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"\n   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_ids)} images")

    # Modèle
    print(f"\n🧠 Chargement {CONFIG['model_name']} (pretrained={CONFIG['pretrained']})...")
    model     = build_model(CONFIG["model_name"], num_classes, CONFIG["pretrained"])
    model.to(device)

    params       = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.SGD(params, lr=CONFIG["learning_rate"],
                                   momentum=CONFIG["momentum"],
                                   weight_decay=CONFIG["weight_decay"])
    # MultiStepLR: diviser le LR par 10 à 60% et 85% des epochs
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(CONFIG["num_epochs"] * 0.6), int(CONFIG["num_epochs"] * 0.85)],
        gamma=0.1
    )

    print("\n" + "=" * 70)
    print(f"   🚀 ENTRAÎNEMENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    history    = {'train_loss': [], 'cls_loss': [], 'bbox_loss': [], 'val_map50': [], 'lr': []}
    best_map50 = 0.0
    start_time = time.time()

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        epoch_start = time.time()
        print(f"\n📅 Epoch [{epoch}/{CONFIG['num_epochs']}]")

        avg_loss, loss_parts = train_one_epoch(model, optimizer, train_loader,
                                               device, CONFIG["grad_clip"])
        val_map50, val_aps   = evaluate_epoch(model, val_loader, device,
                                              class_names_no_bg, CONFIG["score_threshold"])
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_loss)
        history['cls_loss'].append(loss_parts['cls_loss'])
        history['bbox_loss'].append(loss_parts['bbox_loss'])
        history['val_map50'].append(val_map50)
        history['lr'].append(current_lr)

        print(f"   Loss: {avg_loss:.4f} (cls={loss_parts['cls_loss']:.3f} box={loss_parts['bbox_loss']:.3f})"
              f" | mAP@50: {val_map50:.4f} | LR: {current_lr:.2e} | ⏱️ {format_time(time.time()-epoch_start)}")
        if val_aps:
            for name, ap in val_aps.items():
                print(f"      {name:<30} AP={ap:.3f}")

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50': best_map50, 'num_classes': num_classes,
                'classes': CONFIG["classes"], 'cat_mapping': cat_mapping,
                'model_name': CONFIG["model_name"], 'image_size': image_size,
            }, os.path.join(weights_dir, "best.pth"))
            print(f"   💾 Meilleur modèle sauvegardé (mAP@50: {best_map50:.4f})")

        if epoch % CONFIG["save_every"] == 0 or epoch == CONFIG["num_epochs"]:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50': val_map50, 'num_classes': num_classes,
                'classes': CONFIG["classes"], 'cat_mapping': cat_mapping,
                'model_name': CONFIG["model_name"], 'image_size': image_size,
            }, os.path.join(weights_dir, "last.pth"))

    total_time = time.time() - start_time

    print("\n📦 Copie des modèles...")
    for src, dst in [("best.pth", "best_model.pth"), ("last.pth", "final_model.pth")]:
        src_path = os.path.join(weights_dir, src)
        dst_path = os.path.join(train_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"   ✅ {dst} ({os.path.getsize(dst_path)/1024/1024:.1f} MB)")

    history['time_stats'] = {
        'total_time_formatted':     format_time(total_time),
        'avg_epoch_time_formatted': format_time(total_time / CONFIG["num_epochs"]),
    }
    history['config']     = CONFIG
    history['best_map50'] = best_map50
    with open(os.path.join(train_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    if history['train_loss']:
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Total')
        axes[0].plot(epochs, history['cls_loss'],  'r--', label='Cls')
        axes[0].plot(epochs, history['bbox_loss'], 'g--', label='BBox')
        axes[0].set_title('Loss (train)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, history['val_map50'], 'g-')
        axes[1].set_title('mAP@50 (validation)'); axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
        axes[2].plot(epochs, history['lr'], color='orange')
        axes[2].set_title('Learning Rate'); axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, 'training_curves.png'), dpi=150)
        plt.close()

    with open(os.path.join(train_dir, "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"SSD ({CONFIG['model_name']}) - Rapport\n{'='*50}\n\n")
        f.write(f"Modèle:          {CONFIG['model_name']}\n")
        f.write(f"Classes:         {CONFIG['classes']}\n")
        f.write(f"Epochs:          {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']}\n\n")
        f.write(f"Meilleur mAP@50: {best_map50:.4f}\n")
        f.write(f"Temps total:     {format_time(total_time)}\n")
        f.write(f"Chemin:          {train_dir}\n")

    print("\n" + "=" * 70)
    print("   🎉 TERMINÉ")
    print("=" * 70)
    print(f"   Meilleur mAP@50: {best_map50:.4f} ({best_map50*100:.2f}%)")
    print(f"   ⏱️  Temps: {format_time(total_time)}")
    print(f"   📁 Modèles: {train_dir}")
    print("=" * 70)

    return model, history


if __name__ == "__main__":
    train_ssd()
