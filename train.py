"""
Entraînement SSD pour détection des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: Chargées depuis classes.yaml

Architecture duale :
  - Mode nadir   : classe panneau_solaire   (instances_nadir.json)
  - Mode oblique : classes batiment_*        (instances_oblique.json)
  - Mode dual    : nadir puis oblique séquentiellement

Backbone: VGG16 ou MobileNetV3 via torchvision
Variantes disponibles:
  ssd300_vgg16                    -> 300px, backbone VGG16, précision maximale
  ssdlite320_mobilenet_v3_large   -> 320px, backbone MobileNetV3, léger/rapide

Usage :
  python train.py --mode simple
  python train.py --mode attention --cbam-reduction 16
  python train.py --mode optimize --n-trials 30
  python train.py --mode dual --aug panneau_solaire:3
  python train.py --mode dual --model-name ssdlite320_mobilenet_v3_large
"""

import os
import copy
import json
import yaml
import shutil
import argparse
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
# CLASSES PAR MODE
# =============================================================================

MODE_CLASSES = {
    "nadir":   ["panneau_solaire"],
    "oblique": [
        "batiment_peint", "batiment_non_enduit",
        "batiment_enduit", "menuiserie_metallique",
    ],
}

# Tailles d'image fixes par variante SSD
SSD_IMAGE_SIZES = {
    "ssd300_vgg16":                   300,
    "ssdlite320_mobilenet_v3_large":  320,
}

OPTUNA_CONFIG = {
    "n_trials":        30,
    "n_epochs_per_trial": 5,
    "study_name":      "ssd_cadastral",
    "output_dir":      "./optuna_output",
}


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config(args):
    base_annotations = os.getenv(
        "DETECTION_DATASET_ANNOTATIONS_FILE",
        "../dataset1/annotations/instances_default.json",
    )
    ann_dir = os.path.dirname(os.path.abspath(base_annotations))

    annotations_file = args.annotations_file or base_annotations
    base_output      = os.getenv("OUTPUT_DIR", "./output")
    output_dir       = args.output_dir or base_output
    classes_file     = args.classes_file or os.getenv("CLASSES_FILE", "classes.yaml")

    return {
        "mode":             args.mode,
        "images_dir":       args.images_dir or os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
        "annotations_file": annotations_file,
        "ann_dir":          ann_dir,
        "output_dir":       output_dir,
        "classes_file":     classes_file,
        "model_name":       args.model_name or os.getenv("SSD_MODEL", "ssd300_vgg16"),
        "num_epochs":       int(os.getenv("NUM_EPOCHS", "50")),
        "batch_size":       int(os.getenv("BATCH_SIZE", "4")),
        "learning_rate":    float(os.getenv("LEARNING_RATE", "0.01")),
        "momentum":         float(os.getenv("MOMENTUM", "0.9")),
        "weight_decay":     float(os.getenv("WEIGHT_DECAY", "5e-4")),
        "train_split":      float(os.getenv("TRAIN_SPLIT", "0.70")),
        "val_split":        float(os.getenv("VAL_SPLIT", "0.20")),
        "test_split":       float(os.getenv("TEST_SPLIT", "0.10")),
        "save_every":       int(os.getenv("SAVE_EVERY", "5")),
        "score_threshold":  float(os.getenv("SCORE_THRESHOLD", "0.3")),
        "pretrained":       os.getenv("PRETRAINED", "true").lower() == "true",
        "grad_clip":        float(os.getenv("GRAD_CLIP", "1.0")),
        "classes":          None,
    }


def _build_sub_config(base_config, mode_label):
    """Construit un sous-config nadir ou oblique depuis le config de base."""
    ann_dir = base_config["ann_dir"]
    cfg = copy.deepcopy(base_config)
    cfg["mode"] = mode_label
    if mode_label == "nadir":
        cfg["annotations_file"] = os.path.join(ann_dir, "instances_nadir.json")
        cfg["output_dir"]       = os.path.join(base_config["output_dir"], "nadir")
    elif mode_label == "oblique":
        cfg["annotations_file"] = os.path.join(ann_dir, "instances_oblique.json")
        cfg["output_dir"]       = os.path.join(base_config["output_dir"], "oblique")
    return cfg


# =============================================================================
# CHARGEMENT DES CLASSES
# =============================================================================

def load_classes(yaml_path, mode_classes=None):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    all_classes = [c for c in data.get('classes', []) if c != '__background__']
    if mode_classes is not None:
        filtered = [c for c in all_classes if c in mode_classes]
    else:
        filtered = all_classes
    classes = ['__background__'] + filtered
    print("Classes chargees:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


# =============================================================================
# PARSE AUG COEFFICIENTS
# =============================================================================

def parse_aug_coeffs(aug_args, classes):
    """
    Analyse --aug CLASS:COEFF avec correspondance partielle insensible à la casse.
    Retourne un dict {class_name: coeff} pour les classes connues.
    Exemple: ['batiment_peint:3', 'solaire:2']
    """
    coeffs = {}
    for entry in (aug_args or []):
        if ':' not in entry:
            print(f"   [WARN] --aug '{entry}' ignoré (format attendu CLASS:COEFF)")
            continue
        raw_name, raw_coeff = entry.rsplit(':', 1)
        try:
            coeff = int(raw_coeff)
        except ValueError:
            print(f"   [WARN] --aug '{entry}' ignoré (coefficient non entier)")
            continue
        if coeff < 1:
            print(f"   [WARN] --aug '{entry}' ignoré (coefficient < 1)")
            continue
        raw_lower = raw_name.lower()
        matched = [c for c in classes if raw_lower in c.lower()]
        if not matched:
            print(f"   [WARN] --aug '{raw_name}' ne correspond à aucune classe connue")
            continue
        if len(matched) > 1:
            print(f"   [WARN] --aug '{raw_name}' ambigu ({matched}), ignoré")
            continue
        coeffs[matched[0]] = coeff
    return coeffs


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
    print(f"\n   Split des IMAGES (total: {n_total}):")
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
    print("\n   Distribution des classes:")
    print(f"   {'Classe':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"   {'-'*70}")
    for cat_id in coco.getCatIds():
        name  = coco.cats[cat_id]['name']
        train = stats['train'].get(cat_id, 0)
        val   = stats['val'].get(cat_id, 0)
        test  = stats['test'].get(cat_id, 0)
        total = train + val + test
        ok    = " !" if val == 0 or test == 0 else ""
        print(f"   {name:<30} {train:>8} {val:>8} {test:>8} {total:>8}{ok}")
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

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                boxes = [[self.image_size - x2, y1, self.image_size - x1, y2]
                         for x1, y1, x2, y2 in boxes]
            if random.random() < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            if random.random() < 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
            if random.random() < 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        target = {
            'boxes':    torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4), dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),   dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =============================================================================
# MODULES D'ATTENTION (CBAM)
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.fc[-1].weight)

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


class AttentionBackbone(nn.Module):
    """Applique CBAM sur chaque feature map retournée par le backbone SSD."""
    def __init__(self, backbone, channels, cbam_reduction=16, cbam_kernel_size=7):
        super().__init__()
        self.backbone   = backbone
        self.attentions = nn.ModuleList([
            CBAM(c, cbam_reduction, cbam_kernel_size) for c in channels
        ])

    def forward(self, x):
        features = self.backbone(x)
        result = {}
        for i, (k, v) in enumerate(features.items()):
            attn = self.attentions[i] if i < len(self.attentions) else nn.Identity()
            result[k] = attn(v)
        return result


def _probe_backbone_channels(backbone, image_size):
    """Détecte les dimensions des feature maps via un forward factice."""
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size)
        feats = backbone(dummy)
    return [f.shape[1] for f in feats.values()]


# =============================================================================
# MODÈLE
# =============================================================================

def _rebuild_head(model, model_name, num_classes):
    """Recrée la tête de classification pour num_classes."""
    if model_name == "ssd300_vgg16":
        in_channels = [512, 1024, 512, 256, 256, 256]
    else:
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


def get_model_simple(model_name, num_classes, pretrained=True):
    """SSD standard sans attention."""
    if model_name == "ssd300_vgg16":
        weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
        model   = ssd300_vgg16(weights=weights)
    elif model_name == "ssdlite320_mobilenet_v3_large":
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model   = ssdlite320_mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Modèle inconnu: {model_name}. "
                         f"Choisir: ssd300_vgg16 | ssdlite320_mobilenet_v3_large")
    _rebuild_head(model, model_name, num_classes)
    return model


def get_model_attention(model_name, num_classes, pretrained=True,
                        cbam_reduction=16, cbam_kernel_size=7):
    """SSD avec AttentionBackbone (CBAM sur chaque feature map)."""
    image_size = SSD_IMAGE_SIZES.get(model_name, 300)
    model      = get_model_simple(model_name, num_classes, pretrained)
    channels   = _probe_backbone_channels(model.backbone, image_size)
    model.backbone = AttentionBackbone(model.backbone, channels,
                                       cbam_reduction, cbam_kernel_size)
    print(f"   AttentionBackbone (CBAM r={cbam_reduction}, k={cbam_kernel_size}) "
          f"sur {len(channels)} feature maps : {channels}")
    return model


# =============================================================================
# PONDÉRATION PAR AUG_COEFFS (WeightedRandomSampler)
# =============================================================================

def compute_sample_weights(coco, image_ids, cat_mapping, classes, aug_coeffs):
    """
    Calcule les poids d'échantillonnage à partir des coefficients --aug.
    Les images contenant des classes avec coeff élevé sont sur-échantillonnées.
    """
    class_idx_to_coeff = {
        classes.index(name): coeff
        for name, coeff in aug_coeffs.items()
        if name in classes
    }
    if not class_idx_to_coeff:
        return None

    sample_weights = []
    for img_id in image_ids:
        anns  = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        w     = 1.0
        for ann in anns:
            cls_idx = cat_mapping.get(ann['category_id'])
            if cls_idx in class_idx_to_coeff:
                w = max(w, float(class_idx_to_coeff[cls_idx]))
        sample_weights.append(w)

    print("   Poids d'échantillonnage (aug_coeffs):")
    for name, coeff in aug_coeffs.items():
        if name in classes:
            print(f"      {name:<30} coeff={coeff}")

    return torch.tensor(sample_weights, dtype=torch.float32)


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
            print(f"   Erreur batch: {e}"); continue

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
# BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# =============================================================================

def _train_single(config, aug_coeffs, mode_label, training_mode,
                  cbam_reduction=16, cbam_kernel_size=7):
    """
    Lance un entraînement complet SSD sur config.
    training_mode: 'simple' | 'attention'
    """
    classes           = config["classes"]
    num_classes       = len(classes)
    class_names_no_bg = [c for c in classes if c != '__background__']
    model_name        = config["model_name"]
    image_size        = SSD_IMAGE_SIZES.get(model_name, 300)

    print("=" * 70)
    print(f"   SSD ({model_name}) — Mode : {mode_label.upper()} [{training_mode}]")
    print("=" * 70)
    print(f"\n   Images:      {config['images_dir']}")
    print(f"   Annotations: {config['annotations_file']}")
    print(f"   Classes:     {num_classes} (avec __background__): {class_names_no_bg}")
    print(f"   Epochs:      {config['num_epochs']} | Batch: {config['batch_size']} | LR: {config['learning_rate']}")
    print(f"   Image size:  {image_size}px")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device:      {device}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir   = os.path.join(config["output_dir"], f"ssd_{mode_label}_{timestamp}")
    weights_dir = os.path.join(train_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    coco      = COCO(config["annotations_file"])
    cat_ids   = coco.getCatIds()
    coco_cats = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}
    cat_mapping = {}
    for cat_id, cat_name in coco_cats.items():
        if cat_name in classes:
            cat_mapping[cat_id] = classes.index(cat_name)
        else:
            print(f"   Catégorie COCO ignorée: '{cat_name}' (id={cat_id})")

    print(f"\n   cat_mapping: {[(coco_cats[k], v) for k, v in cat_mapping.items()]}")

    train_ids, val_ids, test_ids, split_stats = stratified_split(
        coco, config["train_split"], config["val_split"], config["test_split"], seed=42
    )
    print_split_stats(coco, split_stats)

    test_info_path = os.path.join(train_dir, "test_info.json")
    with open(test_info_path, 'w') as f:
        json.dump({
            'test_image_ids':   test_ids,
            'cat_mapping':      {str(k): v for k, v in cat_mapping.items()},
            'images_dir':       os.path.abspath(config["images_dir"]),
            'annotations_file': os.path.abspath(config["annotations_file"]),
            'num_test_images':  len(test_ids),
            'classes':          classes,
            'model_name':       model_name,
            'image_size':       image_size,
            'mode':             mode_label,
        }, f, indent=2)

    train_dataset = SSDDataset(config["images_dir"], config["annotations_file"],
                               train_ids, cat_mapping, image_size, augment=True)
    val_dataset   = SSDDataset(config["images_dir"], config["annotations_file"],
                               val_ids,   cat_mapping, image_size)

    # WeightedRandomSampler si aug_coeffs fournis
    sample_weights = compute_sample_weights(coco, train_ids, cat_mapping,
                                            classes, aug_coeffs)
    if sample_weights is not None:
        sampler      = WeightedRandomSampler(sample_weights,
                                             num_samples=len(sample_weights),
                                             replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  sampler=sampler, collate_fn=collate_fn, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  shuffle=True, collate_fn=collate_fn, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"\n   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_ids)}")

    print(f"\n   Chargement {model_name} (pretrained={config['pretrained']})...")
    if training_mode == "attention":
        model = get_model_attention(model_name, num_classes, config["pretrained"],
                                    cbam_reduction, cbam_kernel_size)
    else:
        model = get_model_simple(model_name, num_classes, config["pretrained"])
    model.to(device)

    params       = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.SGD(params, lr=config["learning_rate"],
                                   momentum=config["momentum"],
                                   weight_decay=config["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(config["num_epochs"] * 0.6), int(config["num_epochs"] * 0.85)],
        gamma=0.1
    )

    print("\n" + "=" * 70)
    print(f"   ENTRAÎNEMENT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    history    = {'train_loss': [], 'cls_loss': [], 'bbox_loss': [], 'val_map50': [], 'lr': []}
    best_map50 = 0.0
    best_path  = os.path.join(weights_dir, "best.pth")
    start_time = time.time()

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()
        print(f"\nEpoch [{epoch}/{config['num_epochs']}]")

        avg_loss, loss_parts = train_one_epoch(model, optimizer, train_loader,
                                               device, config["grad_clip"])
        val_map50, val_aps   = evaluate_epoch(model, val_loader, device,
                                              class_names_no_bg, config["score_threshold"])
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_loss)
        history['cls_loss'].append(loss_parts['cls_loss'])
        history['bbox_loss'].append(loss_parts['bbox_loss'])
        history['val_map50'].append(val_map50)
        history['lr'].append(current_lr)

        print(f"   Loss: {avg_loss:.4f} (cls={loss_parts['cls_loss']:.3f} box={loss_parts['bbox_loss']:.3f})"
              f" | mAP@50: {val_map50:.4f} | LR: {current_lr:.2e} | {format_time(time.time()-epoch_start)}")
        for name, ap in (val_aps or {}).items():
            print(f"      {name:<30} AP={ap:.3f}")

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50': best_map50, 'num_classes': num_classes,
                'classes': classes, 'cat_mapping': cat_mapping,
                'model_name': model_name, 'image_size': image_size,
                'mode': mode_label, 'training_mode': training_mode,
            }, best_path)
            print(f"   Meilleur modèle sauvegardé (mAP@50: {best_map50:.4f})")

        if epoch % config["save_every"] == 0 or epoch == config["num_epochs"]:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50': val_map50, 'num_classes': num_classes,
                'classes': classes, 'cat_mapping': cat_mapping,
                'model_name': model_name, 'image_size': image_size,
                'mode': mode_label, 'training_mode': training_mode,
            }, os.path.join(weights_dir, "last.pth"))

    total_time = time.time() - start_time

    best_model_path = os.path.join(train_dir, "best_model.pth")
    for src, dst in [("best.pth", "best_model.pth"), ("last.pth", "final_model.pth")]:
        src_path = os.path.join(weights_dir, src)
        dst_path = os.path.join(train_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"   {dst} ({os.path.getsize(dst_path)/1024/1024:.1f} MB)")

    model_info = {
        "model":          f"SSD_{mode_label}",
        "model_name":     model_name,
        "mode":           mode_label,
        "training_mode":  training_mode,
        "best_model":     os.path.abspath(best_model_path),
        "train_dir":      os.path.abspath(train_dir),
        "test_info":      os.path.abspath(test_info_path),
        "classes":        classes,
        "num_classes":    num_classes,
        "image_size":     image_size,
        "best_map50":     best_map50,
        "timestamp":      timestamp,
    }
    info_path = os.path.join(config["output_dir"], f"model_info_{mode_label}.json")
    os.makedirs(config["output_dir"], exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"\n   model_info_{mode_label}.json -> {info_path}")

    history['best_map50'] = best_map50
    history['config']     = {k: str(v) for k, v in config.items()}
    with open(os.path.join(train_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    if history['train_loss']:
        epochs_r = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(epochs_r, history['train_loss'], 'b-', label='Total')
        axes[0].plot(epochs_r, history['cls_loss'],   'r--', label='Cls')
        axes[0].plot(epochs_r, history['bbox_loss'],  'g--', label='BBox')
        axes[0].set_title('Loss (train)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs_r, history['val_map50'], 'g-')
        axes[1].set_title('mAP@50 (validation)'); axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
        axes[2].plot(epochs_r, history['lr'], color='orange')
        axes[2].set_title('Learning Rate'); axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, 'training_curves.png'), dpi=150)
        plt.close()

    with open(os.path.join(train_dir, "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"SSD ({model_name}) — Mode {mode_label} [{training_mode}]\n{'='*50}\n\n")
        f.write(f"Mode:            {mode_label}\n")
        f.write(f"Training mode:   {training_mode}\n")
        f.write(f"Modèle:          {model_name}\n")
        f.write(f"Classes:         {classes}\n")
        f.write(f"Epochs:          {config['num_epochs']} | Batch: {config['batch_size']}\n\n")
        f.write(f"Meilleur mAP@50: {best_map50:.4f}\n")
        f.write(f"Temps total:     {format_time(total_time)}\n")
        f.write(f"Chemin:          {train_dir}\n")

    print("\n" + "=" * 70)
    print(f"   TERMINÉ — Mode {mode_label.upper()} [{training_mode}]")
    print("=" * 70)
    print(f"   Meilleur mAP@50: {best_map50:.4f} ({best_map50*100:.2f}%)")
    print(f"   Temps: {format_time(total_time)}")
    print(f"   Modèle: {best_model_path}")
    print("=" * 70)

    return model, history


# =============================================================================
# OPTIMISATION OPTUNA
# =============================================================================

def _run_optimization(config, aug_coeffs, mode_label, cbam_reduction, cbam_kernel_size,
                      n_trials, n_epochs_per_trial):
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    classes           = config["classes"]
    num_classes       = len(classes)
    class_names_no_bg = [c for c in classes if c != '__background__']
    model_name        = config["model_name"]
    image_size        = SSD_IMAGE_SIZES.get(model_name, 300)
    device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coco      = COCO(config["annotations_file"])
    coco_cats = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    cat_mapping = {
        cat_id: classes.index(cat_name)
        for cat_id, cat_name in coco_cats.items()
        if cat_name in classes
    }

    train_ids, val_ids, _, _ = stratified_split(
        coco, config["train_split"], config["val_split"], config["test_split"], seed=42
    )

    sample_weights = compute_sample_weights(coco, train_ids, cat_mapping,
                                            classes, aug_coeffs)

    # Charge le modèle une seule fois et copie l'état initial
    print(f"   Chargement modèle de référence pour Optuna...")
    ref_model     = get_model_simple(model_name, num_classes, config["pretrained"])
    initial_state = copy.deepcopy(ref_model.state_dict())
    del ref_model
    gc.collect()

    def objective(trial):
        lr           = trial.suggest_float("lr",           1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        momentum     = trial.suggest_float("momentum",     0.7,  0.99)
        use_cbam     = trial.suggest_categorical("use_cbam", [True, False])

        if use_cbam:
            reduction  = trial.suggest_categorical("cbam_reduction",  [8, 16, 32])
            kernel_size = trial.suggest_categorical("cbam_kernel_size", [3, 5, 7])
            model = get_model_attention(model_name, num_classes, pretrained=False,
                                        cbam_reduction=reduction,
                                        cbam_kernel_size=kernel_size)
        else:
            model = get_model_simple(model_name, num_classes, pretrained=False)
            model.load_state_dict(copy.deepcopy(initial_state), strict=False)
        model.to(device)

        train_dataset = SSDDataset(config["images_dir"], config["annotations_file"],
                                   train_ids, cat_mapping, image_size, augment=True)
        val_dataset   = SSDDataset(config["images_dir"], config["annotations_file"],
                                   val_ids,   cat_mapping, image_size)

        if sample_weights is not None:
            sampler      = WeightedRandomSampler(sample_weights,
                                                 num_samples=len(sample_weights),
                                                 replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                      sampler=sampler, collate_fn=collate_fn, num_workers=0)
        else:
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                      shuffle=True, collate_fn=collate_fn, num_workers=0)

        val_loader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)

        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, momentum=momentum, weight_decay=weight_decay,
        )

        best_val_map = 0.0
        for ep in range(n_epochs_per_trial):
            train_one_epoch(model, optimizer, train_loader, device, config["grad_clip"])
            val_map50, _ = evaluate_epoch(model, val_loader, device,
                                          class_names_no_bg, config["score_threshold"])
            best_val_map = max(best_val_map, val_map50)
            trial.report(val_map50, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        del model; gc.collect()
        return best_val_map

    output_dir = OPTUNA_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(),
        pruner=MedianPruner(),
        study_name=f"{OPTUNA_CONFIG['study_name']}_{mode_label}",
    )
    print(f"\n   Optuna: {n_trials} trials × {n_epochs_per_trial} epochs chacun")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print(f"\n   Meilleurs hyperparamètres: {best}")

    with open(os.path.join(output_dir, f"optuna_best_{mode_label}.json"), 'w') as f:
        json.dump({"best_params": best, "best_value": study.best_value}, f, indent=2)

    return best


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SSD — détection des toitures cadastrales",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", default="simple",
        choices=["simple", "attention", "optimize", "dual",
                 "nadir", "oblique", "all"],  # nadir/oblique/all: alias dépréciés
        help="Mode d'entraînement",
    )
    parser.add_argument("--aug",            nargs="*", default=[],
                        metavar="CLASS:COEFF",
                        help="Coefficients d'oversampling par classe (ex: panneau_solaire:3)")
    parser.add_argument("--cbam-reduction", type=int, default=16,
                        help="Facteur de réduction CBAM (ChannelAttention)")
    parser.add_argument("--cbam-kernel-size", type=int, default=7,
                        help="Taille du kernel CBAM (SpatialAttention)")
    parser.add_argument("--n-trials",       type=int, default=OPTUNA_CONFIG["n_trials"],
                        help="Nombre de trials Optuna")
    parser.add_argument("--n-epochs-trial", type=int, default=OPTUNA_CONFIG["n_epochs_per_trial"],
                        help="Epochs par trial Optuna")
    parser.add_argument("--images-dir",       default=None)
    parser.add_argument("--annotations-file", default=None)
    parser.add_argument("--output-dir",       default=None)
    parser.add_argument("--classes-file",     default=None)
    parser.add_argument("--model-name",       default=None,
                        choices=["ssd300_vgg16", "ssdlite320_mobilenet_v3_large"])
    args = parser.parse_args()

    # Alias dépréciés
    mode = args.mode
    if mode == "nadir":
        print("[DEPRECATED] --mode nadir est déprécié. Utilisez --mode dual ou --mode simple avec --annotations-file instances_nadir.json")
        args.mode = "simple"
    elif mode == "oblique":
        print("[DEPRECATED] --mode oblique est déprécié. Utilisez --mode dual ou --mode simple avec --annotations-file instances_oblique.json")
        args.mode = "simple"
    elif mode == "all":
        print("[DEPRECATED] --mode all est déprécié. Utilisez --mode simple.")
        args.mode = "simple"
    mode = args.mode

    config = build_config(args)

    if mode == "dual":
        # Nadir : panneau_solaire
        nadir_cfg          = _build_sub_config(config, "nadir")
        nadir_cfg["classes"] = load_classes(config["classes_file"],
                                            MODE_CLASSES["nadir"])
        aug_coeffs = parse_aug_coeffs(args.aug, nadir_cfg["classes"])
        class_names_no_bg = [c for c in nadir_cfg["classes"] if c != '__background__']
        print(f"\n[DUAL] Phase 1 — Nadir ({class_names_no_bg})")
        _train_single(nadir_cfg, aug_coeffs, "nadir", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)

        # Oblique : 4 classes bâtiment
        oblique_cfg          = _build_sub_config(config, "oblique")
        oblique_cfg["classes"] = load_classes(config["classes_file"],
                                              MODE_CLASSES["oblique"])
        aug_coeffs = parse_aug_coeffs(args.aug, oblique_cfg["classes"])
        class_names_no_bg = [c for c in oblique_cfg["classes"] if c != '__background__']
        print(f"\n[DUAL] Phase 2 — Oblique ({class_names_no_bg})")
        _train_single(oblique_cfg, aug_coeffs, "oblique", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)

    elif mode == "optimize":
        all_mode_classes = (MODE_CLASSES["nadir"] + MODE_CLASSES["oblique"])
        config["classes"] = load_classes(config["classes_file"], all_mode_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        best_params = _run_optimization(
            config, aug_coeffs, "all",
            args.cbam_reduction, args.cbam_kernel_size,
            args.n_trials, args.n_epochs_trial,
        )
        # Entraînement final avec les meilleurs hyperparamètres
        use_cbam = best_params.get("use_cbam", False)
        config["learning_rate"] = best_params.get("lr",           config["learning_rate"])
        config["weight_decay"]  = best_params.get("weight_decay", config["weight_decay"])
        config["momentum"]      = best_params.get("momentum",     config["momentum"])
        cbam_r = best_params.get("cbam_reduction",  args.cbam_reduction)
        cbam_k = best_params.get("cbam_kernel_size", args.cbam_kernel_size)
        training_mode = "attention" if use_cbam else "simple"
        _train_single(config, aug_coeffs, "all", training_mode, cbam_r, cbam_k)

    elif mode == "attention":
        all_mode_classes  = (MODE_CLASSES["nadir"] + MODE_CLASSES["oblique"])
        config["classes"] = load_classes(config["classes_file"], all_mode_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        _train_single(config, aug_coeffs, "all", "attention",
                      args.cbam_reduction, args.cbam_kernel_size)

    else:  # simple (+ anciens alias nadir/oblique/all redirigés ici)
        all_mode_classes  = (MODE_CLASSES["nadir"] + MODE_CLASSES["oblique"])
        config["classes"] = load_classes(config["classes_file"], all_mode_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        _train_single(config, aug_coeffs, "all", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)


if __name__ == "__main__":
    main()
