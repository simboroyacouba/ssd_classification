"""
Entraînement SSD unifié — un seul modèle, toutes classes

Différences avec train.py :
  - Argparse au lieu du dict CONFIG global
  - Sauvegarde dans ssd_unified_{timestamp}/
  - Écrit model_info_unified.json pour eval/inference
  - Aucune séparation nadir / oblique

Usage :
  python train_unified.py
  python train_unified.py --model-name ssdlite320_mobilenet_v3_large
  python train_unified.py --epochs 80 --batch-size 8 --lr 0.005
"""

import os
import json
import yaml
import shutil
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import time
import torch
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
# TAILLES D'IMAGE FIXES
# =============================================================================

SSD_IMAGE_SIZES = {
    "ssd300_vgg16":                  300,
    "ssdlite320_mobilenet_v3_large": 320,
}


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config(args):
    return {
        "images_dir":       args.images_dir,
        "annotations_file": args.annotations_file,
        "output_dir":       args.output_dir,
        "classes_file":     args.classes_file,
        "model_name":       args.model_name,
        "num_epochs":       args.epochs,
        "batch_size":       args.batch_size,
        "learning_rate":    args.lr,
        "momentum":         args.momentum,
        "weight_decay":     args.weight_decay,
        "train_split":      args.train_split,
        "val_split":        args.val_split,
        "test_split":       args.test_split,
        "save_every":       args.save_every,
        "score_threshold":  args.score_threshold,
        "pretrained":       not args.no_pretrained,
        "attention":        args.attention,
        "augment":          args.augment,
        "grad_clip":        args.grad_clip,
        "class_weights":    args.class_weights,
    }


# =============================================================================
# CLASSES
# =============================================================================

def load_classes(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"classes.yaml introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    classes = data.get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    print("Classes chargées:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


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
    print(f"\n   Split (total: {n_total}):")
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
    print(f"\n   {'Classe':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"   {'-'*68}")
    for cat_id in coco.getCatIds():
        name  = coco.cats[cat_id]['name']
        train = stats['train'].get(cat_id, 0)
        val   = stats['val'].get(cat_id, 0)
        test  = stats['test'].get(cat_id, 0)
        total = train + val + test
        warn  = " !" if val == 0 or test == 0 else ""
        print(f"   {name:<30} {train:>8} {val:>8} {test:>8} {total:>8}{warn}")
    print(f"   {'-'*68}")


# =============================================================================
# DATASET + AUGMENTATION
# =============================================================================

import random as _random
import torchvision.transforms as T


class SSDaugmenter:
    """
    Pipeline d'augmentation SSD avec adaptation des bounding boxes.
    Toutes les transformations géométriques ajustent les boîtes.
    """

    def __init__(self,
                 flip_prob=0.5,
                 color_jitter_prob=0.5,
                 expand_prob=0.5,
                 expand_max_ratio=3.0,
                 brightness=0.3,
                 contrast=0.3,
                 saturation=0.3,
                 hue=0.1):
        self.flip_prob         = flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.expand_prob       = expand_prob
        self.expand_max_ratio  = expand_max_ratio
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast,
                                    saturation=saturation, hue=hue)

    def __call__(self, image, boxes):
        """
        image : PIL Image (après resize vers image_size)
        boxes : list of [x1,y1,x2,y2] en pixels dans image_size
        Retourne (image PIL, boxes list)
        """
        W, H = image.size

        # 1. Zoom out (expand) : ajoute du fond autour de l'image
        if boxes and _random.random() < self.expand_prob:
            ratio  = _random.uniform(1.0, self.expand_max_ratio)
            new_W  = int(W * ratio)
            new_H  = int(H * ratio)
            off_x  = _random.randint(0, new_W - W)
            off_y  = _random.randint(0, new_H - H)
            # fond = couleur moyenne de normalisation (gris ~128)
            canvas = Image.new("RGB", (new_W, new_H), (123, 117, 104))
            canvas.paste(image, (off_x, off_y))
            image  = canvas.resize((W, H), Image.BILINEAR)
            # ajuster les boîtes : scale + offset
            sx = W / new_W; sy = H / new_H
            boxes = [
                [(b[0] + off_x) * sx, (b[1] + off_y) * sy,
                 (b[2] + off_x) * sx, (b[3] + off_y) * sy]
                for b in boxes
            ]

        # 2. Retournement horizontal
        if _random.random() < self.flip_prob:
            image = TF.hflip(image)
            boxes = [[W - b[2], b[1], W - b[0], b[3]] for b in boxes]

        # 3. Distorsion photométrique (couleur uniquement, pas de géométrie)
        if _random.random() < self.color_jitter_prob:
            image = self.jitter(image)

        # 4. Nettoyer les boîtes dégénérées
        boxes = [
            [max(0.0, b[0]), max(0.0, b[1]),
             min(float(W), b[2]), min(float(H), b[3])]
            for b in boxes
            if b[2] > b[0] + 1 and b[3] > b[1] + 1
        ]
        return image, boxes


class SSDDataset(Dataset):
    def __init__(self, images_dir, annotations_file, image_ids,
                 cat_mapping, image_size=300, augment=False):
        self.images_dir  = images_dir
        self.coco        = COCO(annotations_file)
        self.image_ids   = image_ids
        self.cat_mapping = cat_mapping
        self.image_size  = image_size
        self.augmenter   = SSDaugmenter() if augment else None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        image    = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image    = image.resize((self.image_size, self.image_size))
        scale_x  = self.image_size / orig_w
        scale_y  = self.image_size / orig_h

        anns   = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
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

        # Augmentation (train uniquement)
        if self.augmenter is not None and boxes:
            image, boxes = self.augmenter(image, boxes)

        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

        target = {
            'boxes':    torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4), dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),   dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
        }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =============================================================================
# MODULES D'ATTENTION
# =============================================================================

class ChannelAttention(torch.nn.Module):
    """SE-Net : attention canal via avg+max pooling."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(channels, mid, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(torch.nn.Module):
    """CBAM : attention spatiale via avg+max sur les canaux."""
    def __init__(self):
        super().__init__()
        self.conv    = torch.nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(torch.nn.Module):
    """Channel + Spatial Attention Module."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class AttentionBackbone(torch.nn.Module):
    """Wrapper : applique un module d'attention sur chaque feature map du backbone SSD."""
    def __init__(self, backbone, channels, attention_type):
        super().__init__()
        self.backbone = backbone
        if attention_type == 'se':
            self.attentions = torch.nn.ModuleList([ChannelAttention(c) for c in channels])
        elif attention_type == 'cbam':
            self.attentions = torch.nn.ModuleList([CBAM(c) for c in channels])
        else:
            self.attentions = torch.nn.ModuleList([torch.nn.Identity() for _ in channels])

    def forward(self, x):
        features = self.backbone(x)
        result = {}
        for i, (k, v) in enumerate(features.items()):
            attn = self.attentions[i] if i < len(self.attentions) else torch.nn.Identity()
            result[k] = attn(v)
        return result


def _probe_backbone_channels(backbone, image_size):
    """Détecte les tailles de canaux de chaque feature map via un forward factice."""
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size)
        feats = backbone(dummy)
    return [f.shape[1] for f in feats.values()]


# =============================================================================
# MODÈLE
# =============================================================================

def build_model(model_name, num_classes, pretrained=True, attention_type=None, image_size=300):
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
        raise ValueError(f"Modèle inconnu: {model_name}")

    if attention_type and attention_type != 'none':
        channels = _probe_backbone_channels(model.backbone, image_size)
        model.backbone = AttentionBackbone(model.backbone, channels, attention_type)
        print(f"   Attention '{attention_type}' ajoutée sur {len(channels)} feature maps : {channels}")

    return model


# =============================================================================
# MÉTRIQUES (mAP@50 simplifié)
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
            mask_p  = pred['labels'] == cls_id
            mask_g  = gt['labels']   == cls_id
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
# PONDÉRATION DES CLASSES (class-balanced sampler)
# =============================================================================

def build_class_weighted_sampler(dataset, coco, cat_mapping, class_names_no_bg):
    """WeightedRandomSampler : sur-échantillonne les images avec des classes rares."""
    num_classes = len(class_names_no_bg)

    # Fréquence globale de chaque classe (nombre d'annotations dans le train set)
    class_counts = np.zeros(num_classes + 1, dtype=np.float32)  # index 1..num_classes
    for img_id in dataset.image_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        for ann in anns:
            cls_idx = cat_mapping.get(ann['category_id'])
            if cls_idx is not None and 1 <= cls_idx <= num_classes:
                class_counts[cls_idx] += 1

    # Poids inverse de fréquence par classe (classes rares → poids élevé)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts

    # Poids de chaque image = max poids parmi toutes ses classes
    sample_weights = []
    for img_id in dataset.image_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        img_w = 0.0
        for ann in anns:
            cls_idx = cat_mapping.get(ann['category_id'])
            if cls_idx is not None and 1 <= cls_idx <= num_classes:
                img_w = max(img_w, class_weights[cls_idx])
        sample_weights.append(max(img_w, 1e-6))

    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    print("   Poids d'echantillonnage par classe :")
    for i, name in enumerate(class_names_no_bg, start=1):
        print(f"      {name:<30} count={int(class_counts[i])}  weight={class_weights[i]:.4f}")

    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


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
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SSD unifié — toutes classes, un seul modèle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir",       default=os.getenv("DETECTION_DATASET_IMAGES_DIR",       "../dataset1/images/default"))
    parser.add_argument("--annotations-file", default=os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"))
    parser.add_argument("--output-dir",       default=os.getenv("OUTPUT_DIR",                         "./runs/detect/train"))
    parser.add_argument("--classes-file",     default=os.getenv("CLASSES_FILE",                       "classes.yaml"))
    parser.add_argument("--model-name",       default=os.getenv("SSD_MODEL",                          "ssd300_vgg16"),
                        choices=["ssd300_vgg16", "ssdlite320_mobilenet_v3_large"])
    parser.add_argument("--epochs",           type=int,   default=int(os.getenv("NUM_EPOCHS",   "50")))
    parser.add_argument("--batch-size",       type=int,   default=int(os.getenv("BATCH_SIZE",    "4")))
    parser.add_argument("--lr",               type=float, default=float(os.getenv("LEARNING_RATE", "0.01")))
    parser.add_argument("--momentum",         type=float, default=float(os.getenv("MOMENTUM",    "0.9")))
    parser.add_argument("--weight-decay",     type=float, default=float(os.getenv("WEIGHT_DECAY","5e-4")))
    parser.add_argument("--train-split",      type=float, default=float(os.getenv("TRAIN_SPLIT", "0.70")))
    parser.add_argument("--val-split",        type=float, default=float(os.getenv("VAL_SPLIT",   "0.20")))
    parser.add_argument("--test-split",       type=float, default=float(os.getenv("TEST_SPLIT",  "0.10")))
    parser.add_argument("--save-every",       type=int,   default=int(os.getenv("SAVE_EVERY",    "5")))
    parser.add_argument("--score-threshold",  type=float, default=float(os.getenv("SCORE_THRESHOLD", "0.3")))
    parser.add_argument("--grad-clip",        type=float, default=float(os.getenv("GRAD_CLIP",   "1.0")))
    parser.add_argument("--no-pretrained",    action="store_true")
    parser.add_argument("--attention",        default=os.getenv("ATTENTION", "none"),
                        choices=["none", "se", "cbam"],
                        help="Mecanisme d'attention sur les feature maps du backbone (none/se/cbam)")
    parser.add_argument("--augment",          action="store_true", default=os.getenv("AUGMENT","0")=="1",
                        help="Activer l'augmentation (flip, color jitter, zoom out)")
    parser.add_argument("--class-weights",    action="store_true", default=os.getenv("CLASS_WEIGHTS","0")=="1",
                        help="Activer le surechantillonnage pondéré par classe (classes rares favorisees)")
    args = parser.parse_args()

    config     = build_config(args)
    classes    = load_classes(config["classes_file"])
    num_classes       = len(classes)
    class_names_no_bg = [c for c in classes if c != '__background__']
    image_size        = SSD_IMAGE_SIZES.get(config["model_name"], 300)

    print("=" * 70)
    print(f"   SSD Unifie ({config['model_name']}) - Toutes classes")
    print("=" * 70)
    print(f"   Images:      {config['images_dir']}")
    print(f"   Annotations: {config['annotations_file']}")
    print(f"   Modele:      {config['model_name']}")
    print(f"   Classes:     {num_classes} (avec __background__)")
    print(f"   Epochs:      {config['num_epochs']} | Batch: {config['batch_size']} | LR: {config['learning_rate']}")
    print(f"   Image size:  {image_size}px (fixe pour SSD)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device:      {device}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir   = os.path.join(config["output_dir"], f"ssd_unified_{timestamp}")
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
            print(f"   Categorie COCO ignoree (absente du yaml) : '{cat_name}' (id={cat_id})")
    print(f"   cat_mapping: { {coco_cats[k]: v for k, v in cat_mapping.items()} }")

    train_ids, val_ids, test_ids, split_stats = stratified_split(
        coco, config["train_split"], config["val_split"], config["test_split"], seed=42
    )
    print_split_stats(coco, split_stats)

    test_info_path = os.path.join(train_dir, "test_info.json")
    test_info = {
        'test_image_ids':   test_ids,
        'cat_mapping':      {str(k): v for k, v in cat_mapping.items()},
        'images_dir':       os.path.abspath(config["images_dir"]),
        'annotations_file': os.path.abspath(config["annotations_file"]),
        'num_test_images':  len(test_ids),
        'classes':          classes,
        'model_name':       config["model_name"],
        'image_size':       image_size,
    }
    with open(test_info_path, 'w') as f:
        json.dump(test_info, f, indent=2)

    train_dataset = SSDDataset(config["images_dir"], config["annotations_file"],
                               train_ids, cat_mapping, image_size,
                               augment=config["augment"])
    val_dataset   = SSDDataset(config["images_dir"], config["annotations_file"],
                               val_ids,   cat_mapping, image_size,
                               augment=False)
    if config["augment"]:
        print("   Augmentation activee : flip | color jitter | zoom out")
    if config["class_weights"]:
        sampler = build_class_weighted_sampler(train_dataset, coco, cat_mapping, class_names_no_bg)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  sampler=sampler, collate_fn=collate_fn, num_workers=0)
        print("   Mode: surechantillonnage pondéré par classe activé")
    else:
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=1,
                               shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"\n   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_ids)} images")

    print(f"\n   Chargement {config['model_name']} (pretrained={config['pretrained']})...")
    model = build_model(config["model_name"], num_classes, config["pretrained"],
                        config["attention"], image_size)
    model.to(device)

    params       = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.SGD(params, lr=config["learning_rate"],
                                   momentum=config["momentum"],
                                   weight_decay=config["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(config["num_epochs"] * 0.6), int(config["num_epochs"] * 0.85)],
        gamma=0.1,
    )

    print("\n" + "=" * 70)
    print(f"   ENTRAÎNEMENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
                'model_name': config["model_name"], 'image_size': image_size,
                'attention': config["attention"],
            }, best_path)
            print(f"   Meilleur modele sauvegarde (mAP@50: {best_map50:.4f})")

        if epoch % config["save_every"] == 0 or epoch == config["num_epochs"]:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50': val_map50, 'num_classes': num_classes,
                'classes': classes, 'cat_mapping': cat_mapping,
                'model_name': config["model_name"], 'image_size': image_size,
                'attention': config["attention"],
            }, os.path.join(weights_dir, "last.pth"))

    total_time = time.time() - start_time

    best_model_path = os.path.join(train_dir, "best_model.pth")
    if os.path.exists(best_path):
        shutil.copy2(best_path, best_model_path)
    if os.path.exists(os.path.join(weights_dir, "last.pth")):
        shutil.copy2(os.path.join(weights_dir, "last.pth"),
                     os.path.join(train_dir, "final_model.pth"))

    # model_info_unified.json (point d'entrée pour eval/inference)
    model_info = {
        "model":       "SSD_unified",
        "model_name":  config["model_name"],
        "best_model":  os.path.abspath(best_model_path),
        "train_dir":   os.path.abspath(train_dir),
        "test_info":   os.path.abspath(test_info_path),
        "classes":     classes,
        "num_classes": num_classes,
        "image_size":  image_size,
        "best_map50":  best_map50,
        "timestamp":   timestamp,
    }
    info_path = os.path.join(config["output_dir"], "model_info_unified.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"\n   model_info_unified.json -> {info_path}")

    history['best_map50'] = best_map50
    history['config']     = {k: str(v) for k, v in config.items()}
    with open(os.path.join(train_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    if history['train_loss']:
        epochs_r = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(epochs_r, history['train_loss'], 'b-', label='Total')
        axes[0].plot(epochs_r, history['cls_loss'],  'r--', label='Cls')
        axes[0].plot(epochs_r, history['bbox_loss'], 'g--', label='BBox')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs_r, history['val_map50'], 'g-')
        axes[1].set_title('mAP@50 (val)'); axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
        axes[2].plot(epochs_r, history['lr'], color='orange')
        axes[2].set_title('LR'); axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, 'training_curves.png'), dpi=150)
        plt.close()

    print("\n" + "=" * 70)
    print(f"   TERMINE")
    print("=" * 70)
    print(f"   Meilleur mAP@50: {best_map50:.4f} ({best_map50*100:.2f}%)")
    print(f"   Temps: {format_time(total_time)}")
    print(f"   Modele: {best_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
