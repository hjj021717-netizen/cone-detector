#!/usr/bin/env python3
"""
라바콘(cone) YOLO 데이터셋 준비 스크립트
Roboflow 다운로드 -> 클래스 확인 -> 단일 클래스(cone) 병합 -> 샘플 확인 -> 백업

사용 예:
    python prepare_cone_dataset.py all
    python prepare_cone_dataset.py download | inspect | merge | preview | backup

API 키는 코드에 쓰지 말고 코랩 보안 비밀(ROBOFLOW_KEY), 환경변수(ROBOFLOW_API_KEY),
또는 실행 시 입력창으로 넘기세요.
"""
import argparse
import glob
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

# keep: 콘으로 취급할 클래스 이름 (data.yaml의 이름과 같아야 함, 대소문자/공백 무시)
DATASETS = {
    "fs": {
        "workspace": "ksae",
        "project": "formula-student-cones-xtl4j",
        "version": 2,
        "keep": ["blue cone", "yellow cone", "big orange"],
    },
    "small": {
        "workspace": "school-workspace",
        "project": "small-plastic-cones",
        "version": 2,
        "keep": ["cone"],
    },
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "test"]


def norm(s):
    return str(s).strip().lower()


def get_api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key.strip()
    try:
        from google.colab import userdata
        key = userdata.get("ROBOFLOW_KEY")
        if key:
            return key.strip()
    except Exception:
        pass
    from getpass import getpass
    return getpass("Roboflow API Key 입력: ").strip()


def load_names(data_yaml):
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def cmd_download(args):
    from roboflow import Roboflow

    rf = Roboflow(api_key=get_api_key())
    for key, d in DATASETS.items():
        loc = Path(args.raw_dir) / key
        print(f"[{key}] 다운로드 -> {loc}")
        rf.workspace(d["workspace"]).project(d["project"]).version(d["version"]).download(
            "yolov8", location=str(loc)
        )


def cmd_inspect(args):
    for key in DATASETS:
        path = Path(args.raw_dir) / key
        names = load_names(path / "data.yaml")
        print(f"[{key}] 클래스 목록:")
        for i, n in enumerate(names):
            mark = "  <- KEEP" if norm(n) in {norm(k) for k in DATASETS[key]["keep"]} else ""
            print(f"   {i}: {n!r}{mark}")
        for sp in SPLITS:
            n_img = len(glob.glob(f"{path}/{sp}/images/*"))
            print(f"   {sp}: 이미지 {n_img}장")
        print()


def to_bbox(parts):
    """YOLO 박스(5개 값) 또는 폴리곤 라벨 -> [cx, cy, w, h]"""
    vals = list(map(float, parts[1:]))
    if len(vals) == 4:
        return vals
    xs, ys = vals[0::2], vals[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def cmd_merge(args):
    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)

    for key, d in DATASETS.items():
        path = Path(args.raw_dir) / key
        names = load_names(path / "data.yaml")
        keep_set = {norm(k) for k in d["keep"]}
        keep_ids = {i for i, n in enumerate(names) if norm(n) in keep_set}
        print(f"[{key}] 콘 클래스 번호 {sorted(keep_ids)} -> {[names[i] for i in sorted(keep_ids)]}")
        if not keep_ids:
            sys.exit(f"[{key}] keep 이름이 data.yaml과 일치하지 않습니다. inspect 결과를 확인하세요.")

        for sp in SPLITS:
            img_dir, lab_dir = path / sp / "images", path / sp / "labels"
            if not img_dir.exists():
                continue
            (out / sp / "images").mkdir(parents=True, exist_ok=True)
            (out / sp / "labels").mkdir(parents=True, exist_ok=True)
            n_keep = n_drop = n_box = n_neg = 0

            for img in img_dir.iterdir():
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                lab = lab_dir / (img.stem + ".txt")
                rows = []
                if lab.exists():
                    rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]

                kept = []
                for r in rows:
                    if int(float(r[0])) in keep_ids:
                        cx, cy, w, h = to_bbox(r)
                        kept.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                if rows and not kept:  # 라벨은 있는데 콘이 없는 이미지는 제외
                    n_drop += 1
                    continue

                new_name = f"{key}_{img.name}"
                shutil.copy(img, out / sp / "images" / new_name)
                (out / sp / "labels" / (Path(new_name).stem + ".txt")).write_text("\n".join(kept))
                n_keep += 1
                n_box += len(kept)
                if not kept:
                    n_neg += 1

            print(f"   {sp:5s} 사용 {n_keep}장 / 박스 {n_box}개 / 배경이미지 {n_neg}장 / 제외 {n_drop}장")

    lines = [f"path: {out.resolve()}", "train: train/images", "val: valid/images"]
    if (out / "test" / "images").exists():
        lines.append("test: test/images")
    lines += ["nc: 1", "names: ['cone']", ""]
    (out / "data.yaml").write_text("\n".join(lines))
    print("\n[data.yaml]\n" + "\n".join(lines))


def cmd_preview(args):
    import cv2
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out_dir)
    imgs = sorted((out / args.split / "images").iterdir())
    if args.prefix:
        imgs = [p for p in imgs if p.name.startswith(args.prefix)]
    picks =
