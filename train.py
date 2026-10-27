#!/usr/bin/env python3
"""YOLO 라바콘 학습 스크립트

사용 예:
    python train.py                       # 기본값으로 학습
    python train.py --epochs 3 --name smoke_test   # 연습 학습
    python train.py --resume              # 끊긴 학습 이어서
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/cone_merged/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="시작 가중치 (n/s/m 등)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--project", default="/content/drive/MyDrive/cone_project/runs")
    ap.add_argument("--name", default="cone_base")
    ap.add_argument("--resume", action="store_true", help="last.pt로 이어서 학습")
    args = ap.parse_args()

    if args.resume:
        last = Path(args.project) / args.name / "weights" / "last.pt"
        YOLO(str(last)).train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience,
        hsv_h=0.5, hsv_s=0.7, hsv_v=0.5,      # 색에 둔감하게 만드는 증강
        mosaic=1.0, scale=0.5, fliplr=0.5,
        project=args.project, name=args.name,
    )

    best = YOLO(str(Path(args.project) / args.name / "weights" / "best.pt"))
    m = best.val(data=args.data)
    print(f"mAP50: {m.box.map50:.3f}  Precision: {m.box.mp:.3f}  Recall: {m.box.mr:.3f}")


if __name__ == "__main__":
    main()
