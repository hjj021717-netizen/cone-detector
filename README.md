# Cone Detector

2026 로보월드 4족보행 로봇경진대회용 라바콘 인식 YOLO 모델입니다.
공개 데이터셋 2종을 병합한 단일 클래스(`cone`) 모델이며, 대회장 촬영 데이터로 파인튜닝하기 전의 베이스 모델입니다.

## Classes

- 0: cone

## Files

- best.pt: 학습된 YOLO 모델
- data.yaml: 데이터셋 설정
- train.py: 학습 스크립트
- prepare_cone_dataset.py: 데이터셋 다운로드 및 병합 (Roboflow API 키 필요)
- training_results/: 학습 곡선, 혼동행렬, 예측 샘플

## Usage

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model('test.jpg', conf=0.25)

for box in results[0].boxes:
    print(model.names[int(box.cls[0])], float(box.conf[0]), box.xyxy[0].tolist())
```

## Training

```
pip install -r requirements.txt
python prepare_cone_dataset.py all      # 데이터 준비 (Roboflow API 키 필요)
python train.py                         # 학습
```

## Notes

- 학습 데이터: 아래 공개 데이터셋 2종을 `cone` 단일 클래스로 병합 (CC BY 4.0)
  - Formula Student Cones (ksae): https://universe.roboflow.com/ksae/formula-student-cones-xtl4j
  - Small Plastic Cones (school-workspace): https://universe.roboflow.com/school-workspace/small-plastic-cones
- 학습 조건: YOLOv8n, imgsz 640, 색 증강(hsv_h 0.5), 최대 100 epochs
- 검증 결과 (공개 데이터 검증 세트 기준)
  - 전체: mAP50 0.666 / Precision 0.931 / Recall 0.638
    - 포뮬러 스튜던트 데이터에 멀리 있는 작은 콘이 많아 Recall이 낮게 나옴
  - 소형 콘(small_): mAP50 0.994 / Precision 0.985 / Recall 0.974
  - 가까운 콘(fs_, 박스 높이가 이미지의 10% 이상): Recall 0.853
- 이 모델은 대회장에서 촬영한 이미지로 학습한 것이 아닙니다. 현장 데이터로 파인튜닝하면 성능이 달라질 수 있습니다.
- 접촉 패널티를 줄이려면 놓침(Recall)이 중요하므로, conf 임계값을 낮춰서 사용하는 것을 권장합니다.
