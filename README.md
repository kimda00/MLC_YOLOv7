
# MLC_YOLOv7

**MLC_YOLOv7**은 YOLOv7에 **Multi-Label Classification (MLC)** 기능을 추가하여,  
단일 객체(예: 교통신호등)에 대해 **여러 클래스 레이블을 동시에 예측**할 수 있도록 확장한 객체 탐지 프로젝트입니다.

교통신호등 탐지를 예시로,  
탐지된 객체에 "적색", "점멸", "화살표" 등 **다중 레이블을 한 번에 부여할 수 있는 모델**을 학습하고 적용하는 것을 목표로 합니다.

---

## **주요 기능**
- YOLOv7 기반 객체 탐지
- **다중 레이블(Multi-Label) 분류 기능 추가**
- 교통신호등 이미지 학습/탐지 파이프라인 구축
- 탐지 결과 시각화
- ONNX, TensorRT 등 다양한 포맷으로 모델 내보내기 지원

---

## **개발 환경**
- Python 3.x
- PyTorch
- YOLOv7 (커스텀)

---

## **프로젝트 실행 순서**

### 1. Training TL Detector (labeled data)
```bash
python train.py --workers 8 --device 0 --batch-size 4 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

### 2. Crop the traffic light images
```bash
python tl_crop.py --weights weights/onlytl.pt --conf 0.25 --img-size 640 --source inference/testset
```

## 3. 클래스별 폴더 구성 및 이미지 분류

## 4. 클래스별 파일명 일괄 변경
``` shell
python change_name.py
```

## 5. MLC 학습용 이미지/CSV 생성
``` shell
python random_pick_pic.py
```

## 6. Training for Multi-Label Classification(MLC)

``` shell
python train_mlc.py --workers 8 --device 0 --batch-size 4 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

## 7. Get results MLC

TODO : MLC 결과의 컨피던스를 설정하는 부분 추가 

``` shell
python detect_with_MLC.py --weights weights/onlytl.pt --conf 0.25 --save-txt --save-conf --img-size 640 --source inference/testset
```

결과는 ./runs/detect/exp에 저장

images/: 원본 이미지

labels/: MLC 결과 라벨

→ 시각화하려면:

``` shell
python pviz.py 
```

- 기타 기능
YOLOv7 테스트

bash
python yolo_test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
모델 내보내기 (export)

bash
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
(ONNX/TensorRT 변환은 README 원문 참고)


## Testing for YOLOv7

``` shell
python yolo_test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```

## Export

```shell
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify         --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

**Pytorch to TensorRT with NMS (and inference)** 

```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

**Pytorch to TensorRT another way** 

```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights yolov7-tiny.pt --grid --include-nms
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16

# Or use trtexec to convert ONNX to TensorRT engine
/usr/src/tensorrt/bin/trtexec --onnx=yolov7-tiny.onnx --saveEngine=yolov7-tiny-nms.trt --fp16
```

