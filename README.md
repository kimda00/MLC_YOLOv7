## 1. Training TL Detector (labeled data)

``` shell
# train p5 models
python train.py --workers 8 --device 0 --batch-size 4 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

```

## 2. Crop the traffic light images

On image:
``` shell
python tl_crop.py --weights weights/onlytl.pt --conf 0.25 --img-size 640 --source inference/testset
```
## 3. 클래스 폴더를 만들고 거기에 이미지 분류해서 넣기

## 4. 클래스별로 이름 변경하기
``` shell
python change_name.py
```

## 5. MLC training 이미지폴더와 CVS 파일 만들기
``` shell
python random_pick_pic.py
```
아직 여러클래스겹치는 부분 수정 필요

## 6. Training for Multi-Label Classification(MLC)

``` shell
python train_mlc.py --workers 8 --device 0 --batch-size 4 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

```

## 7. Get results MLC

TODO : MLC 결과의 컨피던스를 설정하는 부분 추가 

``` shell
python detect_with_MLC.py --weights weights/onlytl.pt --conf 0.25 --save-txt --save-conf --img-size 640 --source inference/testset
```

## 8. Testing for MLC

``` shell
python test_for_MLC.py --device cpu 
```
TODO : make 1 confidence for gt

==============================================================

## Testing for YOLOv7

``` shell
python yolo_test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```

## Export

```shell
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
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







