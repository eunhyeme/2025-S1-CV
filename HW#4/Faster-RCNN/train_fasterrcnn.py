import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import os

# 이미지 변환
def get_transform():
    return T.Compose([T.ToTensor()])

# target 전처리
def prepare_targets(targets):
    prepared = []
    for anns in targets:
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # coco bbox -> x1y1x2y2
            labels.append(ann['category_id'])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        prepared.append({"boxes": boxes, "labels": labels})
    return prepared

# 학습 루프
def train_model(data_path, img_size, epochs, batch_size, project, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # coco dataset 로드
    dataset = CocoDetection(
        root=os.path.join(data_path, "images", "train"),
        annFile=os.path.join(data_path, "annotations", "instances_train.json"),
        transform=get_transform()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # 모델 로딩
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.train()

    # 옵티마이저
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)

    for epoch in range(epochs):
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = prepare_targets(targets)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)

            # forward + loss 계산
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {losses.item()}")

    # 모델 저장
    os.makedirs(f"{project}/{name}", exist_ok=True)
    torch.save(model.state_dict(), f"{project}/{name}/fasterrcnn.pth")

# arg 받기
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="my_frcnn_exp")

    args = parser.parse_args()
    train_model(args.data, args.imgsz, args.epochs, args.batch, args.project, args.name)
