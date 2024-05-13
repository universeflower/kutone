import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ultralytics import YOLOv8  # YOLOv8 모델 클래스를 임포트합니다.

# 하이퍼파라미터 설정
batch_size = 2  # 배치 크기를 2로 설정하여 총 6장의 이미지를 학습에 사용합니다.
epochs = 50
learning_rate = 0.001

# 데이터셋 경로 설정
image_folder_path = "C:/Users/user/Desktop/경기도 어학지원/Sample/01.원천데이터/2.직접촬영/01.금속캔"
label_file_path = "C:/Users/user/Desktop/경기도 어학지원/Sample/02.라벨링데이터/2.직접촬영/01.금속캔"  # labeling 된 파일 경로로 수정합니다.

# 데이터셋 변환
transform = transforms.Compose([
    transforms.Resize((416, 416)),  # YOLOv8는 416x416 이미지를 입력으로 사용
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    # 다른 전처리 및 데이터 증강 기법을 추가할 수 있음
])

# 데이터셋 불러오기
dataset = CustomDataset(image_folder_path, label_file_path, transform=transform)

# DataLoader 설정
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 클래스 수 설정 (예: 데이터셋에 1개의 클래스인 경우)
num_classes = 1  # 캔 이미지의 클래스 수를 설정합니다.

model = YOLOv8(num_classes=num_classes).to(device)

# 손실 함수와 옵티마이저 설정
criterion = ...  # 손실 함수 설정 (예: YOLO 손실 함수)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 모델 훈련
for epoch in range(epochs):
    model.train()
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device)

        # 순전파 및 역전파
        outputs = model(images)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 훈련 과정 출력
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

# 훈련된 모델 저장
torch.save(model.state_dict(), "yolov8_trained.pth")
