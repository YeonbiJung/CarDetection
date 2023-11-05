#basecode는 grad_cam.py 사용하고 grid로 전체 이미지 보려고할때만 이거사용 

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam

# 데이터셋 클래스 정의
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

# 데이터셋에 적용할 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 데이터셋과 데이터 로더 정의
image_folder = 'path/to/your/images'
dataset = ImageDataset(image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 모델 불러오기
model = models.resnet50(pretrained=True)
model.eval()

# 대상 레이어 설정
target_layer = model.layer4[-1]

# Grad-CAM 및 Grad-CAM++ 인스턴스 생성
gradcam = GradCAM(model, target_layer)
gradcam_pp = GradCAMpp(model, target_layer)

# 이미지 순회 및 Grad-CAM 적용
for images, img_names in dataloader:
    inputs = images.requires_grad_()

    # 모델 예측
    outputs = model(inputs)

    # 타겟 클래스 설정 (예시로 가장 높은 점수를 받은 클래스 사용)
    _, preds = torch.max(outputs, 1)
    target = preds.item()

    # Grad-CAM 및 Grad-CAM++ 적용
    mask, _ = gradcam(inputs, class_idx=target)
    mask_pp, _ = gradcam_pp(inputs, class_idx=target)
    
    # 시각화 및 출력
    heatmap, result = visualize_cam(mask, images)
    heatmap_pp, result_pp = visualize_cam(mask_pp, images)
    
    # 모든 이미지를 하나의 그리드로 병합
    images_all = torch.stack([images.squeeze(), heatmap, heatmap_pp, result, result_pp], 0)
    images_all = make_grid(images_all, nrow=5)
    
    # 그리드 이미지 시각화
    plt.figure(figsize=(20, 5))
    plt.imshow(np.transpose(images_all.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.show()

    # 결과 저장 (옵션)
    # plt.savefig(f'output/{img_names[0]}_gradcam.jpg')
