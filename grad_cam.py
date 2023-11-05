import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

# 이미지가 있는 폴더 경로
image_folder = 'path/to/your/image/dataset'

# 데이터셋 클래스 정의
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

# 데이터셋에 적용할 변환 정의(우리 모델 사이즈?)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 데이터셋과 데이터로더 정의
dataset = ImageDataset(image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 사전 훈련된 모델 로드
#이런식으로 사용 => model = models.resnet50(pretrained=True)
#우리꺼로 사용 => baseline.ipynb에 있는 build_model()에서 return 된 model 
model = build_model(num_classes = CFG["NUM_CLASS"]+1)
model.eval()

# Grad-CAM 또는 Grad-CAM++ 레이어 설정
target_layer = model.layer4[2].conv3

# CAM 인스턴스 생성
gradcam = GradCAM(model, target_layer)
gradcam_pp = GradCAMpp(model, target_layer)

# 이미지 순회
for inputs, img_name in dataloader:
    inputs = inputs.requires_grad_()
    
    # 모델 예측
    outputs = model(inputs)
    
    # 타겟 클래스 설정 (예: 가장 높은 점수를 받은 클래스)
    _, predicted = outputs.max(dim=1)
    target = predicted.item()

    # Grad-CAM과 Grad-CAM++ 적용
    mask, _ = gradcam(inputs, target)
    heatmap, result = visualize_cam(mask, inputs)

    mask_pp, _ = gradcam_pp(inputs, target)
    heatmap_pp, result_pp = visualize_cam(mask_pp, inputs)

    # 시각화
    images = torch.stack([inputs.squeeze(), heatmap, heatmap_pp, result, result_pp], 0)
    images = torch.cat([i.unsqueeze(0) for i in images], 0)
    grid_image = make_grid(images, nrow=5)

    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid_image.numpy(), (1, 2, 0)))
    plt.show()

    # 결과 저장 (옵션)
    # plt.savefig(f'{img_name[0]}_gradcam.jpg')
