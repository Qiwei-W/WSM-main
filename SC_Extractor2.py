import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from kymatio.torch import Scattering2D
import torchvision.transforms.functional as TF
import numpy as np

class WaveletScatteringFeatureExtractor(nn.Module):
    def __init__(self,
                 input_size=224,
                 J=2, L=6,
                 use_color=False,
                 grid_h=14, grid_w=14,
                 apply_jitter=False):
        super().__init__()
        self.input_size = input_size
        self.use_color = use_color
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_regions = grid_h * grid_w

        self.scattering = Scattering2D(J=J, L=L, shape=(input_size, input_size))

        transform_list = [transforms.Resize((input_size, input_size))]
        if apply_jitter:
            transform_list.append(transforms.ColorJitter(brightness=0.05, contrast=0.15))
        if not use_color:
            transform_list.append(transforms.Grayscale())
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406] if use_color else [0.485],
                std=[0.229, 0.224, 0.225] if use_color else [0.229]
            )
        ])
        self.transform = transforms.Compose(transform_list)

    def extract_region_features(self, scat_tensor):
        # 输入 scat_tensor: [C, H, W]
        # 批量计算所有patch特征，无for循环
        C, H, W = scat_tensor.shape
        region_h = H // self.grid_h
        region_w = W // self.grid_w

        # 利用 unfold 展开所有patch: 结果形状 [C, grid_h*grid_w, region_h*region_w]
        patches = scat_tensor.unfold(1, region_h, region_h).unfold(2, region_w, region_w)
        # patches形状: [C, grid_h, grid_w, region_h, region_w]
        patches = patches.contiguous().view(C, -1, region_h * region_w)  # [C, num_patches, patch_size]

        # 计算均值和标准差: 按patch维度，dim=2是patch内像素维度
        mean_feat = patches.mean(dim=2)  # [C, num_patches]
        std_feat = patches.std(dim=2)    # [C, num_patches]

        # 排序top-k能量
        sorted_vals, _ = torch.sort(patches, dim=2, descending=True)
        top_20 = max(1, int(0.2 * patches.shape[2]))
        energy_feat = sorted_vals[:, :, :top_20].mean(dim=2)  # [C, num_patches]

        # 连接三个特征，输出 [num_patches, 3*C]
        feats = torch.cat([mean_feat, std_feat, energy_feat], dim=0)  # [3*C, num_patches]
        feats = feats.permute(1, 0).contiguous()  # [num_patches, 3*C]

        return feats

    def extract_patch_features(self, scat_tensor):
        # 同extract_region_features，直接调用
        return self.extract_region_features(scat_tensor)

    def forward_patch_features(self, x):
        """
        输入:
        x: [B, C, H, W]
        输出:
        patch_feats: [B, grid_h*grid_w, 3*C]
        """
        if x.shape[1] == 3:
            x = TF.rgb_to_grayscale(x)

        scat_feats = self.scattering(x).squeeze(1)  # [B, C, H, W]

        # 使用批量操作，避免for循环
        B, C, H, W = scat_feats.shape
        region_h = H // self.grid_h
        region_w = W // self.grid_w

        # 使用unfold批量提取patch
        patches = scat_feats.unfold(2, region_h, region_h).unfold(3, region_w, region_w)
        # patches: [B, C, grid_h, grid_w, region_h, region_w]
        patches = patches.contiguous().view(B, C, self.grid_h * self.grid_w, region_h * region_w)  # [B, C, num_patches, patch_size]

        # 计算均值和标准差
        mean_feat = patches.mean(dim=3)  # [B, C, num_patches]
        std_feat = patches.std(dim=3)    # [B, C, num_patches]

        # top-k能量
        sorted_vals, _ = torch.sort(patches, dim=3, descending=True)
        top_20 = max(1, int(0.2 * patches.shape[3]))
        energy_feat = sorted_vals[:, :, :, :top_20].mean(dim=3)  # [B, C, num_patches]

        # 拼接
        feats = torch.cat([mean_feat, std_feat, energy_feat], dim=1)  # [B, 3*C, num_patches]
        feats = feats.permute(0, 2, 1).contiguous()  # [B, num_patches, 3*C]

        return feats

    def forward(self, x, spatial_stats=False):
        if x.shape[1] == 3:
            x = TF.rgb_to_grayscale(x)

        scat_feats = self.scattering(x).squeeze(1)  # [B, C, H, W]

        if spatial_stats:
            return scat_feats
        else:
            # 调用批量版本extract_region_features进行池化特征提取
            # 这里也可以使用for循环调用单张的extract_region_features，如果数量不大
            all_feats = []
            for i in range(x.shape[0]):
                feat = self.extract_region_features(scat_feats[i])
                all_feats.append(feat)
            return torch.stack(all_feats)

    def preprocess(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return self.transform(img).unsqueeze(0)


def test_wavelet_scattering_feature_extractor():
    model = WaveletScatteringFeatureExtractor(input_size=224, use_color=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)  # batch=2，3通道输入

    with torch.no_grad():
        pooled_feats = model(x, spatial_stats=False)
        print("池化特征形状:", pooled_feats.shape)  # 预期：[2, 800]

        spatial_feats = model(x, spatial_stats=True)
        print("空间统计特征形状:", spatial_feats.shape)  # 预期：[2, C, H, W]

    dummy_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    pil_img = Image.fromarray(dummy_img)
    img_tensor = model.preprocess(pil_img)

    with torch.no_grad():
        feat = model(img_tensor, spatial_stats=False)
        print("单张图池化特征形状:", feat.shape)
        spatial_feat = model(img_tensor, spatial_stats=True)
        print("单张图空间统计特征形状:", spatial_feat.shape)


if __name__ == "__main__":
    test_wavelet_scattering_feature_extractor()
