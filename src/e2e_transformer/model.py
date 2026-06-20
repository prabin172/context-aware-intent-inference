import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, output_dim=128, pretrained=True):
        super().__init__()

        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except Exception:
                weights = None

        try:
            self.base_model = models.resnet18(weights=weights)
        except Exception as e:
            print(f"Warning: could not load pretrained ResNet18 weights ({e}). Using random init.")
            self.base_model = models.resnet18(weights=None)

        if in_channels == 1:
            self.base_model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, output_dim)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feats = self.base_model(x)
        return feats.reshape(b, t, -1)


class MotionMLP(nn.Module):
    def __init__(self, input_dim=66, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        # x: (B, T, 66)
        b, t, d = x.shape
        x = x.reshape(b * t, d)
        out = self.mlp(x)
        return out.reshape(b, t, -1)


class TransformerFusionModel(nn.Module):
    def __init__(
        self,
        num_classes=9,
        gesture_classes=None,
        motion_dim=66,
        embedding_dim=128,
        num_heads=8,
        num_layers=2,
        pretrained_cnn=True,
    ):
        super().__init__()

        if gesture_classes is not None:
            num_classes = gesture_classes

        self.rgb_encoder = CNNEncoder(
            in_channels=3,
            output_dim=embedding_dim,
            pretrained=pretrained_cnn,
        )
        self.depth_encoder = CNNEncoder(
            in_channels=1,
            output_dim=embedding_dim,
            pretrained=pretrained_cnn,
        )
        self.motion_encoder = MotionMLP(
            input_dim=motion_dim,
            output_dim=embedding_dim,
        )

        fused_dim = embedding_dim * 3

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb, depth, motion):
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)
        motion_feat = self.motion_encoder(motion)

        fused = torch.cat([rgb_feat, depth_feat, motion_feat], dim=-1)
        encoded = self.transformer_encoder(fused)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)
