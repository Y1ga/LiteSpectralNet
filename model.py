import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDW(nn.Module):

    def __init__(self, cin, cout, k=3, dilation=1):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.dw = nn.Conv1d(
            cin,
            cin,
            kernel_size=k,
            padding=pad,
            dilation=dilation,
            groups=cin,
            bias=False,
        )
        self.pw = nn.Conv1d(cin, cout, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.SiLU(inplace=True)
        self.res_conv = None
        if cin != cout:
            self.res_conv = nn.Conv1d(cin, cout, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        out = self.bn(out)
        res = x if self.res_conv is None else self.res_conv(x)
        out = out + res
        return self.act(out)


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class LiteSpectralNet(nn.Module):

    def __init__(self, output_channel):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualDW(1, 16, k=3),
            SEBlock(16),
            ResidualDW(16, 32, k=3, dilation=2),
            SEBlock(32),
            ResidualDW(32, 64, k=3),
            SEBlock(64),
        )
        self.output_channel = output_channel
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
            ResidualDW(64, 32, k=3),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder[0](x)
        x = F.interpolate(
            x, size=self.output_channel, mode="linear", align_corners=False
        )
        x = self.decoder[1:](x)

        return x.squeeze(1)


class Loss(nn.Module):
    def __init__(
        self, tv_weight: float = 0.1, mrae_weight: float = 0, eps: float = 1e-6
    ):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.tv_weight = tv_weight
        self.mrae_weight = mrae_weight
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mse_loss = self.mse(y_pred, y_true)
        tv_loss = torch.mean(torch.abs(y_pred[:, 1:] - y_pred[:, :-1]))
        rel_abs = torch.abs(y_pred - y_true) / (torch.abs(y_true) + self.eps)
        mrae_loss = torch.mean(rel_abs)
        total = mse_loss + self.tv_weight * tv_loss + self.mrae_weight * mrae_loss
        return total
