import torch
import torch.nn as nn
import torch.nn.functional as F


class F09Net(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0):
        super(F09Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, 6 * 9 * 256, bias=False),
            nn.BatchNorm1d(num_features=6 * 9 * 256, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, bias=False, padding=2),
            nn.BatchNorm2d(num_features=128, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, bias=False, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=64, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, bias=False, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=64, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, bias=False, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=32, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, bias=False, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=32, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.layer7 = nn.ConvTranspose2d(32, out_channels, kernel_size=5, stride=2, bias=True, padding=2,
                                         output_padding=1)

    def forward(self, x):
        h1 = self.layer1(x)
        h1_reshape = h1.view(-1, 256, 6, 9)
        h2 = self.layer2(h1_reshape)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        h5 = self.layer5(h4)
        h6 = self.layer6(h5)
        output = self.layer7(h6)

        return output


class Prob_F09Net(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0):
        super(Prob_F09Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, 6 * 9 * 256, bias=False),
            nn.BatchNorm1d(num_features=6 * 9 * 256, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, bias=False, padding=2),
            nn.BatchNorm2d(num_features=128, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, bias=False, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=64, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, bias=False, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=64, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, bias=False, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=32, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, bias=False, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=32, momentum=0.01, eps=1e-3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.layer_mean = nn.ConvTranspose2d(32, out_channels, kernel_size=5, stride=2, bias=True, padding=2,
                                             output_padding=1)
        self.layer_var = nn.ConvTranspose2d(32, out_channels, kernel_size=5, stride=2, bias=True, padding=2,
                                            output_padding=1)

    def forward(self, x):
        h1 = self.layer1(x)
        h1_reshape = h1.view(-1, 256, 6, 9)
        h2 = self.layer2(h1_reshape)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        h5 = self.layer5(h4)
        h6 = self.layer6(h5)

        mu = self.layer_mean(h6)
        var = self.layer_var(h6)
        sigma = torch.sqrt(0.1 + 0.9 * F.softplus(var))

        return mu, sigma