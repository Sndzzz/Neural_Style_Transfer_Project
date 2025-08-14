import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy
import os

from torchvision.models import vgg19, VGG19_Weights

# Görsel boyutu
imsize = 512 if torch.cuda.is_available() else 256

# Dönüştürme işlemi
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def image_loader(image_path):
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(torch.float).to(device)

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# İçerik ve stil loss tanımları
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Normalizasyon modülü
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1,1,1).to(device)
        self.std = torch.tensor(std).view(-1,1,1).to(device)
    def forward(self, x):
        return (x - self.mean) / self.std

# VGG19 modeli üzerinde loss’lar eklenmiş yeni ağ oluştur
def get_model_and_losses(cnn, normalization_mean, normalization_std,
                         style_img, content_img,
                         content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):

    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Gereksiz katmanları kes
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i+1]

    return model, style_losses, content_losses

# Stil aktarımı fonksiyonu
def run_classic_nst(content_path, style_path, output_path="output.png", num_steps=300, style_weight=1e6, content_weight=1):

    content_img = image_loader(content_path)
    style_img = image_loader(style_path)
    input_img = content_img.clone()

    # ✅ Güncel model yükleme
    weights = VGG19_Weights.DEFAULT
    cnn = vgg19(weights=weights).features.to(device).eval()

    cnn_normalization_mean = [0.485, 0.456, 0.406]
    cnn_normalization_std = [0.229, 0.224, 0.225]

    model, style_losses, content_losses = get_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        style_img, content_img
    )

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    run = [0]

    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight * style_score + content_weight * content_score
            loss.backward()
            run[0] += 1

            if run[0] % 50 == 0:
                print(f"Step {run[0]} | Style Loss: {style_score.item():.2f} | Content Loss: {content_score.item():.2f}")
            return loss
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    output_img = input_img.squeeze().cpu().detach()
    output_img = transforms.ToPILImage()(output_img)
    output_img.save(output_path)
    return output_path