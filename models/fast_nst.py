import torch
from PIL import Image
import torchvision.transforms as T
import cv2
from moviepy.editor import VideoFileClip
import uuid
import os
import numpy as np
from models.fast_model import TransformerNet
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

def _load_model(model_path, device):
    global _model
    if _model is None:
        _model = TransformerNet().to(device)
        state_dict = torch.load(model_path, map_location=device)

        # Bazı .pth dosyalarında "model_state_dict" gibi farklı anahtarlar olabilir
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        _model.load_state_dict(state_dict)
        _model.eval()
    return _model

def apply_fast_style_transfer(input_image_path, output_image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device)

    transform = T.Compose([
        T.Resize(512),
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255))
    ])

    img = Image.open(input_image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor).cpu()

    out = out.squeeze().clamp(0, 255).numpy()
    out = out.transpose(1, 2, 0).astype('uint8')
    Image.fromarray(out).save(output_image_path)
    return output_image_path

def apply_style_to_video(input_path, output_video_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modeli yükle
    transformer = TransformerNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    transformer.load_state_dict(state_dict)
    transformer.eval()

    # Video yakalama
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Geçici MP4 dosyası
    temp_output_path = output_video_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    # Görüntü dönüşüm
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1, 1, 1])
    ])

    with torch.no_grad():
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tensor = transform(img).unsqueeze(0).to(device)
            output_tensor = transformer(img_tensor).cpu()

            output_tensor = output_tensor.squeeze(0).clamp(0, 255).detach().numpy()
            output_tensor = output_tensor.transpose(1, 2, 0).astype('uint8')
            output_bgr = cv2.cvtColor(output_tensor, cv2.COLOR_RGB2BGR)

            output_bgr = cv2.resize(output_bgr, (width, height))
            out.write(output_bgr)
            frame_count += 1

        print(f"[INFO] İşlenen kare sayısı: {frame_count}")

    cap.release()
    out.release()

    # H.264'e dönüştür
    h264_output_path = output_video_path
    convert_to_h264(temp_output_path, h264_output_path)

    # Geçici dosyayı sil
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    return h264_output_path

def convert_to_h264(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec='libx264', audio=False)