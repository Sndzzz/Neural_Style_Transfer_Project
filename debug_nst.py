import os
from models.classic_nst import run_classic_nst

# Dosya yollarını kendi kullandığınla değiştir
content_image = os.path.join("app", "uploads", "content_manzara.jpg")
style_image   = os.path.join("style_lib", "Vincent VanGogh", "Starry_Night.jpeg")
output_image  = os.path.join("app", "outputs", "debug_test.jpg")

# Gerekli dizini oluştur
os.makedirs(os.path.dirname(output_image), exist_ok=True)

print(f"[DEBUG] content path: {content_image}")
print(f"[DEBUG] style path: {style_image}")
print(f"[DEBUG] output path: {output_image}")

# Stil aktarımını çalıştır
result = run_classic_nst(content_image, style_image, output_image)

print("Çıktı kaydedildi:", result)
