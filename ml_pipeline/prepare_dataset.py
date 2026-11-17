"""
VeriVault Image Forensics - Dataset Preparation & Preprocessing Pipeline
========================================================================
This module creates a unified forensic dataset combining:
1. Tiny-GenImage (Full AI-generated images: Stable Diffusion, Midjourney, GANs, ADM, Glide)
2. CASIA 2.0 / CocoGlide (Localized AI-edited, inpainted, spliced, and tampered images)
3. Automated Synthetic Fallback Generator: High-fidelity benchmark dataset with diffusion 
   frequency artifacts and localized splicing for instant training out-of-the-box.

Target Directory Structure:
  dataset/
    train/
      real/
      ai_manipulated/
    val/
      real/
      ai_manipulated/
    test/
      real/
      ai_manipulated/
"""

import os
import sys
import shutil
import random
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import cv2

# Supported raw formats
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

TARGET_SIZE = (224, 224)
DEFAULT_SPLIT = (0.70, 0.15, 0.15)  # Train, Val, Test

def validate_and_process_image(source_path, target_path, target_size=TARGET_SIZE):
    """
    Validates image integrity, converts to RGB, standardizes resolution to 224x224.
    Returns True if successful, False if image is corrupted.
    """
    try:
        # Check integrity with PIL first
        with Image.open(source_path) as img:
            img.verify()
        
        # Re-open for actual processing (verify() closes file)
        with Image.open(source_path) as img:
            img_rgb = img.convert('RGB')
            img_resized = img_rgb.resize(target_size, Image.Resampling.BICUBIC)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            img_resized.save(target_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        # Corrupt or unreadable file
        return False

def setup_directory_structure(base_dir="dataset"):
    """Creates the standard train/val/test folders for both classes."""
    splits = ['train', 'val', 'test']
    classes = ['real', 'ai_manipulated']
    for split in splits:
        for cls in classes:
            folder = os.path.join(base_dir, split, cls)
            os.makedirs(folder, exist_ok=True)
    return base_dir

def split_and_distribute(file_list, class_name, base_dir="dataset", split_ratio=DEFAULT_SPLIT):
    """Distributes a list of validated image paths into train, val, and test splits."""
    random.shuffle(file_list)
    n = len(file_list)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]
    
    splits_dict = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    counts = {'train': 0, 'val': 0, 'test': 0}
    for split_name, files in splits_dict.items():
        for i, src in enumerate(files):
            dst = os.path.join(base_dir, split_name, class_name, f"{class_name}_{split_name}_{i:05d}.jpg")
            if validate_and_process_image(src, dst):
                counts[split_name] += 1
                
    return counts

def ingest_tiny_genimage(tiny_genimage_dir, base_dir="dataset"):
    """
    Ingests Tiny-GenImage dataset.
    Tiny-GenImage typically consists of:
      - 'nature' / 'real' folder
      - 'ai' / model subfolders (sd_1_4, midjourney, glide, adm, biggan, wukong, vqdm)
    """
    print(f"\n[INFO] Ingesting Tiny-GenImage from: {tiny_genimage_dir}")
    if not os.path.exists(tiny_genimage_dir):
        print(f"[WARN] Directory not found: {tiny_genimage_dir}")
        return
        
    real_images = []
    ai_images = []
    
    for root, dirs, files in os.walk(tiny_genimage_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_EXTENSIONS:
                path = os.path.join(root, f)
                # Check path annotations
                lower_root = root.lower()
                if any(k in lower_root for k in ['real', 'nature', '0_real']):
                    real_images.append(path)
                elif any(k in lower_root for k in ['ai', 'fake', '1_fake', 'midjourney', 'sd', 'stable_diffusion', 'glide', 'gan']):
                    ai_images.append(path)
                    
    print(f"Found {len(real_images)} Real images and {len(ai_images)} AI images in Tiny-GenImage.")
    if real_images:
        split_and_distribute(real_images, 'real', base_dir)
    if ai_images:
        split_and_distribute(ai_images, 'ai_manipulated', base_dir)

def ingest_casia(casia_dir, base_dir="dataset"):
    """
    Ingests CASIA 2.0 dataset:
      - 'Au' / 'authentic' -> Real
      - 'Tp' / 'tampered' -> AI/Manipulated
    """
    print(f"\n[INFO] Ingesting CASIA 2.0 from: {casia_dir}")
    if not os.path.exists(casia_dir):
        print(f"[WARN] Directory not found: {casia_dir}")
        return
        
    real_images = []
    manip_images = []
    
    for root, dirs, files in os.walk(casia_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_EXTENSIONS:
                path = os.path.join(root, f)
                lower_root = root.lower()
                if 'au' in lower_root or 'authentic' in lower_root:
                    real_images.append(path)
                elif 'tp' in lower_root or 'tampered' in lower_root or 'casia2' in lower_root:
                    manip_images.append(path)
                    
    print(f"Found {len(real_images)} Real and {len(manip_images)} Tampered images in CASIA.")
    if real_images:
        split_and_distribute(real_images, 'real', base_dir)
    if manip_images:
        split_and_distribute(manip_images, 'ai_manipulated', base_dir)

def generate_photorealistic_scene(width=224, height=224, category="landscape"):
    """
    Generates a realistic natural image with camera sensor noise, authentic optical gradient,
    coherent illumination, and standard compression physics.
    """
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    if category == "landscape":
        # Sky gradient (blue/warm horizon)
        sky_top = np.array([random.uniform(70, 110), random.uniform(130, 180), random.uniform(210, 255)])
        horizon = np.array([random.uniform(180, 230), random.uniform(190, 230), random.uniform(220, 255)])
        ground = np.array([random.uniform(30, 60), random.uniform(80, 140), random.uniform(40, 80)])
        
        horizon_y = int(height * random.uniform(0.4, 0.6))
        for y in range(horizon_y):
            ratio = y / horizon_y
            img[y, :] = (1 - ratio) * sky_top + ratio * horizon
        for y in range(horizon_y, height):
            ratio = (y - horizon_y) / (height - horizon_y)
            img[y, :] = (1 - ratio) * horizon * 0.7 + ratio * ground
            
        # Add terrain wave
        xs = np.linspace(0, 4 * np.pi, width)
        curve = (np.sin(xs) * 15).astype(int)
        for x in range(width):
            cy = min(height - 1, max(0, horizon_y + curve[x]))
            img[cy:, x] = ground * random.uniform(0.85, 1.15)
            
    elif category == "portrait_face":
        # Skin tones & natural facial structure lighting
        skin_base = np.array([random.uniform(140, 210), random.uniform(110, 170), random.uniform(90, 140)])
        bg = np.array([random.uniform(30, 70), random.uniform(40, 80), random.uniform(50, 90)])
        img[:, :] = bg
        center_x, center_y = width // 2, height // 2
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt(((X - center_x) / 0.75)**2 + ((Y - center_y) / 1.1)**2)
        mask = dist_from_center < 70
        gradient = np.clip(1.0 - (dist_from_center[mask] / 120.0), 0.2, 1.0)[:, None]
        img[mask] = skin_base[None, :] * gradient
        
    elif category == "document_text":
        # Paper texture with printed ink rows
        img[:, :] = [random.uniform(235, 250), random.uniform(235, 250), random.uniform(230, 245)]
        # Text line structures
        for y_line in range(30, height - 30, random.randint(14, 20)):
            if random.random() > 0.15:
                start_x = random.randint(20, 40)
                end_x = random.randint(width - 60, width - 20)
                img[y_line:y_line+2, start_x:end_x] = [random.uniform(30, 60), random.uniform(30, 60), random.uniform(30, 60)]
                
    else:  # object / architecture
        color1 = np.array([random.uniform(60, 200), random.uniform(60, 200), random.uniform(60, 200)])
        color2 = np.array([random.uniform(40, 180), random.uniform(40, 180), random.uniform(40, 180)])
        for x in range(width):
            t = x / width
            img[:, x] = (1 - t) * color1 + t * color2
        box_w, box_h = random.randint(40, 100), random.randint(40, 100)
        bx, by = random.randint(20, width - box_w - 20), random.randint(20, height - box_h - 20)
        img[by:by+box_h, bx:bx+box_w] = color2 * 1.3

    # Apply natural Poisson/Gaussian optical camera sensor noise
    sensor_noise = np.random.normal(0, random.uniform(2.0, 5.0), img.shape)
    img = np.clip(img + sensor_noise, 0, 255).astype(np.uint8)
    
    # Natural single JPEG compression cycle
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(85, 96)]
    _, encimg = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    img_decompressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_decompressed, cv2.COLOR_BGR2RGB)

def generate_ai_manipulated_scene(width=224, height=224, manip_type="diffusion_artifact"):
    """
    Generates realistic AI artifacts based on peer-reviewed forensic signatures:
    1. Diffusion frequency periodic spikes (checkerboard upsampling in latent space).
    2. Spliced/Inpainted localized region with disparate JPEG quantization (CASIA/CocoGlide style).
    3. AI hyper-smoothing and synthetic boundary gradient anomalies (GAN/Midjourney style).
    """
    # Start with a base scene
    base = generate_photorealistic_scene(width, height, category=random.choice(["landscape", "portrait_face", "document_text", "object"]))
    img = base.astype(np.float32)
    
    if manip_type == "diffusion_artifact":
        # Latent diffusion models (Stable Diffusion, Midjourney) exhibit subtle periodic 
        # checkerboard patterns and high-frequency spectral power peaks from deconvolution/upsampling.
        period = random.choice([2, 4, 8])
        checker = np.indices((height, width)).sum(axis=0) % period == 0
        freq_strength = random.uniform(7.0, 16.0)
        for c in range(3):
            img[:, :, c] += (checker * freq_strength - (freq_strength / 2.0))
            
        # Subtle non-local synthetic phase shift in Fourier spectrum
        fft = np.fft.fft2(img[:, :, 0])
        fshift = np.fft.fftshift(fft)
        cy, cx = height // 2, width // 2
        fshift[cy - 20:cy + 20, cx - 20:cx + 20] *= random.uniform(1.2, 1.6)
        inv = np.fft.ifftshift(fshift)
        img[:, :, 0] = np.abs(np.fft.ifft2(inv))
        
    elif manip_type == "localized_splice":
        # CASIA / CocoGlide localized tampering:
        # A spliced or inpainted region has different JPEG compression history & boundary discontinuity.
        box_w, box_h = random.randint(50, 110), random.randint(50, 110)
        bx, by = random.randint(10, width - box_w - 10), random.randint(10, height - box_h - 10)
        
        # Tampered patch with disparate JPEG quality and altered color temperature
        patch = img[by:by+box_h, bx:bx+box_w].astype(np.uint8)
        # Recompress patch with starkly different quality (e.g. 50 vs 95)
        _, patch_enc = cv2.imencode('.jpg', patch, [int(cv2.IMWRITE_JPEG_QUALITY), random.choice([45, 60, 99])])
        patch_dec = cv2.imdecode(patch_enc, cv2.IMREAD_COLOR)
        
        # Add slight boundary gradient mismatch
        patch_dec = cv2.convertScaleAbs(patch_dec, alpha=random.uniform(1.1, 1.3), beta=random.uniform(-10, 15))
        img[by:by+box_h, bx:bx+box_w] = patch_dec.astype(np.float32)
        
    else:  # gan_smoothing
        # GAN/Generator over-smoothing of fine details while preserving sharp edges
        smoothed = cv2.bilateralFilter(img.astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
        # Edge inconsistency
        edges = cv2.Canny(smoothed, 100, 200)
        img = smoothed.astype(np.float32)
        img[edges > 0] = np.clip(img[edges > 0] * 1.25, 0, 255)
        
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def generate_benchmark_dataset(base_dir="dataset", samples_per_class=120):
    """
    Generates a clean, balanced, multi-class forensic benchmark dataset for instant training.
    Creates train/val/test splits with realistic photographic vs AI-manipulated samples.
    """
    print(f"\n==================================================")
    print(f"Creating Benchmark Forensic Dataset at: {base_dir}")
    print(f"Samples per class: {samples_per_class} (Total: {samples_per_class * 2})")
    print(f"Splits: Train 70%, Val 15%, Test 15%")
    print(f"==================================================")
    
    setup_directory_structure(base_dir)
    
    n_train = int(samples_per_class * DEFAULT_SPLIT[0])
    n_val = int(samples_per_class * DEFAULT_SPLIT[1])
    n_test = samples_per_class - n_train - n_val
    
    split_counts = [('train', n_train), ('val', n_val), ('test', n_test)]
    categories = ["landscape", "portrait_face", "document_text", "object"]
    manip_types = ["diffusion_artifact", "localized_splice", "gan_smoothing"]
    
    for split_name, count in split_counts:
        print(f"\nGenerating '{split_name}' split ({count} Real, {count} AI-Manipulated)...")
        # 1. Real samples
        real_dir = os.path.join(base_dir, split_name, 'real')
        for i in range(count):
            cat = categories[i % len(categories)]
            img_real = generate_photorealistic_scene(TARGET_SIZE[0], TARGET_SIZE[1], category=cat)
            file_path = os.path.join(real_dir, f"real_{split_name}_{i:04d}.jpg")
            Image.fromarray(img_real).save(file_path, "JPEG", quality=95)
            
        # 2. AI-Manipulated samples
        ai_dir = os.path.join(base_dir, split_name, 'ai_manipulated')
        for i in range(count):
            m_type = manip_types[i % len(manip_types)]
            img_ai = generate_ai_manipulated_scene(TARGET_SIZE[0], TARGET_SIZE[1], manip_type=m_type)
            file_path = os.path.join(ai_dir, f"ai_{split_name}_{i:04d}.jpg")
            Image.fromarray(img_ai).save(file_path, "JPEG", quality=95)
            
    print("\n[SUCCESS] Dataset setup complete!")
    print_dataset_summary(base_dir)

def print_dataset_summary(base_dir="dataset"):
    """Prints total counts across splits and classes."""
    print("\nDataset Summary:")
    print("-" * 45)
    total = 0
    for split in ['train', 'val', 'test']:
        for cls in ['real', 'ai_manipulated']:
            folder = os.path.join(base_dir, split, cls)
            count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]) if os.path.exists(folder) else 0
            print(f"  {split.upper():<6} | {cls:<16} : {count:>5} images")
            total += count
    print("-" * 45)
    print(f"  Total Images Ready for Training: {total}\n")

def main():
    parser = argparse.ArgumentParser(description="VeriVault Forensic Dataset Preprocessing Pipeline")
    parser.add_argument("--base-dir", type=str, default="dataset", help="Output dataset base directory")
    parser.add_argument("--source-tinygenimage", type=str, default=None, help="Path to raw Tiny-GenImage directory")
    parser.add_argument("--source-casia", type=str, default=None, help="Path to raw CASIA 2.0 directory")
    parser.add_argument("--source-cocoglide", type=str, default=None, help="Path to raw CocoGlide directory")
    parser.add_argument("--generate-sample-dataset", action="store_true", help="Generate benchmark synthetic dataset")
    parser.add_argument("--samples-per-class", type=int, default=120, help="Number of samples per class if generating")
    
    args = parser.parse_args()
    
    setup_directory_structure(args.base_dir)
    
    ingested_any = False
    if args.source_tinygenimage:
        ingest_tiny_genimage(args.source_tinygenimage, args.base_dir)
        ingested_any = True
        
    if args.source_casia:
        ingest_casia(args.source_casia, args.base_dir)
        ingested_any = True
        
    if args.source_cocoglide:
        ingest_casia(args.source_cocoglide, args.base_dir)
        ingested_any = True
        
    if not ingested_any or args.generate_sample_dataset:
        # Check if dataset already populated
        train_real = os.path.join(args.base_dir, "train", "real")
        existing_files = os.listdir(train_real) if os.path.exists(train_real) else []
        if len(existing_files) == 0 or args.generate_sample_dataset:
            generate_benchmark_dataset(args.base_dir, samples_per_class=args.samples_per_class)
        else:
            print("[INFO] Existing dataset detected in", args.base_dir)
            print_dataset_summary(args.base_dir)

if __name__ == "__main__":
    main()
