from flask import Flask, render_template, request, jsonify
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import base64
from PIL import Image
import io
import os

app = Flask(__name__)

model = None
device = None
features_store = []
gradients_store = []

CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted Surface', 'Rolled-in Scale', 'Scratches']

# Observed focus score distribution from research (used for context indicator)
FOCUS_DIST_MIN = 0.30
FOCUS_DIST_MAX = 0.77
FOCUS_DIST_MEAN = 0.52
FOCUS_DIST_STD = 0.12


def forward_hook(module, input, output):
    features_store.clear()
    features_store.append(output.detach())


def backward_hook(module, grad_input, grad_output):
    gradients_store.clear()
    gradients_store.append(grad_output[0].detach())


def load_model():
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)

    weights_path = os.path.join('model', 'resnet18_neu.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    print(f"Model ready on: {device}")


def preprocess(pil_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return transform(pil_img).unsqueeze(0).to(device)


def compute_gradcam(img_tensor):
    features_store.clear()
    gradients_store.clear()

    with torch.enable_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

        score = output[0, pred_class]
        model.zero_grad()
        score.backward()

    grads = gradients_store[0].cpu().numpy()[0]    # (C, H, W)
    fmap = features_store[0].cpu().numpy()[0]       # (C, H, W)

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, pred_class, confidence


def focus_score(cam, threshold=0.6):
    return float(np.sum(cam > threshold) / cam.size)


def make_overlay(img_tensor, cam):
    img_np = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (img_np * 0.5 + heatmap * 0.5).clip(0, 255).astype(np.uint8)
    return img_np, overlay


def to_b64(img_np):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def focus_label(score):
    if score < 0.40:
        return 'Focused', 'focused'
    elif score < 0.60:
        return 'Moderate', 'moderate'
    else:
        return 'Diffuse', 'diffuse'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    pil_img = Image.open(request.files['image'].stream)
    img_tensor = preprocess(pil_img)

    cam, pred_class, conf = compute_gradcam(img_tensor)
    fscore = focus_score(cam)
    flabel, flabel_cls = focus_label(fscore)

    original_np, overlay_np = make_overlay(img_tensor, cam)

    # Percentile position within observed distribution
    pct = (fscore - FOCUS_DIST_MIN) / (FOCUS_DIST_MAX - FOCUS_DIST_MIN)
    pct = round(float(np.clip(pct, 0, 1)) * 100, 1)

    return jsonify({
        'predicted_class': CLASS_NAMES[pred_class],
        'class_index': pred_class,
        'confidence': round(conf * 100, 3),
        'focus_score': round(fscore, 4),
        'focus_label': flabel,
        'focus_label_class': flabel_cls,
        'focus_percentile': pct,
        'original': to_b64(original_np),
        'overlay': to_b64(overlay_np),
    })


@app.route('/consistency', methods=['POST'])
def consistency():
    files = request.files.getlist('images')
    if len(files) < 2:
        return jsonify({'error': 'Upload at least 2 images'}), 400

    results = []
    for f in files[:6]:
        pil_img = Image.open(f.stream)
        img_tensor = preprocess(pil_img)
        cam, pred_class, conf = compute_gradcam(img_tensor)
        fscore = focus_score(cam)
        original_np, overlay_np = make_overlay(img_tensor, cam)

        results.append({
            'predicted_class': CLASS_NAMES[pred_class],
            'confidence': round(conf * 100, 3),
            'focus_score': round(fscore, 4),
            'original': to_b64(original_np),
            'overlay': to_b64(overlay_np),
        })

    focus_scores = [r['focus_score'] for r in results]
    return jsonify({
        'results': results,
        'variance': round(float(np.var(focus_scores)), 4),
        'std_dev': round(float(np.std(focus_scores)), 4),
        'range': round(float(max(focus_scores) - min(focus_scores)), 4),
    })


if __name__ == '__main__':
    load_model()
    app.run(debug=True)
