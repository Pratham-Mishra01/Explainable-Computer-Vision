# Steel Surface Defect Analysis — Demo

Explainable CV demo for ResNet-18 + Grad-CAM on the NEU Steel Defect Dataset.

## Project Structure

```
steel-defect-demo/
├── app.py                  # Flask backend
├── requirements.txt
├── model/
│   └── resnet18_neu.pth    # ← place your saved weights here
└── templates/
    └── index.html          # Frontend
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model weights
# Copy resnet18_neu.pth into the model/ folder

# 4. Run
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Features

### Classify & Explain
- Upload any steel surface image
- Returns predicted defect class + confidence
- Grad-CAM overlay showing model attention
- Focus Score with position on observed distribution (range 0.30–0.77, n=50)

### Consistency Explorer
- Upload 2–6 images (same class recommended)
- Side-by-side Grad-CAM comparison
- Standard deviation / variance of Focus Scores across images
- Quantifies explanation instability

## What Is NOT Included (and Why)

**Confidence vs. Focus Score correlation** is excluded.  
Confidence scores on this dataset cluster in the range 0.9997–1.0000 (variance < 0.001).  
Correlating a near-constant with a variable produces statistically meaningless results.  
The -0.183 correlation observed in research is noise, not a finding.  
See Section 9 of the research paper for full discussion.

## Defect Classes

| Index | Class           |
|-------|-----------------|
| 0     | Crazing         |
| 1     | Inclusion       |
| 2     | Patches         |
| 3     | Pitted Surface  |
| 4     | Rolled-in Scale |
| 5     | Scratches       |

## Technical Notes

- Grad-CAM hooks attached to `model.layer4[-1]`
- Uses `register_full_backward_hook` (no deprecation warning)
- Focus Score threshold: 0.6 (active pixels / total pixels)
- Model expects 224×224 RGB input; grayscale images are auto-converted
