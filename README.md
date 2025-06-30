
# Player Re-Identification Project

This repository implements **single-camera, long-term player re-identification** for sports footage using:

- provided **YOLOv11** model for detection (`model.pt`)
- **Supervision's BYTETrack** for short-term tracking
- **ResNet50 (pretrained)** for feature embeddings and long-term re-ID
- **Cosine similarity** for matching lost-and-found tracks

## Structure

```
player_reid_project/
├── main.py
├── detector.py
├── tracker.py
├── reid_model.py
├── utils.py
├── config.yaml
└── requirements.txt
```

## Setup

1. **Clone** or unzip this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the provided model **YOLOv11 model** (`model.pt`) in the project root.
4. Ensure `config.yaml` paths point to your video and model.

## Usage

```bash
python main.py
```

This will:
- Read `config.yaml` for model paths and thresholds
- Run detection → short-term track → long-term re-ID
- Save annotated output video as `data/output_video.avi`


## 🧪 Tracking Logic

- Detection: Uses provided model to detect persons in each frame.
- Embedding: ReID model extracts 128-d features for each person.
- Matching:
  - Cosine similarity is used to match embeddings frame-to-frame.
  - IOU matching for backup when embedding is ambiguous.
- ID Persistence: Track IDs are preserved through short occlusions.
