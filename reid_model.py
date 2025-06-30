
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity

class ReIDMatcher:
    def __init__(self, similarity_threshold=0.85):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Pretrained ResNet50 backbone
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.threshold = similarity_threshold

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.memory = {}  # track_id -> (feat, last_seen)

    def extract_feature(self, img):
        with torch.no_grad():
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            feat = self.model(tensor)
        return feat.squeeze().cpu().numpy()

    def match(self, new_feat, frame_id):
        best_id, best_score = None, 0
        for tid, (feat, last_seen) in self.memory.items():
            sim = cosine_similarity([new_feat], [feat])[0][0]
            if sim > best_score and sim >= self.threshold:
                best_id, best_score = tid, sim
        return best_id

    def update_memory(self, track_id, feat, frame_id):
        self.memory[track_id] = (feat, frame_id)
