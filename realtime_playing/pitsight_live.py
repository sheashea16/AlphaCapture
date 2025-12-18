import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys, os
sys.path.append(os.path.abspath(".."))
from alphacapture import AlphaCapture

# pit bounding boxes
PIT_BOXES = [
    (103, 252, 161, 311),  # Pit 0
    (165, 252, 226, 310),  # Pit 1
    (229, 253, 290, 312),  # Pit 2
    (329, 252, 390, 312),  # Pit 3
    (392, 252, 454, 312),  # Pit 4
    (456, 251, 519, 313),  # Pit 5
    (510, 185, 584, 313),  # Store 0
    (450, 185, 503, 235),  # Pit 7
    (390, 183, 445, 237),  # Pit 8
    (330, 184, 383, 237),  # Pit 9
    (238, 185, 294, 237),  # Pit 10
    (177, 184, 232, 238),  # Pit 11
    (116, 183, 174, 238),  # Pit 12
    (40, 184, 108, 310),   # Store 1
]

# PitSight model class
class PitSight(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
      x = self.features(x)
      x = x.view(x.size(0), -1)
      return self.regressor(x).squeeze(1)

# configurations
CAM_INDEX = 1
MODEL_PATH = "pitsight_v7.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLAYABLE_PITS = [0,1,2,3,4,5,7,8,9,10,11,12]

# models
pitsight = PitSight()
pitsight.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
pitsight.to(DEVICE)
pitsight.eval()

alphacapture = AlphaCapture(max_player=0, depth=8)

# transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# capture setup
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("\n🎥 Webcam ready")
print("➡️ Press SPACE to analyze board")
print("❌ Press ESC to quit\n")

# draw pits and counts

display_frame = None

def draw_pits(frame, pit_boxes, pit_counts):
    overlay = frame.copy()

    for pit_idx, (x1, y1, x2, y2) in enumerate(pit_boxes):
        # draw box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # draw count if playable pit
        if pit_idx in pit_counts:
            text = str(pit_counts[pit_idx])
            cv2.putText(
                overlay,
                text,
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )
    return overlay

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # If we have an annotated frame, show it; otherwise show live feed
    if display_frame is None:
        cv2.imshow("AlphaCapture Activated — Press SPACE", frame)
    else:
        cv2.imshow("AlphaCapture Activated — Press SPACE", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # exit condition
    if key == 27:  # ESC
        break

    # analyze board
    if key == 32:  # SPACE
        print("📸 Captured frame, analyzing...")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pit_counts = {}

        for pit_idx in PLAYABLE_PITS:
            x1, y1, x2, y2 = PIT_BOXES[pit_idx]
            crop = frame_rgb[y1:y2, x1:x2]

            inp = transform(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                raw_pred = pitsight(inp).item()
                raw_pred = max(0, min(10, raw_pred))   # clamp
                pred = int(round(raw_pred))

            pit_counts[pit_idx] = pred

        # draw annotated frame
        display_frame = frame.copy()

        for pit_idx, (x1, y1, x2, y2) in enumerate(PIT_BOXES):
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if pit_idx in pit_counts:
                cv2.putText(
                    display_frame,
                    str(pit_counts[pit_idx]),
                    (x1 + 5, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

        # board state
        board = [0] * 14
        for idx, count in pit_counts.items():
            board[idx] = count

        board[6] = 0
        board[13] = 0
        state = (board, 0)

        best_sequence = alphacapture.best_sequence(state)

        print("\n==============================")
        print("Board:", board)
        print("Best Sequence:", best_sequence)
        print("==============================\n")

# clean up
cap.release()
cv2.destroyAllWindows()
print("👋 Session ended")
