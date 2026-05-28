"""Export dqn_mancala.pt weights to JSON for browser inference."""
import json, torch

model = torch.load("dqn_mancala.pt", map_location="cpu", weights_only=True)

out = {}
for k, v in model.items():
    out[k] = v.tolist()

with open("../web/src/logic/dqn_weights.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))

print("Exported", list(out.keys()))
