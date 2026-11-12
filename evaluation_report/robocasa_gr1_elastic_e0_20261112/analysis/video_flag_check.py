# Read-only: for the two validator-flagged task cells, list the videos in completion order with their final-step flag,
# next to the official per-episode JSON successes, and report the episodes where the flag is 1 but the JSON success is false.
import glob, json, os, struct
B = "/mnt/data/vmo-ai-task/anhdh35/elastic_gr1/eval"


def frames(path):
    """Number of video samples from the mp4 'stsz' box (no external tool needed)."""
    with open(path, "rb") as f:
        data = f.read()
    i = -1
    while True:
        i = data.find(b"stsz", i + 1)
        if i < 0:
            return None
        version_flags, _size, count = struct.unpack(">III", data[i + 4:i + 16])
        if version_flags == 0 and 0 < count < 100000:
            return count


for ev, task in (("e01_flat_k1_n2", "PnPMilkToMicrowaveClose"), ("e05_flat_k1_n5", "PosttrainPnPNovelFromPlateToCardboardboxSplitA")):
    t = task + "_GR1ArmsAndWaistFourierHands_Env"
    js = json.load(open(f"{B}/{ev}/n12/{t}.json"))["successes"]
    vids = sorted(glob.glob(f"{B}/{ev}/videos/n12/{t}/*.mp4"), key=os.path.getmtime)
    flags = [v.endswith("_success1.mp4") for v in vids]
    fr = [frames(v) for v in vids]
    print(ev, task, "videos", len(vids), "json successes", sum(js), "success1 videos", sum(flags))
    print("  frames per video (set over the 50 videos):", sorted(set(fr)))
    for i, (s, fl, v, n) in enumerate(zip(js, flags, vids, fr)):
        if fl and not s:
            print(f"  episode index {i} (0-based): JSON success False, video flag 1, frames {n}, bytes {os.path.getsize(v)}, file {os.path.basename(v)}")
    print("  JSON successes whose video flag is 0 (success reached, not held to the last rendered step):", sum(1 for s, fl in zip(js, flags) if s and not fl))
