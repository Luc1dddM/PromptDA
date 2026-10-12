"""Reproducible scene-balanced subsets and experiment frame records."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def scene_subset(samples, limit, seed):
    """Equal scene quotas (redistributing shortages), stratified over frame order."""
    groups = defaultdict(list)
    for sample in samples:
        groups[sample[0].split('/')[0]].append(sample)
    scenes = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(scenes)
    quotas = dict.fromkeys(scenes, 0)
    remaining = min(limit, len(samples))
    while remaining:
        for scene in scenes:
            if quotas[scene] < len(groups[scene]):
                quotas[scene] += 1
                remaining -= 1
                if not remaining:
                    break
    selected = []
    for scene in sorted(scenes):
        frames = sorted(groups[scene], key=lambda sample: sample[0])
        count = quotas[scene]
        for index in range(count):
            lo = index * len(frames) // count
            hi = (index + 1) * len(frames) // count
            selected.append(frames[rng.randrange(lo, hi)])
    return selected


def save_manifest(dataset, path):
    if hasattr(dataset, 'dataset_folder'):  # ARKitScenes (video, filename, direction)
        identifiers = [f'{sample[0]}/{sample[1]}' for sample in dataset.samples]
    else:
        identifiers = [sample[0] for sample in dataset.samples]
    manifest = {
        'frame_count': len(identifiers),
        'scene_count': len({identifier.split('/')[0] for identifier in identifiers}),
        'sha256': hashlib.sha256('\n'.join(identifiers).encode()).hexdigest(),
        'identifiers': identifiers,
        'skipped_scenes': getattr(dataset, 'skipped_scenes', []),
    }
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f'Frame selection changed for existing experiment: {path}. Use a new run/output directory.')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
