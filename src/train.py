import random
from collections import defaultdict

def subsample_manifest(src_manifest_path, dst_manifest_path, max_per_identity=5, seed=42):
    """Kimlik basina goruntu sayisini sinirlar, kimlik cesitliligini korur."""
    random.seed(seed)

    by_identity = defaultdict(list)
    with open(src_manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_uri, emb_uri = line.split('\t')
            identity = img_uri.split('/')[-2]
            by_identity[identity].append(line)

    n_original = sum(len(v) for v in by_identity.values())
    subsampled_lines = []
    for identity, lines in by_identity.items():
        random.shuffle(lines)
        subsampled_lines.extend(lines[:max_per_identity])

    with open(dst_manifest_path, 'w') as f:
        f.write('\n'.join(subsampled_lines))

    print(f'Orijinal: {n_original:,} ornek, {len(by_identity):,} kimlik')
    print(f'Alt-orneklenmis: {len(subsampled_lines):,} ornek, {len(by_identity):,} kimlik (ayni)')
    return dst_manifest_path

def train(cfg, manifest_path, resume_from=None):
    manifest_path = subsample_manifest(
        src_manifest_path=manifest_path,
        dst_manifest_path='/kaggle/working/manifest_200k.txt',
        max_per_identity=5,
    )
    # ... fonksiyonun geri kalan kodu (varsa) buradan devam eder
