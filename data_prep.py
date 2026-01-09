import os, shutil, random, cv2

def convert_box(sz, b):
    return ((b[0]+b[2])/2.0/sz[0], (b[1]+b[3])/2.0/sz[1], (b[2]-b[0])/sz[0], (b[3]-b[1])/sz[1])

def process():
    base = "data/DeepPCB/PCBData"
    if not os.path.exists(base): return
    
    [os.makedirs(f"data/yolo_dataset/{d}/{s}", exist_ok=True) for d in ['images','labels'] for s in ['train','val','test']]
    
    samples = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith("_test.jpg"):
                test_path = os.path.join(root, f)
                base_id = f[:-9]
                group_id = base_id[:5]
                annot_path = os.path.join(os.path.dirname(root), f"{group_id}_not", f"{base_id}.txt")
                if os.path.exists(annot_path):
                    samples.append({'test': test_path, 'annot': annot_path, 'id': base_id})
    
    print(f"Found {len(samples)} samples")
    random.shuffle(samples)
    splits = {'train': samples[:int(len(samples)*0.8)], 'val': samples[int(len(samples)*0.8):int(len(samples)*0.9)], 'test': samples[int(len(samples)*0.9):]}
    
    for s_name, s_list in splits.items():
        print(f"Processing {s_name}: {len(s_list)} samples")
        for s in s_list:
            shutil.copy(s['test'], f"data/yolo_dataset/images/{s_name}/{s['id']}.jpg")
            
            h, w = cv2.imread(s['test']).shape[:2]
            with open(s['annot']) as f: lines = [list(map(int, l.strip().split())) for l in f if l.strip()]
            lbls = [f"{l[4]-1} {' '.join(f'{x:.6f}' for x in convert_box((w,h), l[:4]))}" for l in lines if len(l) >= 5 and 1 <= l[4] <= 6]
            with open(f"data/yolo_dataset/labels/{s_name}/{s['id']}.txt", 'w') as f: f.write('\n'.join(lbls))
    
    with open("data/yolo_dataset/dataset.yaml", "w") as f: f.write(f"train: {os.path.abspath('data/yolo_dataset/images/train')}\nval: {os.path.abspath('data/yolo_dataset/images/val')}\ntest: {os.path.abspath('data/yolo_dataset/images/test')}\nnames:\n  0: open\n  1: short\n  2: mousebite\n  3: spur\n  4: copper\n  5: pinhole")
    
    print("Dataset preparation complete")

if __name__ == "__main__": process()
