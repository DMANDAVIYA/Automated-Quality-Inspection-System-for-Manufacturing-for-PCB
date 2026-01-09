import os, shutil, random, cv2
CLASSES = {1:"open", 2:"short", 3:"mousebite", 4:"spur", 5:"copper", 6:"pinhole"}

def convert_box(sz, b):
    return ((b[0]+b[2])/2.0/sz[0], (b[1]+b[3])/2.0/sz[1], (b[2]-b[0])/sz[0], (b[3]-b[1])/sz[1])

def process():
    if not os.path.exists("data/DeepPCB/PCBData"): return
    [os.makedirs(f"data/yolo_dataset/{d}/{s}", exist_ok=True) for d in ['images','labels'] for s in ['train','val','test']]
    samples = [{'img': os.path.join(r, f), 'lbl': os.path.join(r, f.replace("test.jpg", "not.txt")), 'id': f[:-9]} for r, _, fs in os.walk("data/DeepPCB/PCBData") for f in fs if f.endswith("test.jpg")]
    random.shuffle(samples)
    splits = {'train': samples[:int(len(samples)*0.8)], 'val': samples[int(len(samples)*0.8):int(len(samples)*0.9)], 'test': samples[int(len(samples)*0.9):]}

    for s_name, s_list in splits.items():
        for s in s_list:
            shutil.copy(s['img'], f"data/yolo_dataset/images/{s_name}/{s['id']}.jpg")
            h, w = cv2.imread(s['img']).shape[:2]
            try:
                with open(s['lbl']) as f: lines = [list(map(int, l.strip().split())) for l in f]
                lbls = [f"{l[4]-1} {' '.join(f'{x:.6f}' for x in convert_box((w,h), l[:4]))}" for l in lines if 5 <= len(l) and 1 <= l[4] <= 6]
                with open(f"data/yolo_dataset/labels/{s_name}/{s['id']}.txt", 'w') as f: f.write('\n'.join(lbls))
            except: pass

    with open("data/yolo_dataset/dataset.yaml", "w") as f: f.write(f"train: {os.path.abspath('data/yolo_dataset/images/train')}\nval: {os.path.abspath('data/yolo_dataset/images/val')}\ntest: {os.path.abspath('data/yolo_dataset/images/test')}\nnames:\n  0: open\n  1: short\n  2: mousebite\n  3: spur\n  4: copper\n  5: pinhole")

if __name__ == "__main__": process()
