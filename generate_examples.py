import os, cv2, glob
from inspection import PCBAligner, PCBDetector
from sahi.utils.cv import visualize_object_predictions

os.makedirs("examples/annotated", exist_ok=True)
detector = PCBDetector('output/pcb_defect_v8/weights/best.pt')
aligner = PCBAligner()

for defective in sorted(glob.glob("examples/defective/*.jpg")):
    name = os.path.basename(defective).replace("_defective.jpg", "")
    
    img = cv2.imread(defective)
    res = detector.detect(img)
    
    annotated = visualize_object_predictions(img, res.object_prediction_list)['image']
    cv2.imwrite(f"examples/annotated/{name}_annotated.jpg", annotated)
    
    print(f"Processed {name}: {len(res.object_prediction_list)} defects detected")

print("\nAnnotated images saved to examples/annotated/")
