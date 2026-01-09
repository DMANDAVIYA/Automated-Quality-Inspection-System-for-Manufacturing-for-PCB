import os, cv2, random, glob
from inspection import PCBAligner, PCBDetector
from sahi.utils.cv import visualize_object_predictions

def main():
    aligner, detector = PCBAligner(), PCBDetector('output/pcb_defect_v8/weights/best.pt' if os.path.exists('output/pcb_defect_v8/weights/best.pt') else 'yolov8n.pt')
    if not (tests := glob.glob("data/DeepPCB/PCBData/**/*_test.jpg", recursive=True)): return
    
    test_path = random.choice(tests)
    print(f"Processing: {test_path}")
    
    try:
        aligned = aligner.align(cv2.imread(test_path.replace("_test.jpg", "_temp.jpg")), cv2.imread(test_path))
        res = detector.detect(aligned)
        cv2.imwrite("output/final_result.jpg", visualize_object_predictions(aligned, res.object_prediction_list)['image'])
        print("Draft saved to output/final_result.jpg")
    except Exception as e: print(f"Error: {e}")

if __name__ == "__main__": 
    os.makedirs("output", exist_ok=True)
    main()
