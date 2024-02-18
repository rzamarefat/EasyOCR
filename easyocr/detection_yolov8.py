from ultralytics import YOLO

def get_detector(trained_model, device='cpu', *args, **kwargs):
    trained_model = r"C:\Users\ASUS\Desktop\github_projects\EasyOCR\best.pt"
    net = YOLO(trained_model)
    net.to(device)

    return net

def get_textbox(detector, image, text_threshold, **kwargs):
    results = []
    
    res = detector.predict(image, conf=text_threshold)

    print("from YOLO predictions")
    print(res)
    exit()

