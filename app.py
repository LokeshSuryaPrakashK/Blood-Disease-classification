from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import base64
import io

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN_MODEL_PATH = "best_disease_model.pth"
LEUKEMIA_MODEL_PATH = "best_lukemia_model.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

disease_info = {
    "leukemia": {
        "RBC": "Reduced RBC count may be observed.",
        "WBC": "Abnormal proliferation of immature white blood cells.",
        "Platelets": "Platelet count often decreased.",
        "Summary": "Cancer of blood-forming tissues affecting WBC production.",
        "Remedies": "Chemotherapy, targeted therapy, bone marrow transplant, regular blood monitoring."
    },
    "malaria": {
        "RBC": "Parasitic ring forms inside RBC.",
        "WBC": "Usually normal morphology.",
        "Platelets": "Platelet count often reduced.",
        "Summary": "Plasmodium parasite infection affecting red blood cells.",
        "Remedies": "Antimalarial medications (ACT), hydration, mosquito prevention."
    },
    "sickle": {
        "RBC": "Sickle-shaped red blood cells.",
        "WBC": "Typically normal.",
        "Platelets": "May be slightly elevated.",
        "Summary": "Genetic disorder causing abnormal hemoglobin.",
        "Remedies": "Hydroxyurea therapy, pain management, blood transfusions."
    },
    "anemia": {
        "RBC": "Reduced RBC count and pale cells.",
        "WBC": "Usually normal.",
        "Platelets": "Typically normal.",
        "Summary": "Condition characterized by low hemoglobin levels.",
        "Remedies": "Iron supplements, vitamin B12/folate, dietary improvements."
    },
    "thalassemia": {
        "RBC": "Microcytic and target-shaped cells.",
        "WBC": "Usually normal.",
        "Platelets": "Normal or slightly altered.",
        "Summary": "Inherited blood disorder affecting hemoglobin synthesis.",
        "Remedies": "Regular transfusions, iron chelation therapy, bone marrow transplant."
    },
    "normal": {
        "RBC": "Round biconcave cells with central pallor.",
        "WBC": "Normal morphology and distribution.",
        "Platelets": "Normal distribution.",
        "Summary": "No abnormal morphological findings detected.",
        "Remedies": "Maintain balanced diet and regular health checkups."
    }
}

checkpoint = torch.load(MAIN_MODEL_PATH, map_location=DEVICE)
CLASS_NAMES = checkpoint["class_names"]
NUM_CLASSES = checkpoint["num_classes"]

disease_model = models.efficientnet_b3(weights=None)
in_features = disease_model.classifier[1].in_features

disease_model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)

disease_model.load_state_dict(checkpoint["model_state"])
disease_model.to(DEVICE)
disease_model.eval()

def load_leukemia_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]

    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


leukemia_model, LEUKEMIA_SUBCLASSES = load_leukemia_model(LEUKEMIA_MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def leukemia_predict(image):
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = leukemia_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item()

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["file"]

        image = Image.open(file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = disease_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        disease_name = CLASS_NAMES[predicted.item()].lower()
        confidence_percent = round(confidence.item() * 100, 2)

        leukemia_stage = None
        leukemia_conf = None

        if disease_name == "leukemia":
            sub_pred, sub_conf = leukemia_predict(image)
            leukemia_stage = LEUKEMIA_SUBCLASSES[sub_pred]
            leukemia_conf = round(sub_conf * 100, 2)

        info = disease_info.get(disease_name, {})

        target_layers = [disease_model.features[-1]]
        cam = GradCAM(model=disease_model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted.item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        original_image = np.array(image.resize((224, 224))) / 255.0

        visualization = show_cam_on_image(
            original_image,
            grayscale_cam,
            use_rgb=True
        )

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        uploaded_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return render_template(
            "index.html",
            prediction=disease_name.capitalize(),
            confidence=confidence_percent,
            leukemia_stage=leukemia_stage,
            leukemia_confidence=leukemia_conf,
            disease_details=info,
            uploaded_image=uploaded_base64,
            gradcam_image=gradcam_base64
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)