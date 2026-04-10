import io
import json
import time
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import os

app = FastAPI(title="Food Vision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FOOD_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

FOOD_INFO = {
    "pizza": {"calories": 285, "cuisine": "Italian", "emoji": "🍕"},
    "hamburger": {"calories": 354, "cuisine": "American", "emoji": "🍔"},
    "sushi": {"calories": 200, "cuisine": "Japanese", "emoji": "🍣"},
    "ramen": {"calories": 436, "cuisine": "Japanese", "emoji": "🍜"},
    "tacos": {"calories": 226, "cuisine": "Mexican", "emoji": "🌮"},
    "french_fries": {"calories": 312, "cuisine": "American", "emoji": "🍟"},
    "ice_cream": {"calories": 207, "cuisine": "Universal", "emoji": "🍦"},
    "chocolate_cake": {"calories": 352, "cuisine": "Universal", "emoji": "🍰"},
    "waffles": {"calories": 291, "cuisine": "Belgian", "emoji": "🧇"},
    "pancakes": {"calories": 227, "cuisine": "American", "emoji": "🥞"},
    "donuts": {"calories": 452, "cuisine": "American", "emoji": "🍩"},
    "hot_dog": {"calories": 290, "cuisine": "American", "emoji": "🌭"},
    "dumplings": {"calories": 220, "cuisine": "Chinese", "emoji": "🥟"},
    "gyoza": {"calories": 210, "cuisine": "Japanese", "emoji": "🥟"},
    "spring_rolls": {"calories": 165, "cuisine": "Asian", "emoji": "🥢"},
    "samosa": {"calories": 308, "cuisine": "Indian", "emoji": "🫓"},
    "chicken_curry": {"calories": 290, "cuisine": "Indian", "emoji": "🍛"},
    "pad_thai": {"calories": 357, "cuisine": "Thai", "emoji": "🍜"},
    "paella": {"calories": 347, "cuisine": "Spanish", "emoji": "🥘"},
    "lasagna": {"calories": 377, "cuisine": "Italian", "emoji": "🫕"},
    "apple_pie": {"calories": 237, "cuisine": "American", "emoji": "🥧"},
    "cheesecake": {"calories": 321, "cuisine": "American", "emoji": "🍰"},
    "tiramisu": {"calories": 240, "cuisine": "Italian", "emoji": "🍮"},
    "macarons": {"calories": 101, "cuisine": "French", "emoji": "🍬"},
    "churros": {"calories": 116, "cuisine": "Spanish", "emoji": "🥐"},
    "steak": {"calories": 271, "cuisine": "American", "emoji": "🥩"},
    "grilled_salmon": {"calories": 208, "cuisine": "Universal", "emoji": "🐟"},
    "caesar_salad": {"calories": 184, "cuisine": "American", "emoji": "🥗"},
    "greek_salad": {"calories": 211, "cuisine": "Greek", "emoji": "🥗"},
    "hummus": {"calories": 177, "cuisine": "Middle Eastern", "emoji": "🫙"},
    "falafel": {"calories": 333, "cuisine": "Middle Eastern", "emoji": "🧆"},
    "pho": {"calories": 215, "cuisine": "Vietnamese", "emoji": "🍲"},
    "bibimbap": {"calories": 490, "cuisine": "Korean", "emoji": "🍚"},
    "nachos": {"calories": 346, "cuisine": "Mexican", "emoji": "🧀"},
    "guacamole": {"calories": 149, "cuisine": "Mexican", "emoji": "🥑"},
    "eggs_benedict": {"calories": 364, "cuisine": "American", "emoji": "🍳"},
    "french_toast": {"calories": 283, "cuisine": "French", "emoji": "🍞"},
    "omelette": {"calories": 154, "cuisine": "Universal", "emoji": "🍳"},
    "fried_rice": {"calories": 238, "cuisine": "Asian", "emoji": "🍚"},
    "spaghetti_bolognese": {"calories": 367, "cuisine": "Italian", "emoji": "🍝"},
    "spaghetti_carbonara": {"calories": 348, "cuisine": "Italian", "emoji": "🍝"},
    "risotto": {"calories": 337, "cuisine": "Italian", "emoji": "🍚"},
    "ravioli": {"calories": 219, "cuisine": "Italian", "emoji": "🍝"},
    "gnocchi": {"calories": 267, "cuisine": "Italian", "emoji": "🍝"},
    "clam_chowder": {"calories": 187, "cuisine": "American", "emoji": "🍲"},
    "lobster_bisque": {"calories": 165, "cuisine": "French", "emoji": "🦞"},
    "creme_brulee": {"calories": 332, "cuisine": "French", "emoji": "🍮"},
    "macaroni_and_cheese": {"calories": 358, "cuisine": "American", "emoji": "🧀"},
    "chicken_wings": {"calories": 290, "cuisine": "American", "emoji": "🍗"},
    "poutine": {"calories": 480, "cuisine": "Canadian", "emoji": "🍟"},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(FOOD_CLASSES))
    
    model_path = "/app/model/food_model.pth" if os.path.exists("/app/model/food_model.pth") else "model/food_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded fine-tuned model from {model_path}")
    else:
        print("No fine-tuned weights found — using ImageNet pretrained backbone (demo mode)")
    
    model.eval()
    model.to(DEVICE)
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
def serve_frontend():
    frontend_path = "/app/frontend/index.html" if os.path.exists("/app/frontend/index.html") else "frontend/index.html"
    return FileResponse(frontend_path)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": "MobileNetV3-Large",
        "classes": len(FOOD_CLASSES)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
    
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")
    
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    start = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    elapsed = round((time.time() - start) * 1000, 1)
    
    top5_probs, top5_idx = torch.topk(probs, 5)
    
    predictions = []
    for prob, idx in zip(top5_probs.tolist(), top5_idx.tolist()):
        name = FOOD_CLASSES[idx]
        info = FOOD_INFO.get(name, {"calories": None, "cuisine": "Unknown", "emoji": "🍽️"})
        predictions.append({
            "label": name.replace("_", " ").title(),
            "raw_label": name,
            "confidence": round(prob * 100, 2),
            "calories_per_serving": info["calories"],
            "cuisine": info["cuisine"],
            "emoji": info["emoji"],
        })
    
    return {
        "predictions": predictions,
        "inference_time_ms": elapsed,
        "device": str(DEVICE),
        "image_size": f"{image.width}x{image.height}"
    }

@app.get("/classes")
def get_classes():
    return {"classes": [c.replace("_", " ").title() for c in FOOD_CLASSES], "total": len(FOOD_CLASSES)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
