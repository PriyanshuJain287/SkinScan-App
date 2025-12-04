import os
import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import openai
from openai import OpenAI
from PIL import Image
import cv2
import requests 
from dotenv import load_dotenv 
from bson import ObjectId
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO

# Torch imports for the classification model
import torch
import torch.nn as nn
from torchvision import models, transforms

# Sklearn imports for clinical preprocessing & joblib for loading
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib 

# ---------------- CONFIG AND SETUP ----------------

app = Flask(__name__)
CORS(app)

# Load environment variables from the .env file
load_dotenv() 

# -------- Secure MongoDB Setup --------
MONGO_URI = os.getenv('MONGO_URI') 
WEATHERBIT_API = os.getenv("WEATHERBIT_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

if not MONGO_URI:
    raise ValueError("No MONGO_URI found in environment variables. Please set it in your .env file.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

client = MongoClient(MONGO_URI) 
db = client['user-info']
users_collection = db['users']
db_blogs = client['blogs']
blogs_collection = db_blogs['blog-info']
profile_info = client['profile']['profile-info']
profiles_collection = db['profiles']

# -------- YOLO Model Setup --------
model = YOLO("models/yolov8s_e100.pt")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------- CLASSIFICATION MODEL CONFIG (Optional - only if files exist) --------
BEST_MODEL_PATH = "models/best_resnet50_attention_clinical_multiclass.pth" 
SCALER_PATH = "models/clinical_scaler.joblib" 
PCA_PATH = "models/clinical_pca.joblib" 
OHE_COLS_PATH = "models/clinical_ohe_columns.json" 

NUM_CLINICAL_FEATURES_FINAL = 28
NUM_IMAGE_FEATURES_REDUCED = 112
CLASS_ORDER = ["BCC", "ACK", "SCC", "MEL", "NEV", "SEK"]
NUM_CLASSES = len(CLASS_ORDER)
CLINICAL_USE_COLS = ["age", "gender", "fitspatrick", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation"]

# Global objects to be loaded from disk
GLOBAL_SCALER = None
GLOBAL_PCA = None
CLINICAL_MODEL_COLUMNS = []
clf_model = None
classification_loaded = False

# ------------ CLINICAL BOOST FIX ---------------
def apply_clinical_boost(prob_dict, clinical):
    """
    Post-processing function to manually boost class probabilities
    based on key clinical input (ABCDE features).
    This directly fixes the issue of clinical data not contributing.
    """
    # Create a mutable copy of the probabilities dictionary
    p = prob_dict.copy()

    # Convert string keys (from JSON) to floats for modification
    for k in p:
        p[k] = float(p[k])

    # --- Malignancy/Advanced Lesion Boost (Evolution, Bleeding) ---
    # If the lesion has changed (Evolution) or is bleeding (Bleed), boost severe classes (MEL/SCC).
    if clinical.get("changed", 0) == 1 or clinical.get("bleed", 0) == 1:
        p["MEL"] = p.get("MEL", 0) + 0.7
        p["SCC"] = p.get("SCC", 0) + 0.4
        # Minor decrease in benign/less severe
        p["NEV"] = max(0.0, p.get("NEV", 0) - 0.2)
        p["SEK"] = max(0.0, p.get("SEK", 0) - 0.2)


    # --- SCC/ACK Boost (Itch and Elevation) ---
    # SCCs (Squamous Cell Carcinomas) and AKs (Actinic Keratosis) are often itchy and elevated.
    if clinical.get("itch", 0) == 1 and clinical.get("elevation", 0) == 1:
        p["SCC"] = p.get("SCC", 0) + 0.4
        p["ACK"] = p.get("ACK", 0) + 0.4
        

    # --- Benign Boost (Stable Lesion) ---
    # If all key clinical symptoms are absent, boost the benign Nevus class.
    if all(clinical.get(k, 0) == 0 for k in ["itch","grew","hurt","changed","bleed","elevation"]):
        p["NEV"] = p.get("NEV", 0) + 0.6
        # Minor decrease in malignant classes
        p["MEL"] = max(0.0, p.get("MEL", 0) - 0.1)
        p["SCC"] = max(0.0, p.get("SCC", 0) - 0.1)


    # --- Normalization ---
    total = sum(p.values())
    # Normalize the boosted probabilities back to sum to 1.0
    return {k: round(v/total, 4) for k, v in p.items()}

# -------- Try to Load Classification Model (Optional) --------
try:
    # Load Clinical Preprocessing Objects
    GLOBAL_SCALER = joblib.load(SCALER_PATH)
    GLOBAL_PCA = joblib.load(PCA_PATH)
    with open(OHE_COLS_PATH, 'r') as f:
        CLINICAL_MODEL_COLUMNS = json.load(f)
    
    # ResNet50 Attention Model Class
    class ResNet50ChannelAttention(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES, num_cli=NUM_CLINICAL_FEATURES_FINAL, num_img_red=NUM_IMAGE_FEATURES_REDUCED):
            super().__init__()
            base = models.resnet50(weights=None) 
            feat_dim = base.fc.in_features
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
            self.att_fc = nn.Sequential(
                nn.Linear(num_cli, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
            )
            self.reducer = nn.Sequential(
                nn.Linear(feat_dim, num_img_red),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.classifier = nn.Linear(num_img_red + num_cli, num_classes)
            
        def forward(self, x_img, x_cli):
            x = self.feature_extractor(x_img)
            x = torch.flatten(x, 1)
            att = self.att_fc(x_cli)
            att = torch.sigmoid(att)
            x_att = x * att
            x_red = self.reducer(x_att)
            x_comb = torch.cat([x_red, x_cli], dim=1)
            return self.classifier(x_comb)

    # Load Classification Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf_model = ResNet50ChannelAttention(num_classes=NUM_CLASSES).to(device)
    
    state = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False) 
    clf_model.load_state_dict(state["model_state"]) 
    clf_model.eval()
    
    classification_loaded = True
    print(f"✅ Loaded classification model: {BEST_MODEL_PATH} on {device}")
    
except Exception as e:
    print(f"⚠️ Classification model not loaded: {e}")
    print("🔵 Falling back to YOLO detection only")
    classification_loaded = False

# -------- Image Preprocessing Functions --------
def apply_shades_of_gray_p6(image_bgr):
    img = image_bgr.astype(np.float32)
    R = img[:, :, 2]; G = img[:, :, 1]; B = img[:, :, 0]
    eps = 1e-8
    Rp = np.power(R + eps, 6).mean()
    Gp = np.power(G + eps, 6).mean()
    Bp = np.power(B + eps, 6).mean()
    Rm = np.power(Rp, 1/6)
    Gm = np.power(Gp, 1/6)
    Bm = np.power(Bp, 1/6)
    Gray = (Rm + Gm + Bm) / 3.0
    sR = Gray / (Rm + 1e-6)
    sG = Gray / (Gm + 1e-6)
    sB = Gray / (Bm + 1e-6)
    out = img.copy()
    out[:, :, 2] = np.clip(out[:, :, 2] * sR, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] * sG, 0, 255)
    out[:, :, 0] = np.clip(out[:, :, 0] * sB, 0, 255)
    return out.astype(np.uint8)

def apply_clahe(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L = clahe.apply(L)
    lab = cv2.merge([L, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_image(image_bgr):
    img = apply_shades_of_gray_p6(image_bgr)
    img = apply_clahe(img)
    return img

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------- Clinical Data Processor --------
def process_clinical_data(data_row, scaler, pca, model_columns):
    if not model_columns:
        raise RuntimeError("Clinical model columns not loaded. Cannot process data.")
        
    df_single = pd.DataFrame([data_row])
    
    numeric_cols = ["age"] 
    categorical_cols = [col for col in CLINICAL_USE_COLS if col not in numeric_cols]
    
    for nc in numeric_cols:
        df_single[nc] = pd.to_numeric(df_single[nc], errors='coerce').fillna(45.0) 

    for cc in categorical_cols:
        df_single[cc] = df_single[cc].astype(object).fillna("missing").astype(str)

    df_cat = pd.get_dummies(df_single[categorical_cols], drop_first=False)
    df_combined = pd.concat([df_single[numeric_cols], df_cat], axis=1)
    
    df_aligned = pd.DataFrame(0, index=[0], columns=model_columns, dtype=np.float32)
    
    for col in df_combined.columns:
        if col in df_aligned.columns:
            df_aligned[col] = df_combined[col].iloc[0]
            
    X_scaled = scaler.transform(df_aligned.values.astype(np.float32))
    X_pca = pca.transform(X_scaled)
    
    if X_pca.shape[1] < NUM_CLINICAL_FEATURES_FINAL:
        pad_width = NUM_CLINICAL_FEATURES_FINAL - X_pca.shape[1]
        X_pca = np.concatenate([X_pca, np.zeros((X_pca.shape[0], pad_width), dtype=np.float32)], axis=1)
        
    return X_pca[0] 

# -------- JSON Encoder --------
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super(JSONEncoder, self).default(obj)

app.json_encoder = JSONEncoder

# ----------------- ROUTES -----------------

@app.route('/')
def home():
    return "SkinScan Backend is Running!"

# -------- User Authentication Routes --------
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    full_name = data.get('full_name')
    email = data.get('email')
    password = data.get('password')

    if not full_name or not email or not password:
        return jsonify({"message": "All fields are required"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"message": "Email already exists"}), 400

    hashed_password = generate_password_hash(password)
    users_collection.insert_one({
        "full_name": full_name,
        "email": email,
        "password": hashed_password
    })
    return jsonify({"message": "User created successfully!"}), 201

@app.route("/api/login", methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401
    if check_password_hash(user["password"], password):
        return jsonify({"message": "Login successful", "email": email, "full_name": user["full_name"]}), 200
    else:
        return jsonify({"error": "Invalid email or password"}), 401

# -------- Enhanced Skin Lesion Analysis Route --------
@app.route("/api/upload", methods=["POST"])
def detect_lesion():
    # Basic validation
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Failed to read image"}), 400
    
    # Run YOLO detection
    results = model(img)
    response_boxes = []
    
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            response_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})
        
        # If classification model is loaded, try to run classification
        if classification_loaded:
            try:
                # Get clinical data from form
                clinical_data = {
                    "region": request.form.get("region", "trunk"),
                    "itch": int(request.form.get("itch", 0)),
                    "grew": int(request.form.get("grew", 0)),
                    "hurt": int(request.form.get("hurt", 0)),
                    "changed": int(request.form.get("changed", 0)),
                    "bleed": int(request.form.get("bleed", 0)),
                    "elevation": int(request.form.get("elevation", 0)),
                    "age": int(request.form.get("age", 45)),  # Updated default
                    "gender": request.form.get("gender", "other"),
                    "fitspatrick": int(request.form.get("fitspatrick", 3))
                }
                
                # Process clinical data
                cli_vec_np = process_clinical_data(clinical_data, GLOBAL_SCALER, GLOBAL_PCA, CLINICAL_MODEL_COLUMNS)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cli_tensor = torch.tensor(cli_vec_np, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Process image for classification
                img_preprocessed_bgr = preprocess_image(img)
                img_rgb = cv2.cvtColor(img_preprocessed_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tensor = VAL_TRANSFORM(img_pil).unsqueeze(0).to(device)
                
                # Run classification
                with torch.no_grad():
                    outputs = clf_model(img_tensor, cli_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                
                # Get initial probabilities
                prob_dict = {
                    CLASS_ORDER[i]: float(probabilities[i].item()) for i in range(NUM_CLASSES)
                }

                # --- APPLY CLINICAL BOOST FIX ---
                boosted_probs = apply_clinical_boost(prob_dict, clinical_data)
                
                # Final prediction based on boosted probabilities
                predicted_class = max(boosted_probs, key=boosted_probs.get)
                confidence = boosted_probs[predicted_class]

                is_cancerous = predicted_class in ["MEL", "BCC", "SCC"]
                cancer_status = "Potentially Cancerous" if is_cancerous else "Non-Cancerous"
                
                return jsonify({
                    "status": "Lesion Detected & Classified",
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": boosted_probs,
                    "detections": response_boxes,
                    "cancer_status": cancer_status,
                    "is_cancerous": is_cancerous
                })
                
            except Exception as e:
                print(f"Classification failed, falling back to detection: {e}")
                # Fall back to basic detection
                return jsonify({
                    "status": "Lesion Detected",
                    "message": "Lesion detected but classification failed",
                    "detections": response_boxes
                })
        else:
            # Classification not loaded, return basic detection
            return jsonify({
                "status": "Lesion Detected", 
                "message": "Lesion detected successfully",
                "detections": response_boxes
            })
    else:
        return jsonify({
            "status": "No Lesion Detected",
            "message": "No lesions detected in the image"
        })

# -------- Weather API --------
@app.route("/api/weather", methods=["POST"])
def get_weather():
    data = request.get_json()
    latitude = data.get('lat')
    longitude = data.get('lon')

    if not latitude or not longitude:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    try:
        open_weather_url = f"https://api.weatherbit.io/v2.0/current?lat={latitude}&lon={longitude}&key={WEATHERBIT_API}&include=minutely"
        response = requests.get(open_weather_url)
        response.raise_for_status()
        
        weather_data = response.json()
        current_weather = weather_data.get('data', {})[0]
        temp = current_weather.get('temp')
        uvi = current_weather.get('uv')

        if temp is None or uvi is None:
            return jsonify({"error": "Weather data is incomplete from the external API"}), 500

        return jsonify({"temp": temp, "uvi": uvi}), 200

    except requests.exceptions.RequestException as e:
        print(f"Error fetching from WeatherBit: {e}")
        return jsonify({"error": "Failed to fetch weather data from external service"}), 502
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected server error occurred"}), 500

# -------- Blog Routes --------
@app.route("/api/blogs", methods=['GET'])
def get_blogs():
    try:
        blogs_cursor = blogs_collection.find().sort("created_at", -1)
        blogs_list = []
        for blog in blogs_cursor:
            blogs_list.append({
                "_id": str(blog["_id"]),
                "title": blog.get("title"),
                "slug": blog.get("slug"),
                "category": blog.get("category"),
                "summary": blog.get("summary"),
                "tags": blog.get("tags", []),
                "content": blog.get("content", []),
                "references": blog.get("references", []),
                "created_at": blog.get("created_at"),
                "updated_at": blog.get("updated_at")
            })
        return jsonify(blogs_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def serialize_blog(blog):
    return {
        "_id": str(blog["_id"]),
        "title": blog.get("title"),
        "slug": blog.get("slug"),
        "category": blog.get("category"),
        "summary": blog.get("summary"),
        "tags": blog.get("tags", []),
        "content": blog.get("content", []),
        "references": blog.get("references", []),
        "image_url": blog.get("image_url"),
        "created_at": blog.get("created_at"),
        "updated_at": blog.get("updated_at")
    }

@app.route("/api/blogs/<slug>", methods=['GET'])
def get_blog_by_slug(slug):
    try:
        blog = blogs_collection.find_one({"slug": slug})
        if not blog:
            return jsonify({"error": "Blog not found"}), 404
        return jsonify(serialize_blog(blog)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------- Nearby Hospitals API --------
@app.route("/api/nearby-hospitals", methods=["POST"])
def get_nearby_hospitals():
    print("🔄 Nearby hospitals endpoint called")
    
    data = request.get_json()
    latitude = data.get('lat')
    longitude = data.get('lon')
    radius = data.get('radius', 5000)

    if not latitude or not longitude:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    if not GOOGLE_PLACES_API_KEY:
        return jsonify({"error": "Google Places API key not configured"}), 500

    try:
        places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            'location': f'{latitude},{longitude}',
            'radius': radius,
            'type': 'hospital',
            'keyword': 'cancer oncology',
            'key': GOOGLE_PLACES_API_KEY
        }
        
        response = requests.get(places_url, params=params)
        response.raise_for_status()
        
        places_data = response.json()
        
        if places_data.get('status') != 'OK':
            return jsonify({"error": f"Google Places API error: {places_data.get('status')}"}), 400

        hospitals = []
        for place in places_data.get('results', [])[:10]:
            hospital_info = {
                'name': place.get('name'),
                'address': place.get('vicinity', 'Address not available'),
                'rating': place.get('rating'),
                'place_id': place.get('place_id'),
                'location': place.get('geometry', {}).get('location', {}),
                'open_now': place.get('opening_hours', {}).get('open_now')
            }
            
            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {
                'place_id': place.get('place_id'),
                'fields': 'formatted_phone_number,website',
                'key': GOOGLE_PLACES_API_KEY
            }
            
            try:
                details_response = requests.get(details_url, params=details_params)
                details_data = details_response.json()
                if details_data.get('status') == 'OK':
                    result = details_data.get('result', {})
                    hospital_info['phone'] = result.get('formatted_phone_number')
                    hospital_info['website'] = result.get('website')
            except Exception as e:
                print(f"Error fetching details for {place.get('name')}: {e}")
            
            hospitals.append(hospital_info)

        print(f"✅ Found {len(hospitals)} hospitals")
        return jsonify({"hospitals": hospitals}), 200

    except requests.exceptions.RequestException as e:
        print(f"❌ Google Places API error: {e}")
        return jsonify({"error": "Failed to fetch hospital data"}), 502
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return jsonify({"error": "An unexpected server error occurred"}), 500

# -------- Profile Management Routes --------
@app.route('/api/profile/update_skin_tone', methods=['POST'])
def api_update_skin_tone():
    try:
        data = request.json
        username = data.get('username')
        fitzpatrick_level = data.get('fitzpatrickLevel')
        
        print(f"Updating skin tone for {username} to level {fitzpatrick_level}")
        
        if not username or fitzpatrick_level is None:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        result = profile_info.update_one(
            {'userId': username},
            {'$set': {
                'skinProfile.fitzpatrickLevel': fitzpatrick_level,
                'updatedAt': datetime.utcnow()
            }},
            upsert=True
        )
        
        print(f"MongoDB update result: {result.modified_count} documents modified")
        
        return jsonify({
            'success': True, 
            'message': 'Skin tone updated successfully',
            'fitzpatrickLevel': fitzpatrick_level
        })
    
    except Exception as e:
        print(f"Error updating skin tone: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/profile/update', methods=['POST'])
def api_update_profile():
    try:
        data = request.json
        username = data.get('username')
        
        print(f"Updating profile for: {username}")
        print(f"Received data: {data}")
        
        if not username:
            return jsonify({'success': False, 'error': 'Username is required'}), 400
        
        update_data = {
            'userId': username,
            'personalInfo': {
                'name': data.get('name', ''),
                'email': data.get('email', ''),
                'gender': data.get('gender', ''),
                'age': data.get('age', '')
            },
            'skinProfile': {
                'fitzpatrickLevel': data.get('fitzpatrickLevel', 3),
                'skinType': data.get('skinType', ''),
                'conditions': data.get('skinConditions', [])
            },
            'updatedAt': datetime.utcnow()
        }
        
        existing_profile = profile_info.find_one({'userId': username})
        if not existing_profile:
            update_data['createdAt'] = datetime.utcnow()
            print("Creating new profile")
        else:
            print("Updating existing profile")
        
        result = profile_info.update_one(
            {'userId': username},
            {'$set': update_data},
            upsert=True
        )
        
        print(f"Profile update successful: {result.modified_count} modified")
        
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    
    except Exception as e:
        print(f"Error updating profile: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/profile/<username>', methods=['GET'])
def api_get_profile(username):
    try:
        print(f"Fetching profile for: {username}")
        
        profile = profile_info.find_one({'userId': username})
        
        if not profile:
            print("Profile not found, returning default")
            return jsonify({
                'success': True, 
                'profile': {
                    'userId': username,
                    'personalInfo': {'name': '', 'email': '', 'gender': '', 'age': ''},
                    'skinProfile': {'fitzpatrickLevel': 3, 'skinType': '', 'conditions': []}
                }
            })
        
        if '_id' in profile:
            profile['_id'] = str(profile['_id'])
        
        print(f"Profile found: {profile}")
        return jsonify({'success': True, 'profile': profile})
    
    except Exception as e:
        print(f"Error fetching profile: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    

# -------- Enhanced Chatbot Configuration --------
CHATBOT_SYSTEM_PROMPT = """
You are SkinAI, an intelligent and empathetic dermatology assistant for the SkinScan mobile app. Your role is to:

APP NAVIGATION HELP:
- Guide users to use different features: skin analysis, hospital finder, blog, reminders, profile
- Explain what each screen does and how to use it
- Help users understand their skin analysis results
- Assist with setting up reminders and finding healthcare providers

SKIN HEALTH EXPERTISE:
- Provide accurate information about skin conditions, prevention, and care
- Explain medical terms in simple language (melanoma, BCC, SCC, etc.)
- Offer sun protection advice and skincare routines
- Guide on when to seek professional medical help

CONVERSATIONAL STYLE:
- Be warm, friendly, and conversational like a helpful friend
- Ask follow-up questions to understand user needs better
- Provide detailed, helpful responses (3-5 paragraphs when needed)
- Use emojis occasionally to make it engaging 👍

CRITICAL SAFETY:
- ⚠️ ALWAYS state you are an AI assistant and not a substitute for medical advice
- ⚠️ NEVER diagnose - always recommend consulting dermatologists
- ⚠️ For urgent concerns, advise immediate medical attention
- ⚠️ Be clear about limitations

APP FEATURES TO MENTION:
- Skin Analysis: "Take a photo of any skin spot for AI analysis"
- Hospital Finder: "Find nearby dermatologists and skin cancer centers" 
- Blog: "Read educational articles about skin health"
- Reminders: "Set reminders for skin checks and sun protection"
- Profile: "Set up your skin type and preferences"

Be genuinely helpful and make users feel supported in their skin health journey! 🌟
"""

# Enhanced quick responses for better fallback
QUICK_RESPONSES = {
    "skin analysis": "You can analyze skin spots using our AI! 📸 Go to the Home screen, take or upload a photo, and fill in the clinical information. The AI will check for concerning features and provide insights. Remember, this is screening only - always consult a doctor for diagnosis!",
    
    "hospitals": "Need to find a dermatologist? 🏥 Go to the Hospitals screen to find nearby skin cancer centers and dermatology clinics. You can see ratings, contact info, and get directions!",
    
    "blog": "Want to learn more about skin health? 📚 Check out our Blog section for educational articles about sun protection, skin cancer awareness, and skincare tips written by medical professionals!",
    
    "reminders": "Stay on top of your skin health! ⏰ Use the Reminders screen to set alerts for monthly skin self-checks, sunscreen reapplication, and dermatologist appointments.",
    
    "profile": "Personalize your experience! 👤 Go to your Profile to set your skin type, Fitzpatrick level, and health conditions. This helps us provide better recommendations!",
    
    "skin cancer": "Skin cancer comes in different types. 🎗️ Melanoma is the most serious but least common. Basal cell carcinoma (BCC) and squamous cell carcinoma (SCC) are more common but less likely to spread. Regular skin checks and sun protection are crucial! Always see a dermatologist for proper evaluation.",
    
    "sun protection": "Sun safety is key! ☀️ Use broad-spectrum SPF 30+ sunscreen daily, reapply every 2 hours, wear protective clothing and hats, seek shade during peak hours (10 AM-4 PM). Don't forget ears, neck, and hands!",
    
    "mole check": "Follow the ABCDE rule for moles: 🔍 Asymmetry (uneven halves), Border irregularity, Color variation, Diameter (>6mm), and Evolution (changing). If you notice any of these, please see a dermatologist promptly!",
    
    "skin routine": "Basic skincare routine: 🧴 Gentle cleanser, moisturizer, and daily sunscreen. For specific concerns, you might add vitamin C (morning) or retinol (night). Always patch test new products!",
    
    "when to see doctor": "See a dermatologist if you notice: 🚨 New growths, changing moles, non-healing sores, itching/bleeding spots, or any skin changes that concern you. Early detection saves lives!"
}

@app.route("/api/chat", methods=["POST"])
def chat_with_ai():
    """
    Enhanced chatbot endpoint with navigation help and conversational AI
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_message = data['message']
        user_id = data.get('user_id', 'anonymous')
        
        # Check if OpenAI API key is available
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not OPENAI_API_KEY:
            # Enhanced fallback with better responses
            response_text = get_enhanced_fallback_response(user_message)
            
            return jsonify({
                "success": True,
                "response": response_text,
                "type": "enhanced_fallback",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Use OpenAI API with new client structure
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": CHATBOT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=800,
                temperature=0.8,
                presence_penalty=0.3,
                frequency_penalty=0.2
            )
            
            ai_response = response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Enhanced fallback
            ai_response = get_enhanced_fallback_response(user_message)
        
        # Store chat history
        try:
            chat_collection = db['chat_history']
            chat_collection.insert_one({
                "user_id": user_id,
                "user_message": user_message,
                "ai_response": ai_response,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            print(f"Failed to save chat history: {e}")
        
        return jsonify({
            "success": True,
            "response": ai_response,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        # Even on error, provide a helpful response
        error_response = "I'm here to help! 🌟 You can ask me about:\n\n• Skin analysis and how to use the app\n• Finding nearby dermatologists\n• Skin cancer information and prevention\n• Skincare routines and sun protection\n• Setting reminders for skin checks\n\nWhat would you like to know?"
        
        return jsonify({
            "success": True,
            "response": error_response,
            "type": "error_fallback",
            "timestamp": datetime.utcnow().isoformat()
        })

def get_enhanced_fallback_response(user_message):
    """Enhanced fallback responses with better matching"""
    message_lower = user_message.lower()
    
    # Navigation and app features
    if any(word in message_lower for word in ['analyze', 'scan', 'photo', 'picture', 'image', 'skin spot']):
        return QUICK_RESPONSES['skin analysis']
    elif any(word in message_lower for word in ['hospital', 'doctor', 'dermatologist', 'clinic', 'find help']):
        return QUICK_RESPONSES['hospitals']
    elif any(word in message_lower for word in ['blog', 'article', 'learn', 'education', 'read']):
        return QUICK_RESPONSES['blog']
    elif any(word in message_lower for word in ['reminder', 'alert', 'schedule', 'remember']):
        return QUICK_RESPONSES['reminders']
    elif any(word in message_lower for word in ['profile', 'settings', 'skin type', 'fitzpatrick']):
        return QUICK_RESPONSES['profile']
    
    # Skin health topics
    elif any(word in message_lower for word in ['cancer', 'melanoma', 'bcc', 'scc', 'basal', 'squamous']):
        return QUICK_RESPONSES['skin cancer']
    elif any(word in message_lower for word in ['sun', 'uv', 'protect', 'sunscreen', 'spf']):
        return QUICK_RESPONSES['sun protection']
    elif any(word in message_lower for word in ['mole', 'check', 'abcde', 'spot']):
        return QUICK_RESPONSES['mole check']
    elif any(word in message_lower for word in ['routine', 'skincare', 'products', 'cleanse', 'moisturize']):
        return QUICK_RESPONSES['skin routine']
    elif any(word in message_lower for word in ['when to see', 'doctor', 'dermatologist', 'appointment', 'urgent']):
        return QUICK_RESPONSES['when to see doctor']
    
    # Greetings and general help
    elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return "Hello! I'm SkinAI, your dermatology assistant! 🌟 I can help you with skin analysis, finding doctors, learning about skin health, and navigating the app. What would you like to know today?"
    elif any(word in message_lower for word in ['help', 'what can you do', 'features']):
        return "I can help you with many things! 🎯\n\n• **Skin Analysis**: Guide you through taking photos for AI screening\n• **Find Help**: Locate nearby dermatologists and skin clinics\n• **Education**: Explain skin conditions and prevention\n• **Reminders**: Set up skin check alerts\n• **Navigation**: Help you use all app features\n\nWhat would you like assistance with?"
    elif any(word in message_lower for word in ['thank', 'thanks']):
        return "You're very welcome! 😊 I'm here whenever you need help with your skin health journey. Don't hesitate to ask if you have more questions!"
    
    # Default helpful response
    return "I'm here to help with your skin health! 🌟 You can ask me about:\n\n• Using the skin analysis feature\n• Finding nearby dermatologists\n• Understanding skin conditions\n• Sun protection and skincare\n• Setting up reminders\n• Navigating the app\n\nWhat would you like to know more about?"

@app.route("/api/chat/quick", methods=["POST"])
def quick_chat():
    """
    Enhanced quick responses with better matching
    """
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        response = get_enhanced_fallback_response(message)
        
        return jsonify({
            "success": True,
            "response": response,
            "type": "enhanced_quick",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": True,
            "response": "Hello! I'm SkinAI! I can help you with skin analysis, finding doctors, and answering skin health questions. What would you like to know?",
            "error": str(e)
        })

# -------- Main Execution --------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
