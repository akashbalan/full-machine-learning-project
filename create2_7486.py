import pandas as pd
import numpy as np
import re

# ======================================
# 🧾 Step 1: Load dataset
# ======================================
df = pd.read_csv("train.csv")

# ======================================
# 🔹 Step 2: Clean and extract info
# ======================================
df['price_log'] = np.log(df['price'])
df['catalog_content'] = df['catalog_content'].astype(str).str.lower()

def extract_item_name(text):
    match = re.search(r'item name:\s*(.*?)(?:\n|$)', text)
    return match.group(1).strip() if match else None

def extract_value(text):
    match = re.search(r'value:\s*([\d\.]+)', text)
    return float(match.group(1)) if match else None

# Updated to capture units like "fl oz"
def extract_unit(text):
    match = re.search(r'unit:\s*([^\n]+)', text)  # capture everything after 'unit:' till newline
    return match.group(1).strip() if match else None

def extract_pack(text):
    match = re.search(r'\(pack of\s*(\d+)\)', text)
    return int(match.group(1)) if match else None

df['item_name'] = df['catalog_content'].apply(extract_item_name)
df['value'] = df['catalog_content'].apply(extract_value)
df['unit'] = df['catalog_content'].apply(extract_unit)
df['pack'] = df['catalog_content'].apply(extract_pack)
df['pack'] = df['pack'].fillna(1).astype(int)

# ======================================
# 🔹 Step 3: Clean unit column
# ======================================
def clean_unit(unit):
    if pd.isna(unit):
        return None
    unit = str(unit).lower()
    unit = unit.replace("unit:", "")  # remove prefix if present
    unit = unit.strip()
    return unit

df['unit_clean'] = df['unit'].apply(clean_unit)

# ======================================
# 🔹 Step 4: Normalize units to liquid/solid
# (commented out in your original code)
# ======================================
# liquid_units = ['fl', 'fl oz', 'fluid', 'ml', 'milliliter', 'millilitre', 'liter', 'liters', 'ltr', 'mililitro']
# solid_units = ['oz', 'ounce', 'ounces', 'gram', 'grams', 'gramm', 'lb', 'pound', 'pounds', 'kg', 'g', 
#                'count', 'each', 'ct', 'jar', 'bottle', 'can', 'tea', 'pack', 'piece', 'bag', 'box', 
#                'pouch', 'bucket', 'per', 'comes', 'units', 'case', 'product', 'foot', 'cm', 'unit', 
#                'paper', 'capsule', 'bottles', 'ziplock', 'carton', 'packs', 'sq', 'k', 'in']

# def normalize_unit(unit):
#     if unit is None:
#         return None
#     unit = unit.lower()
#     if any(lq in unit for lq in liquid_units):
#         return 'liquid'
#     else:
#         return 'solid'

# df['unit_type'] = df['unit_clean'].apply(normalize_unit)

# ======================================
# 🧠 Step 5: Load precomputed embeddings
# ======================================
train_text_embeddings = np.load("train_text_embeddings.npy", allow_pickle=True)
train_image_embeddings = np.load("train_image_embeddings.npy", allow_pickle=True)

print("✅ Embeddings loaded successfully")
print("Text:", train_text_embeddings.shape)
print("Image:", train_image_embeddings.shape)

# ======================================
# 🔗 Step 6: Add as single columns
# ======================================
df['text_embeddings'] = [emb.tolist() for emb in train_text_embeddings]
df['image_embeddings'] = [emb.tolist() for emb in train_image_embeddings]

# ======================================
# 🏷 Step 8: Extract extra attributes from catalog_content
# ======================================
keywords = [
    # Freshness & Natural Quality
    "freshness & natural quality", "farm-fresh", "handpicked", "pure", "wholesome", 
    "clean-label", "non-gmo", "unprocessed", "pesticide-free", "minimally processed",
    "sun-ripened", "field-grown", "garden-fresh", "naturally sourced",
    "locally harvested", "freshly packed", "preservative-free", "naturally flavorful",

    # Taste & Texture
    "rich", "smooth", "creamy", "crunchy", "tender", "juicy", "crisp", "buttery", 
    "velvety", "melt-in-your-mouth", "zesty", "tangy", "flavorful", "mouthwatering",
    "irresistible", "perfectly balanced", "bold flavor", "authentic taste",

    # Health & Nutrition
    "nutritious", "high-protein", "low-fat", "low-calorie", r"\w+-free",  # supports gluten-free, sugar-free, etc.
    "vegan", "keto-friendly", "heart-healthy", "antioxidant-rich", "high-fiber",
    "vitamin-packed", "immunity-boosting", "energy-boosting", "cholesterol-free",
    "potassium", "manganese", "calcium",

    # Quality & Craftsmanship
    "premium", "top-grade", "artisanal", "handcrafted", "gourmet", "chef’s choice",
    "small-batch", "signature recipe", "award-winning", "time-tested", "slow-cooked",
    "masterfully blended", "authentic recipe", "heritage recipe", "traditional", "classic",

    # Aroma & Appeal
    "aromatic", "fragrant", "delightful", "inviting", "fresh aroma", "subtle scent",
    "natural essence", "indulgent", "heavenly", "satisfying",

    # Ethical & Sustainable
    "eco-friendly", "sustainably sourced", "fair-trade", "cruelty-free", "ethically produced",
    "recyclable packaging", "environmentally conscious", "responsibly made"
]

pattern = re.compile(r'\b(?:' + '|'.join(keywords) + r')\b', re.IGNORECASE)

def extract_attributes(text):
    if pd.isna(text):
        return ""
    matches = pattern.findall(text)
    seen = set()
    unique_matches = [m.lower() for m in matches if not (m.lower() in seen or seen.add(m.lower()))]
    return ', '.join(unique_matches)

df["extra_attributes"] = df["catalog_content"].apply(extract_attributes)

# ======================================
# 💾 Step 7: Save final dataset
# ======================================
df.to_csv("final7486_s3.csv", index=False)
print("✅ Final dataset saved to final7486_s3.csv")
print("Final shape:", df.shape)

# Optional: check units
# print(df[['unit', 'unit_clean', 'unit_type', 'extra_attributes']].head(20))
