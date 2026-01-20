"""
Astrabon Product Search - Streamlit App
With Image Scraping from Astrabon Website

Save as: app.py
Run with: streamlit run app.py
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from openai import OpenAI
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import time
import base64
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pathlib import Path

load_dotenv()


class ProductRetriever:
    def __init__(self, api_key: str, embeddings_dir: str, 
                 embedding_model: str = "text-embedding-3-small",
                 generation_model: str = "gpt-4o"):
        """Initialize retriever"""
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.products = []
        self.embeddings = None
        self.index = None
        self.embedding_dimension = None
        self.image_cache = {}  # Cache scraped images
        self.load_embeddings(embeddings_dir)

    
    def load_embeddings(self, embeddings_dir: str):
        """Load embeddings and FAISS index (OS-agnostic & safe)"""

        embeddings_dir = Path(embeddings_dir).expanduser().resolve()

        index_path = embeddings_dir / "products.index"
        products_path = embeddings_dir / "products.json"
        embeddings_path = embeddings_dir / "embeddings.npy"

        # ---- Validation (fail fast) ----
        if not index_path.exists():
            raise FileNotFoundError(f"❌ FAISS index not found: {index_path}")
        if not products_path.exists():
            raise FileNotFoundError(f"❌ Products JSON not found: {products_path}")
        if not embeddings_path.exists():
            raise FileNotFoundError(f"❌ Embeddings file not found: {embeddings_path}")

        # ---- Load FAISS index ----
        self.index = faiss.read_index(str(index_path))

        # ---- Load metadata ----
        with open(products_path, "r", encoding="utf-8") as f:
            self.products = json.load(f)

        # ---- Load embeddings ----
        self.embeddings = np.load(embeddings_path)
        self.embedding_dimension = self.embeddings.shape[1]

        # ---- Verify normalization ----
        first_norm = np.linalg.norm(self.embeddings[0])
        is_normalized = 0.99 <= first_norm <= 1.01

        print(f"✅ Loaded {len(self.products)} products")
        print(f"📐 Embedding dimension: {self.embedding_dimension}")
        print(f"📊 First vector norm: {first_norm:.4f} (normalized={is_normalized})")

        if not is_normalized:
            print("⚠️ WARNING: Embeddings are not L2-normalized. "
                "FAISS cosine similarity may behave incorrectly.")

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        if len(embedding) != self.embedding_dimension:
            if len(embedding) > self.embedding_dimension:
                embedding = embedding[:self.embedding_dimension]
            else:
                padding = np.zeros(self.embedding_dimension - len(embedding), dtype=np.float32)
                embedding = np.concatenate([embedding, padding])
        
        return embedding

    def retrieve_products(self, query: str, top_k: int = 5, 
                         category_filter: Optional[str] = None,
                         min_score: float = 0.25) -> List[Dict[str, Any]]:
        """Retrieve products with filtering"""
        start_time = time.time()
        
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        search_k = min(top_k * 10, len(self.products))
        distances, indices = self.index.search(query_embedding, search_k)
        
        print(f"\n🔍 Query: '{query}'")
        print(f"Top 5 results:")
        for i, (idx, dist) in enumerate(zip(indices[0][:5], distances[0][:5])):
            if idx < len(self.products):
                p = self.products[idx]
                print(f"  {i+1}. [{dist:.3f}] {p.get('title', 'N/A')} - {p.get('product_type', 'N/A')}")
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.products):
                product = self.products[idx].copy()
                similarity_score = float(dist)
                
                if similarity_score < min_score:
                    continue
                
                if category_filter:
                    product_category = product.get('category', '').lower()
                    if category_filter.lower() not in product_category:
                        continue
                
                product['similarity_score'] = similarity_score
                results.append(product)
                
                if len(results) >= top_k:
                    break
        
        elapsed = time.time() - start_time
        print(f"⏱️ Retrieved {len(results)} products in {elapsed:.2f}s")
        
        return results

    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def identify_image_product(self, image: Image.Image) -> Dict[str, str]:
        """Use GPT-4 Vision to identify product"""
        start_time = time.time()
        base64_image = self.encode_image_to_base64(image)
        
        prompt = """Analyze this kitchenware product image.

Use SIMPLE product names that match typical e-commerce listings.

Good examples:
- "fry pan" or "frying pan"
- "sauce pan" or "cooking pot"
- "dinner plate"
- "whiskey glass"
- "table spoon"

Provide ONLY:

Product Type: [Simple common name]
Category: [Kitchenware And Accessories, Cutlery, Glassware, Knife And Accessories, or Tableware]
Material: [nonstick coating, stainless steel, ceramic, glass, etc.]
Primary Color: [main color]

Keep it simple!"""

        response = self.client.chat.completions.create(
            model=self.generation_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            max_tokens=300,
            temperature=0.2
        )
        
        result_text = response.choices[0].message.content
        
        parsed = {}
        for line in result_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip()] = value.split('(')[0].strip()
        
        elapsed = time.time() - start_time
        
        st.info(f"✅ Image identification: {elapsed:.2f}s")
        st.success(f"📦 **{parsed.get('Product Type', 'Unknown')}**")
        st.info(f"🏷️ Category: {parsed.get('Category', 'Unknown')} | Material: {parsed.get('Material', 'Unknown')}")
        
        return parsed

    def create_search_query(self, detection: Dict[str, str]) -> str:
        """Create search query from detection"""
        parts = []
        
        product_type = detection.get('Product Type', '').lower()
        if product_type and product_type != 'unknown':
            # Normalize variations
            if 'frying pan' in product_type or 'skillet' in product_type:
                product_type = 'fry pan'
            elif 'saucepan' in product_type:
                product_type = 'sauce pan'
            
            parts.extend([product_type] * 3)  # Triple weight
        
        category = detection.get('Category', '')
        if category and category.lower() != 'unknown':
            parts.append(category)
        
        material = detection.get('Material', '')
        if material and 'non' in material.lower():
            parts.append(material)
        
        query = ' '.join(parts)
        print(f"🔍 Search query: '{query}'")
        return query

    def search_by_image(self, image: Image.Image, top_k: int = 5):
        """Search using image"""
        st.markdown("### 🔍 Analyzing Image...")
        
        detection = self.identify_image_product(image)
        search_query = self.create_search_query(detection)
        
        st.success(f"🔍 Searching: **'{search_query}'**")
        
        start_time = time.time()
        detected_category = detection.get('Category', '')
        
        products = self.retrieve_products(
            search_query,
            top_k=top_k,
            category_filter=detected_category if detected_category.lower() != 'unknown' else None,
            min_score=0.25
        )
        
        if len(products) == 0:
            st.warning("⚠️ Trying broader search...")
            products = self.retrieve_products(search_query, top_k=top_k, min_score=0.20)
        
        if len(products) == 0:
            st.info("🔄 Simplified search...")
            simple_query = detection.get('Product Type', '').lower()
            if 'pan' in simple_query:
                simple_query = 'pan'
            products = self.retrieve_products(simple_query, top_k=top_k, min_score=0.15)
        
        elapsed = time.time() - start_time
        st.success(f"✅ Found {len(products)} products in {elapsed:.2f}s")
        
        for p in products:
            p['detection'] = detection
        
        return products, detection

    def generate_response(self, query: str, top_k: int = 5, image: Optional[Image.Image] = None):
        """Generate AI response"""
        total_start = time.time()
        
        if image is not None:
            products, detection = self.search_by_image(image, top_k)
            is_image_search = True
        else:
            products = self.retrieve_products(query, top_k, min_score=0.25)
            detection = None
            is_image_search = False
        
        context = "# Retrieved Products:\n\n"
        for i, p in enumerate(products, 1):
            context += f"## Product {i}:\n"
            context += f"- Name: {p.get('title', 'N/A')}\n"
            context += f"- Type: {p.get('product_type', 'N/A')}\n"
            context += f"- Category: {p.get('category', 'N/A')}\n"
            context += f"- Brand: {p.get('brand', 'N/A')}\n"
            context += f"- Price: MVR {p.get('mvr', 'N/A')} / USD {p.get('usd', 'N/A')}\n"
            context += f"- Match: {p.get('similarity_score', 0):.1%}\n\n"
        
        if is_image_search:
            prompt = f"""You are Dhon, a friendly AI shopping assistant for Astrabon Maldives.

Customer uploaded an image. Detected: {detection.get('Product Type', 'unknown')}

{context}

Instructions:
- Greet warmly if this is first interaction
- If products match well (>40%), enthusiastically recommend them with details
- If low scores, politely suggest checking the website or describe what you see
- Be conversational, helpful, and use emojis naturally
- Keep it concise but friendly

Your response (as Dhon):"""
        else:
            prompt = f"""You are Dhon, a friendly AI shopping assistant for Astrabon Maldives.

Customer query: {query}

{context}

Instructions:
- Greet if appropriate
- Recommend products with enthusiasm
- Highlight features, prices, and benefits
- Be conversational and helpful
- Use emojis naturally

Your response (as Dhon):"""
        
        response = self.client.chat.completions.create(
            model=self.generation_model,
            messages=[
                {"role": "system", "content": "You are Dhon, a helpful, friendly, and enthusiastic shopping assistant for Astrabon Maldives. You speak naturally, use emojis, and make shopping enjoyable."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=600
        )
        
        total_time = time.time() - total_start
        
        return {
            'query': query,
            'response': response.choices[0].message.content,
            'products': products,
            'detection': detection,
            'is_image_search': is_image_search,
            'time': total_time
        }


def scrape_product_image(product_url: str) -> Optional[str]:
    """Scrape product image from Astrabon"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(product_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try og:image first (most reliable)
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                return og_image['content']
            
            # Try common image selectors
            selectors = [
                'img.product-image',
                'img[itemprop="image"]',
                'div.product-gallery img',
                '.product-img img',
                'img[alt*="product"]',
                'img[src*="products"]',
                'img[src*="cdn"]'
            ]
            
            for selector in selectors:
                img = soup.select_one(selector)
                if img and img.get('src'):
                    src = img['src']
                    if src.startswith('//'):
                        return 'https:' + src
                    elif src.startswith('/'):
                        return 'https://www.astrabonmaldives.com' + src
                    return src
            
            # Fallback: find any reasonable image
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if src and any(x in src.lower() for x in ['product', 'item', 'cdn', 'upload', 'image']):
                    if src.startswith('//'):
                        return 'https:' + src
                    elif src.startswith('/'):
                        return 'https://www.astrabonmaldives.com' + src
                    return src
    except Exception as e:
        print(f"Error scraping image: {e}")
    
    return None


def load_image_from_url(url: str) -> Optional[Image.Image]:
    """Load image from URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error loading image: {e}")
    return None


def display_product(product: Dict, index: int, retriever: ProductRetriever):
    """Display product card with image"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        product_url = product.get('url', '')
        product_id = product.get('id', '')
        
        # Check cache first
        if product_id in retriever.image_cache:
            img_url = retriever.image_cache[product_id]
        else:
            # Try existing image_url
            img_url = product.get('image_url')
            
            # If no image, scrape it
            if not img_url or img_url == product_url:
                if product_url:
                    img_url = scrape_product_image(product_url)
                    if img_url:
                        retriever.image_cache[product_id] = img_url
        
        # Load and display image
        img = None
        if img_url:
            img = load_image_from_url(img_url)
        
        if img:
            st.image(img, use_container_width=True)
        else:
            # Beautiful placeholder
            product_type = product.get('product_type', 'Product')
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 50px 15px; text-align: center; border-radius: 8px; color: white;">
                <div style="font-size: 36px; margin-bottom: 8px;">📦</div>
                <div style="font-size: 12px; opacity: 0.9;">{product_type}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Color code based on similarity
        score = product.get('similarity_score', 0)
        if score >= 0.6:
            badge_color = "#4CAF50"
        elif score >= 0.4:
            badge_color = "#FF9800"
        else:
            badge_color = "#9E9E9E"
        
        st.markdown(f"""
        <div style="padding: 10px;">
            <h4 style="color: #1E88E5; margin-bottom: 8px;">{index}. {product.get('title', 'N/A')}</h4>
            <p style="margin: 3px 0; font-size: 0.9em;"><b>Type:</b> {product.get('product_type', 'N/A')}</p>
            <p style="margin: 3px 0; font-size: 0.9em;"><b>Category:</b> {product.get('category', 'N/A')}</p>
            <p style="margin: 3px 0; font-size: 0.9em;"><b>Brand:</b> {product.get('brand', 'N/A')}</p>
            <p style="margin: 8px 0;"><b>Price:</b> <span style="color: #4CAF50; font-weight: bold; font-size: 1.1em;">MVR {product.get('mvr', 'N/A')}</span> <span style="color: #666;">/ USD {product.get('usd', 'N/A')}</span></p>
            <p style="margin: 8px 0;">
                <a href="{product.get('url', '#')}" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">
                    🔗 View on Astrabon →
                </a>
            </p>
            <span style="background: {badge_color}; color: white; padding: 5px 15px; border-radius: 15px; font-size: 0.85em; font-weight: 600;">
                Match: {score:.1%}
            </span>
        </div>
        """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Dhon - Astrabon Assistant", page_icon="🛍️", layout="wide")
    
    # Dhon's avatar URL
    DHON_AVATAR = "https://i.postimg.cc/yNf93gM0/dhon.jpg"
    
    st.markdown("""
    <style>
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
    }
    .dhon-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with Dhon
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="dhon-header">
            <img src="{DHON_AVATAR}" style="width: 120px; height: 120px; border-radius: 50%; border: 4px solid white; margin-bottom: 1rem;">
            <h1 style="margin: 0; font-size: 2.5rem;">السلام عليكم! I'm Dhon</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Your AI Shopping Assistant for Astrabon Maldives</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ API key loaded")
        else:
            api_key = st.text_input("OpenAI API Key:", type="password")
        
        BASE_DIR = Path(__file__).resolve().parent
        DEFAULT_EMBEDDINGS_DIR = BASE_DIR / "product_embeddings"

        embeddings_dir = st.text_input(
            "Embeddings Directory:",
            value=str(DEFAULT_EMBEDDINGS_DIR)
        )

        top_k = st.slider("Products to show:", 1, 10, 5)
        
        if 'retriever' in st.session_state:
            st.metric("Total Products", len(st.session_state.retriever.products))
            st.metric("Images Cached", len(st.session_state.retriever.image_cache))
    
    if not api_key:
        st.warning("⚠️ Please set OPENAI_API_KEY in .env file")
        return
    
    # Initialize retriever
    if 'retriever' not in st.session_state:
        try:
            with st.spinner("🔄 Loading embeddings..."):
                st.session_state.retriever = ProductRetriever(api_key, embeddings_dir)
            st.success("✅ System ready!")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Show welcome message on first load
    if 'welcomed' not in st.session_state:
        st.session_state.welcomed = True
        welcome_msg = {
            'query': '',
            'response': """مرحباً! (Marhaba!) Welcome to Astrabon Maldives! 🌴

I'm Dhon, your personal shopping assistant. I'm here to help you find the perfect kitchenware and cookware products!

**What can I do for you?**
- 📸 Upload a photo of any kitchenware item, and I'll find similar products
- 💬 Ask me about specific products (e.g., "Show me affordable fry pans")
- 🔍 Search by category, brand, or price range
- 💡 Get recommendations based on your needs

Try uploading an image or ask me anything! I'm here to make your shopping experience amazing! 🛍️✨""",
            'products': [],
            'detection': None,
            'is_image_search': False,
            'time': 0,
            'is_welcome': True
        }
        st.session_state.chat_history.append(welcome_msg)
    
    # Search mode
    mode = st.radio("", ["💬 Text Search", "📸 Image Search"], horizontal=True, key="search_mode")
    
    # Quick action buttons
    st.markdown("### 🎯 Quick Actions")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("🍳 Cookware", use_container_width=True):
            st.session_state.quick_search = "Show me cookware products"
    with quick_col2:
        if st.button("🍴 Cutlery", use_container_width=True):
            st.session_state.quick_search = "Show me cutlery"
    with quick_col3:
        if st.button("🥃 Glassware", use_container_width=True):
            st.session_state.quick_search = "Show me glassware"
    with quick_col4:
        if st.button("🔪 Knives", use_container_width=True):
            st.session_state.quick_search = "Show me kitchen knives"
    
    if mode == "📸 Image Search":
        st.markdown("---")
        uploaded = st.file_uploader("📷 Upload product image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded:
            img = Image.open(uploaded)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Search Similar Products", type="primary", use_container_width=True):
                try:
                    result = st.session_state.retriever.generate_response(
                        "Find similar",
                        top_k=top_k,
                        image=img
                    )
                    
                    st.markdown("---")
                    st.markdown("### 🤖 Assistant Response")
                    st.write(result['response'])
                    st.caption(f"⏱️ Processed in {result['time']:.2f}s")
                    
                    st.markdown("### 📦 Matching Products")
                    for i, p in enumerate(result['products'], 1):
                        display_product(p, i, st.session_state.retriever)
                        if i < len(result['products']):
                            st.markdown("---")
                    
                    st.session_state.chat_history.append(result)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Chat history
    if len(st.session_state.chat_history) > 0:
        st.markdown("---")
        st.header("💬 Chat with Dhon")
        
        DHON_AVATAR = "https://i.postimg.cc/yNf93gM0/dhon.jpg"
        
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                if chat.get('is_welcome'):
                    # Skip user message for welcome
                    pass
                elif chat['is_image_search']:
                    st.write("🖼️ Uploaded an image")
                    if chat['detection']:
                        st.caption(f"Detected: {chat['detection'].get('Product Type', 'Unknown')}")
                else:
                    st.write(chat['query'])
            
            with st.chat_message("assistant", avatar=DHON_AVATAR):
                st.write(chat['response'])
                
                if len(chat['products']) > 0:
                    with st.expander(f"📦 View {len(chat['products'])} Products"):
                        for i, p in enumerate(chat['products'], 1):
                            display_product(p, i, st.session_state.retriever)
                            if i < len(chat['products']):
                                st.markdown("---")
    
    if mode == "💬 Text Search":
        st.markdown("---")
        
        # Handle quick search buttons
        if 'quick_search' in st.session_state:
            query = st.session_state.quick_search
            del st.session_state.quick_search
            
            with st.chat_message("user"):
                st.write(query)
            
            DHON_AVATAR = "https://i.postimg.cc/yNf93gM0/dhon.jpg"
            with st.chat_message("assistant", avatar=DHON_AVATAR):
                with st.spinner("🔍 Dhon is searching..."):
                    try:
                        result = st.session_state.retriever.generate_response(query, top_k)
                        
                        st.write(result['response'])
                        st.caption(f"⏱️ {result['time']:.2f}s")
                        
                        with st.expander(f"📦 View {len(result['products'])} Products", expanded=True):
                            for i, p in enumerate(result['products'], 1):
                                display_product(p, i, st.session_state.retriever)
                                if i < len(result['products']):
                                    st.markdown("---")
                        
                        st.session_state.chat_history.append(result)
                    except Exception as e:
                        st.error(f"❌ {e}")
        
        query = st.chat_input("💬 Ask Dhon about products...")
        
        if query:
            with st.chat_message("user"):
                st.write(query)
            
            DHON_AVATAR = "https://i.postimg.cc/yNf93gM0/dhon.jpg"
            with st.chat_message("assistant", avatar=DHON_AVATAR):
                with st.spinner("🔍 Dhon is searching..."):
                    try:
                        result = st.session_state.retriever.generate_response(query, top_k)
                        
                        st.write(result['response'])
                        st.caption(f"⏱️ {result['time']:.2f}s")
                        
                        with st.expander(f"📦 View {len(result['products'])} Products", expanded=True):
                            for i, p in enumerate(result['products'], 1):
                                display_product(p, i, st.session_state.retriever)
                                if i < len(result['products']):
                                    st.markdown("---")
                        
                        st.session_state.chat_history.append(result)
                    except Exception as e:
                        st.error(f"❌ {e}")
    
    # Clear button
    if len(st.session_state.chat_history) > 0:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.retriever.image_cache = {}
            st.rerun()


if __name__ == "__main__":
    main()