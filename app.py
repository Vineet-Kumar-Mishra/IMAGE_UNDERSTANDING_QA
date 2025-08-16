import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure API key
API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

# Initialize Gemini model
@st.cache_resource
def get_model():
    """Initialize and cache the Gemini model"""
    return genai.GenerativeModel("gemini-1.5-flash")

model = get_model()

def prepare_image(uploaded_file):
    """
    Converts uploaded Streamlit file to PIL Image for Gemini
    """
    if uploaded_file is None:
        return None
    
    try:
        # Open image using PIL
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary (for PNG with transparency)
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def get_instructions(user_question):
    instructions = (
                        "You are an expert AI assistant specialized in analyzing images and extracting information. "
                        "Please analyze the provided image carefully and answer the user's question accurately. "
                        "If the image contains text (like documents, receipts, signs, etc.), extract and read it precisely. "
                        "If the image contains objects, describe them in detail. "
                        "If you cannot find relevant information for the user's question, clearly state what you can observe instead. "
                        "Be comprehensive but concise in your response.\n\n"
                        f"User Question: {user_question}"
                    )
    return instructions

def generate_response(prompt_text, image=None):
    """
    Generates text response from Gemini Vision
    """
    try:
        if image:
            # Generate content with both text and image
            response = model.generate_content([prompt_text, image])
        else:
            # Generate content with text only
            response = model.generate_content(prompt_text)
        
        # Check if response was blocked
        if hasattr(response, 'prompt_feedback'):
            if response.prompt_feedback.block_reason:
                return f"Response blocked: {response.prompt_feedback.block_reason}"
        
        # Return the generated text
        return response.text
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ------------------ Streamlit UI ------------------
st.set_page_config(
    page_title="Image Document Extractor", 
    layout="wide",
    page_icon="📄"
)

st.title("📄 Image Document Information Extractor")

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an image containing text or objects you want to analyze"
    )
    
    # User input
    user_question = st.text_area(
        "Your Question:",
        placeholder="Ask anything about the image content...",
        height=100
    )
    
    # Submit button - positioned right after input
    if st.button("Analyze Image", type="primary", use_container_width=True):
        
        # Validation
        if not uploaded_file:
            st.warning("Please upload an image first.")
        elif not user_question.strip():
            st.warning("Please provide a question about the image.")
        else:
            with st.spinner("Analyzing image..."):
                
                # Prepare image
                processed_image = prepare_image(uploaded_file)
                
                if processed_image is None:
                    st.error("Failed to process the uploaded image.")
                else:
                    # Enhanced instruction prompt
                    instruction = get_instructions(user_question)
                    
                    # Get response
                    response_text = generate_response(instruction, processed_image)
                    
                    # Display results right below the button
                    st.markdown("---")
                    st.subheader("AI Analysis:")
                    
                    if response_text.startswith("Error") or response_text.startswith("Response blocked"):
                        st.error(response_text)
                    else:
                        st.success("Analysis completed!")
                        st.write(response_text)

with col2:
    st.subheader("Preview")
    
    # Display uploaded image
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
    else:
        st.info("Upload an image to see preview here")