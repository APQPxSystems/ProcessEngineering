import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io

# Streamlit Configurations
st.set_page_config(page_title="DOE | kentjkdigitals", layout="wide")
hide_st_style = """
                <style>
                #MainMenu {visibility:hidden;}
                footer {visibility:hidden;}
                header {visibility:hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Remove top white space
st.markdown("""
        <style>
            .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

# Function to convert PDF to PNG
def convert_pdf_to_png(pdf_file):
    # Read the PDF from the uploaded file as a byte stream
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    # Create a list to store image objects
    images = []
    
    # Convert each page to an image (PNG)
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        
        # Render the page to an image (pixmap)
        pixmap = page.get_pixmap()
        
        # Convert the pixmap to a PIL Image
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        images.append(img)
    
    return images

# Streamlit App
def main():
    st.title("PDF to PNG Converter")
    
    # File uploader to upload the PDF
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Call the function to convert PDF to PNG
        images = convert_pdf_to_png(uploaded_file)
        
        # Show the images and give the option to download them
        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}", use_container_width=True)
            
            # Convert image to bytes and provide download button
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            st.download_button(
                label=f"Download Page {i+1} as PNG",
                data=img_byte_arr,
                file_name=f"page_{i+1}.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
