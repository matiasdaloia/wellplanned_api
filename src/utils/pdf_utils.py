import base64
import io

import fitz
from PIL import Image


def pdf_to_base64_images(file_content: bytes | io.BytesIO) -> list[str]:
    """
    Convert all pages of a PDF to base64 encoded PNG images.
    Returns a list of base64 strings, one for each page.
    """
    if isinstance(file_content, bytes):
        file_content = io.BytesIO(file_content)

    pdf_document = fitz.open(stream=file_content, filetype="pdf")
    base64_images = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(base64_image)

    pdf_document.close()
    return base64_images
