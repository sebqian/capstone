"""Converts pdf data into text data or more."""
from pathlib import Path
import fitz


class PDFtoTextGenerator():

    def __init__(self) -> None:
        pass

    def pdf_file_to_text(self, pdf_file_path: Path) -> str:
        """Reads a single pdf file and converts it to string."""
        if not pdf_file_path.exists():
            raise FileNotFoundError(f'{pdf_file_path} does not exist.')
        text = ''
        with fitz.open(str(pdf_file_path)) as doc:  # open document
            text = chr(12).join([page.get_text() for page in doc])
        return text

    def read_pdf(self, pdf_file_path: Path, output_file: Path):
        """Reads pdf, converts to text and saves."""
        text = self.pdf_file_to_text(pdf_file_path)
        with open(output_file, 'w') as f:
            f.write(text)
