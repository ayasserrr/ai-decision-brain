import os
import sys
from pathlib import Path
from typing import List, Dict
import PyPDF2

try:
    from configs import Config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from configs import Config


class DocumentLoader:
    """Handles loading documents from various formats"""
    
    def __init__(self):
        self.documents_dir = Config.DOCUMENTS_DIR
        self.supported_formats = ['.pdf']
    
    def load_pdf_file(self, file_path: Path) -> str:
        """Load text from a .pdf file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    
    def load_single_document(self, file_path: Path) -> Dict[str, str]:
        """Load a single document and return its content with metadata"""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                content = self.load_pdf_file(file_path)
            else:
                print(f"[WARNING] Unsupported file format: {file_path.name}")
                return None
            
            return {
                'content': content,
                'filename': file_path.name,
                'filepath': str(file_path),
                'file_type': file_extension
            }
        
        except Exception as e:
            print(f"[ERROR] Error loading {file_path.name}: {str(e)}")
            return None
    
    def load_all_documents(self) -> List[Dict[str, str]]:
        """Load all documents from the documents directory"""
        if not self.documents_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.documents_dir}")
        
        documents = []
        files = [f for f in self.documents_dir.iterdir() if f.is_file()]
        
        print(f"Loading documents from: {self.documents_dir}")
        print(f"Found {len(files)} files")
        print("-" * 50)
        
        for file_path in files:
            if file_path.suffix.lower() in self.supported_formats:
                print(f"Loading: {file_path.name}...", end=" ")
                doc = self.load_single_document(file_path)
                if doc:
                    documents.append(doc)
                    print(f"SUCCESS ({len(doc['content'])} characters)")
                else:
                    print("FAILED")
            else:
                print(f"[SKIP] Unsupported format: {file_path.name}")
        
        print("-" * 50)
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def get_document_stats(self, documents: List[Dict[str, str]]) -> Dict:
        """Get statistics about loaded documents"""
        total_chars = sum(len(doc['content']) for doc in documents)
        file_types = {}
        
        for doc in documents:
            file_type = doc['file_type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'avg_characters': total_chars // len(documents) if documents else 0,
            'file_types': file_types
        }
    
    def display_stats(self, documents: List[Dict[str, str]]):
        """Display statistics about loaded documents"""
        stats = self.get_document_stats(documents)
        
        print("\n" + "=" * 50)
        print("Document Statistics")
        print("=" * 50)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Characters: {stats['total_characters']:,}")
        print(f"Average Characters per Document: {stats['avg_characters']:,}")
        print(f"\nFile Types:")
        for file_type, count in stats['file_types'].items():
            print(f"  {file_type}: {count} file(s)")
        print("=" * 50)


# Test function
if __name__ == "__main__":
    loader = DocumentLoader()
    documents = loader.load_all_documents()
    loader.display_stats(documents)