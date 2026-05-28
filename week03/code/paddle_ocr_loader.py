from typing import List

from paddleocr import PaddleOCR
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader


class PaddleOCRReader(BaseReader):
    def __int__(self):
        self.orc = PaddleOCR(use_doc_orientation_classify=False,
                             use_doc_unwarping=False,
                             use_textline_orientation=False,)
        
    def load_data(self, file, extra_info=None) -> List[Document]:
        file_path = str(file)
        
        result = self.orc.ocr(file_path)

        all_text = []

        for res in result:
            data = res.json
            lines = data['res'].get('rec_texts', [])
            all_text.extend(lines)

        
        text = '\n'.join(all_text)
        
        
        return [Document(text=text,
                         metadata={ "file_path": file_path,
                                    "file_name": Path(file_path).name,
                                    "loader": "PaddleOCRReader",
                                }
                    )
        ]
