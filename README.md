### Handwriting Reader ###
Custom trained OCR capable of reading handwritten words. It uses BiLSTM architecture with CTC Loss function.

To train the model, use the training package's train.ipynb file.

### Examples ###
This handwriting reader can be called directly via running the below code:
```python
from handwriting_recognition.engine import HandwritingRecognizer
import cv2 as cv

hr_ocr = HandwritingRecognizer()
out1 = hr_ocr.inference("sample_data/a02-000-00-02.png")
print(out1)

# Or pass the image as numpy object
im = cv.imread("sample_data/a02-000-00-02.png")

out2 = hr_ocr.inference("")
print(out2)
```
