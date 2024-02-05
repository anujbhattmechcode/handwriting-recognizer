from handwriting_recognition.engine import HandwritingRecognizer

if __name__ == "__main__":
    hr_ocr = HandwritingRecognizer()
    out = hr_ocr.inference("sample_data/a02-000-00-02.png")
    print(out)
