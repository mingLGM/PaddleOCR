# # Initialize PaddleOCR instance
# from paddleocr import PaddleOCR
# ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False)

# # Run OCR inference on a sample image 
# result = ocr.predict(
#     input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

# # Visualize the results and save the JSON results
# for res in result:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")
    

from paddleocr import TextRecognition
model = TextRecognition(model_name="PP-OCRv5_server_rec")
output = model.predict(input="general_ocr_rec_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")