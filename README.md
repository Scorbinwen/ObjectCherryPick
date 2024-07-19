# DesignCherryPick
> Detect if given image contains only one target object, it's based on pretrained multimodal object detection framework [GroundingDino](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)

## Quickstart
run eval.py to detect given image:
> python eval.py

single object             |  multiple object
:-------------------------:|:-------------------------:
 ![00010419_49c0b62_b_pred](https://github.com/Scorbinwen/DesignCherryPick/assets/29889669/088e6b95-846e-4484-aed2-8c6b80e24315)|  ![00009815_242d385_b_pred](https://github.com/Scorbinwen/DesignCherryPick/assets/29889669/99d37fd2-e957-4c83-93f4-055a6a38a19a)

before Custom NMS             |  after Custom NMS
:-------------------------:|:-------------------------:
![图片1](https://github.com/user-attachments/assets/dceb1280-ffbb-42db-a418-d560246abaa4)|![图片2](https://github.com/user-attachments/assets/9c35e25c-16b5-4568-b8a9-417b249f4736)

## Disclaimer

This is not an officially supported Google product.
