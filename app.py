import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pdf2image import convert_from_path

# Import your utility functions
from utils.predict_bounding_boxes import predict_bounding_boxes
from utils.manga_ocr import get_text_from_image
from utils.translate_manga import translate_manga
from utils.process_contour import process_contour
from utils.write_text_on_image import add_text

# Load the object detection model
best_model_path = "./model_creation/runs/detect/train5"
object_detection_model = YOLO(os.path.join(best_model_path, "weights/best.pt"))

app = FastAPI()

UPLOAD_DIR = "uploads"
RESULT_DIR = "result"
BOUNDS_DIR = "./bounding_box_images"
OUTPUT_IMAGES_DIR = "output_images"

# Ensure the directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(BOUNDS_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

def pdf_to_images(pdf_path, output_folder):
    images = convert_from_path(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    for i, image in enumerate(images):
        filename = f"{i+1:03}.png"
        image.save(os.path.join(output_folder, filename), "PNG")

def images_to_pdf(image_folder, output_pdf):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    if not image_files:
        print("No images found in the folder.")
        return
    first_image = Image.open(os.path.join(image_folder, image_files[0])).convert('RGB')
    images = [Image.open(os.path.join(image_folder, img)).convert('RGB') for img in image_files[1:]]
    first_image.save(output_pdf, save_all=True, append_images=images)
    print(f"PDF saved as {output_pdf}.")

def list_files_in_directory(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths

def process_manga_pdf(pdf_path):
    pdf_to_images(pdf_path, OUTPUT_IMAGES_DIR)
    pages = list_files_in_directory(OUTPUT_IMAGES_DIR)

    for page in pages:
        loop_X = 0
        x = True
        while x and loop_X < 3:
            try:
                # Clear the bounding box images directory
                if os.path.exists(BOUNDS_DIR):
                    [os.unlink(os.path.join(BOUNDS_DIR, f)) for f in os.listdir(BOUNDS_DIR) if os.path.isfile(os.path.join(BOUNDS_DIR, f))]

                name_x = os.path.basename(page)
                image = Image.open(page)
                results = predict_bounding_boxes(object_detection_model, page)
                image = np.array(image)

                previous_positions = []
                copter_diff = 100

                for result in results:
                    x1, y1, x2, y2 = result[:4]
                    is_close = False
                    for px1, py1, px2, py2 in previous_positions:
                        if (
                            abs(x1 - px1) <= copter_diff
                            and abs(x2 - px2) <= copter_diff
                            and abs(y1 - py1) <= copter_diff
                            and abs(y2 - py2) <= copter_diff
                        ):
                            is_close = True
                            break
                    if not is_close:
                        detected_image = image[int(y1):int(y2), int(x1):int(x2)]
                        im = Image.fromarray(np.uint8(detected_image))
                        text = get_text_from_image(im)
                        detected_image, cont = process_contour(detected_image)
                        text_translated = translate_manga(text)
                        add_text(detected_image, text_translated, cont)

                    previous_positions.append((x1, y1, x2, y2))

                result_image = Image.fromarray(image)
                result_image.save(f"./{RESULT_DIR}/{name_x}")
                x = False
            except Exception as e:
                print(e)
                loop_X += 1

    final_pdf_path = os.path.join(RESULT_DIR, "final.pdf")
    images_to_pdf(RESULT_DIR, final_pdf_path)
    return final_pdf_path

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Process the PDF and generate the translated version
    output_pdf = process_manga_pdf(file_location)
    
    # Return the translated PDF for download
    return FileResponse(path=output_pdf, filename="translated_" + file.filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)