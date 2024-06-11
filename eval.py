import os

import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import metrics
from config import *
class DesignCherryPick:
    def __init__(self,
                 text,
                 model_id="IDEA-Research/grounding-dino-base",
                 device="cuda",
                 save_path="pred_dir",
                 visualize=False):
        self.model_id = model_id
        self.device = device
        self.save_path = save_path
        self.text = text
        self.visualize = visualize
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

    def get_iou(self, bbox1, bbox2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        bbox1, bbox2: list or tuple of 4 elements
            Each bounding box is represented as [x1, y1, x2, y2],
            where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

        Returns:
        float
            IoU value.
        """

        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate the (x, y) coordinates of the intersection of the two rectangles
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Calculate the area of intersection rectangle
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # Calculate the area of both bounding boxes
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # Calculate the IoU
        iou = inter_area / float(bbox1_area + bbox2_area - inter_area)

        return iou

    def visualize_predictions(self, predictions):
        """
        Visualize the prediction results by drawing bounding boxes on the image.

        Args:
        - predictions (dict): Dictionary containing prediction results with keys:
            - 'scores' (torch.Tensor): Tensor of confidence scores.
            - 'labels' (list of str): List of labels.
            - 'boxes' (torch.Tensor): Tensor of bounding box coordinates.

        Returns:
        - None
        """
        # Load the image
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a copy of the image for visualization
        vis_image = image.copy()

        # Convert tensors to numpy arrays
        scores = predictions['scores'].cpu().numpy()
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels']

        # Set color to green (RGB)
        green_color = (0, 255, 0)  # OpenCV uses BGR format by default

        # Draw bounding boxes and labels on the image
        for i, (score, box) in enumerate(zip(scores, boxes)):
            if score < 0.1:  # You can set a threshold to filter low-confidence boxes
                continue

            x_min, y_min, x_max, y_max = [int(coord) for coord in box]

            # Draw bounding box
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), green_color, 2)

            # Draw label and score
            label_text = f"{labels[i]}: {score:.2f}" if labels[i] else f"Score: {score:.2f}"
            cv2.putText(vis_image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 2)

        # Save the image if save_path is provided
        if self.save_path:
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.save_path, self.filename+"_pred.png"), vis_image_bgr)
    def predict_object(self):
        image = Image.open(self.image_path)
        starttime = time.time()
        inputs = self.processor(images=image, text=self.text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        endtime = time.time()
        valid_ind = [index for index, element in enumerate(predictions["labels"]) if element != ""]
        print("valid_ind", valid_ind)
        print("predictions before", predictions)
        predictions = {"scores":torch.tensor([predictions["scores"].tolist()[index] for index in valid_ind], device=self.device),
                       "labels":[predictions["labels"][index] for index in valid_ind],
                       "boxes":torch.tensor([predictions["boxes"].tolist()[index] for index in valid_ind], device=self.device)}
        print("predictions after", predictions)
        print("detection last time:{:5f}s".format(endtime - starttime))

        predictions = self.customed_nms(predictions)
        if self.visualize:
            self.visualize_predictions(predictions)
        return predictions

    def check_if_multiple(self, image_path):
        self.image_path = image_path
        dir_path, filename_with_ext = os.path.split(self.image_path)
        filename_with_ext = filename_with_ext
        filename, ext = os.path.splitext(filename_with_ext)

        self.dir_path = dir_path
        self.filename = filename
        self.ext = ext

        predictions = self.predict_object()
        bboxes = predictions["boxes"]
        image = cv2.imread(self.image_path)
        if len(bboxes) >= THREE_BOXES:
            return MULTIPLE_OBJECT
        if len(bboxes) == TWO_BOXES:
            if self.get_iou(bboxes[0], bboxes[1]) > iou_threshold or self.check_reflect(image, bboxes[0], bboxes[1]) == REFLECTION:
                return SINGLE_OBJECT
            else:
                return MULTIPLE_OBJECT
        else:
            return SINGLE_OBJECT

    def check_similarity(self, image, box_up, box_down):
        # Crop the second bounding box area
        x2_min, y2_min, x2_max, y2_max = [int(coord) for coord in box_down]
        crop2 = image[y2_min:y2_max, x2_min:x2_max]
        cv2.imwrite(os.path.join(self.dir_path, "crop2.jpg"), crop2)

        # crop and flip box_up refer to box_down
        # keep box_up's x identical with box_down's, while y unchanged
        # Crop the first bounding box area
        x1_min, y1_min, x1_max, y1_max = [int(coord) for coord in box_up]
        new_x1_min = x2_min
        new_y1_min = y1_max - (y2_max - y2_min)
        new_x1_max = x2_max
        new_y1_max = y1_max
        crop1 = image[new_y1_min:new_y1_max, new_x1_min:new_x1_max]
        crop1 = cv2.flip(crop1, 0)
        cv2.imwrite(os.path.join(self.dir_path, "crop1.jpg"), crop1)

        crop1_resized = crop1
        crop2_resized = cv2.resize(crop2, (crop1_resized.shape[1], crop1_resized.shape[0]),
                                   interpolation=cv2.INTER_AREA)

        # Convert images to grayscale
        crop1_gray = cv2.cvtColor(crop1_resized, cv2.COLOR_BGR2GRAY)
        crop2_gray = cv2.cvtColor(crop2_resized, cv2.COLOR_BGR2GRAY)
        # Calculate SSIM
        ssim_score = metrics.structural_similarity(crop1_gray, crop2_gray, full=True)
        print(f"SSIM Score: ", round(ssim_score[0], 2))
        return round(ssim_score[0], 2)

    def check_reflect(self, image, box_a, box_b):
        a_tl_x, a_tl_y, a_br_x, a_br_y = box_a
        b_tl_x, b_tl_y, b_br_x, b_br_y = box_b
        if abs(a_tl_x - b_tl_x) < close_threshold and abs(a_br_x - b_br_x) < close_threshold:
            if a_tl_y < b_tl_y:
                box_up = box_a
                box_down = box_b
            else:
                box_down = box_a
                box_up = box_b
            if self.check_similarity(image, box_up, box_down) > sim_threshold:
                return REFLECTION
        return MULTIPLE_OBJECT

    import torch

    def calculate_area(self, bbox):
        # Calculate the area of a bounding box
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height

    def calculate_intersection_area(self, bbox_a, bbox_b):
        # Calculate the intersection area of two bounding boxes
        x_left = max(bbox_a[0], bbox_b[0])
        y_top = max(bbox_a[1], bbox_b[1])
        x_right = min(bbox_a[2], bbox_b[2])
        y_bottom = min(bbox_a[3], bbox_b[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        return intersection_area

    def is_bbox_a_in_bbox_b(self, bbox_a, bbox_b, threshold=0.9):
        # Calculate the area of bbox_a
        area_a = self.calculate_area(bbox_a)

        # Calculate the intersection area
        intersection_area = self.calculate_intersection_area(bbox_a, bbox_b)

        # Check if the intersection area is at least 90% of the area of bbox_a
        return intersection_area >= threshold * area_a

    def is_affiliated(self, bbox_a, bbox_b):
        if self.is_bbox_a_in_bbox_b(bbox_a, bbox_b) or self.is_bbox_a_in_bbox_b(bbox_b, bbox_a):
            return True
        return False

    def customed_nms(self, prediction):
        # Extract scores
        scores = prediction['scores']

        # Get the sorted indices of scores in ascending order
        sorted_indices = torch.argsort(scores, descending=False)

        # Reorder labels and boxes based on sorted indices
        sorted_scores = scores[sorted_indices]
        # sorted_labels = prediction['labels'][sorted_indices]
        sorted_boxes = prediction['boxes'][sorted_indices]

        res_bboxes = torch.empty((0, 4), device=sorted_boxes.device)
        res_scores = torch.tensor([], device=sorted_scores.device)
        for bbox_a, score_a in zip(sorted_boxes, sorted_scores):
            remove_index = []
            for rmv_idx, (bbox_b, score_b) in enumerate(zip(res_bboxes, res_scores)):
                if self.is_affiliated(bbox_a, bbox_b):
                    remove_index.append(rmv_idx)
            filtered_indices = [i for i in range(len(res_bboxes)) if i not in remove_index]
            res_bboxes = res_bboxes[filtered_indices]
            res_bboxes = torch.cat((res_bboxes, bbox_a.unsqueeze(0)), dim=0)

            res_scores = res_scores[filtered_indices]
            res_scores = torch.cat((res_scores, score_a.unsqueeze(0)), dim=0)
        # return predictions
        # Create sorted prediction dictionary
        res_prediction = {
            'scores': res_scores,
            'labels': [prediction["labels"][0]] * len(res_bboxes),
            'boxes': res_bboxes
        }
        print("res_prediction", res_prediction)
        return res_prediction

if __name__ == "__main__":
    imgs_dir = "data\\picked_data"
    cherrypick = DesignCherryPick(text="perfume bottle", visualize=True)
    # cherrypick.check_if_multiple("perfume_bottle.png")
    for image_path in os.listdir(imgs_dir):
        if cherrypick.check_if_multiple(os.path.join(imgs_dir, image_path)) == MULTIPLE_OBJECT:
            text = "MULTIPLE_OBJECT"
        else:
            text = "SINGLE_OBJECT"
        position = (15, 35)  # 10 pixels from the left, 30 pixels from the top (adjust as needed)
        image = cv2.imread(os.path.join("pred_dir", cherrypick.filename+"_pred.png"))
        cv2.putText(image,
                    text,
                    org=position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=3,
                    lineType=cv2.LINE_4)
        cv2.imwrite(os.path.join("pred_dir", cherrypick.filename+"_pred.png"), image)