# Gloved vs. Ungloved Hand Detection

This project involves fine-tuning a YOLOv8 model to detect `gloved_hand` and `bare_hand` for a safety compliance system.



## Dataset

* **Name:** Gloves - Object Detection
* **Source:** [Roboflow Universe](https://universe.roboflow.com/glove-uylxg/glove-q7czq/dataset/1)
* **Description:** The dataset contains 1.7k images with two primary classes for this task: `glove` and `hand`. For this project, the `glove` class was treated as `gloved_hand` and the `hand` class as `bare_hand`. The images feature a variety of scenes, lighting conditions, and glove types, providing a solid foundation for training a robust model.

---

## Preprocessing and Training

The model was trained in a Google Colab environment using a Tesla T4 GPU. The entire process is documented in the `glove_hand_training.ipynb` notebook.

### Model Architecture

* **Model:** **YOLOv8s** (small version), pre-trained on the COCO dataset.
* **Reasoning:** This model was chosen as it offers an excellent balance between speed and accuracy, making it suitable for potential real-time deployment on factory video streams where inference time is a critical factor.

### Preprocessing & Data Handling

* **Data Aggregation:** All images and labels from the dataset's existing `train`, `valid`, and `test` directories were initially combined into a single master list.
* **Cross-Validation Splitting:** Instead of relying on the original splits, a **5-fold cross-validation** (`KFold` from Scikit-learn) was implemented. This strategy ensures that the final performance metrics are robust and not skewed by a single, potentially biased, data split. For each fold, the entire dataset was re-split into a new training and validation set.
* **Image Augmentation:** The Ultralytics training pipeline was configured with `augment=True`. As seen in the training logs, this applied a variety of on-the-fly augmentations to improve model generalization, including:
    * Blur and Median Blur
    * Conversion to Grayscale
    * CLAHE (Contrast Limited Adaptive Histogram Equalization)
    * Standard geometric augmentations like scaling, translation, and flipping (`fliplr=0.5`).

### Training Process

For each fold, the script performed these steps:
1.  Initialized a new `yolov8s.pt` model with pre-trained COCO weights.
2.  Generated temporary `.txt` files defining the training and validation image paths for the current fold.
3.  Created a `data.yaml` file pointing to these temporary files.
4.  Executed the `model.train()` function with the hyperparameters above.
5.  Saved the best-performing model weights (`best.pt`) for that fold based on validation performance.

---

### Final Model Selection & Metrics

* **Reason for Model Selection:**
    Based on the cross-validation results, the model from **Fold 2 (`runs/fold2_train/weights/best.pt`)** was chosen because it achieved the highest **`mAP50-95` of 0.820**. While other folds had slightly higher `mAP50`, a higher `mAP50-95` indicates better performance across a wider range of Intersection over Union (IoU) thresholds, suggesting a more precise and reliable model overall.

* **Final Cross-Validation Metrics:**
    * **mAP50:** `0.9743 ± 0.0031`
    * **Precision:** `0.9735 ± 0.0097`
    * **Recall:** `0.9547 ± 0.0076`

---

## What Worked and What Didn't

### What Worked Well

* **YOLOv8s:** The model was very effective. The high mAP scores show it learned to distinguish between gloved and bare hands with high accuracy.
* **Cross-Validation:** This technique gave confidence in the model's performance and provided a clear, data-driven way to select the best-performing weights.
* **Roboflow Dataset:** The dataset was well-labeled and diverse enough to achieve good results without extensive manual data collection.

### Challenges & Areas for Improvement

* **Class Ambiguity:** Some gloves are skin-colored, which could potentially confuse the model. Further training on a more diverse set of glove colors and materials would improve robustness.
* **Occlusion & Clutter:** In a real-world factory setting, hands may be partially obscured by tools, equipment, or other objects. While the model is robust, extreme occlusion remains a challenge.
* **Motion Blur:** The current model is trained on static images. Performance on live video streams with significant motion blur could be lower. Techniques like frame averaging or models specifically designed for video could be explored.

---

## How to Run the Script

1.  **Setup Environment:**
    * Clone the repository.
    * Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Prepare Files:**
    * Place the selected model weights, **`glove_detector_best_fold2.pt`**, in the project's root directory.
    * Add your test `.jpg` images into a folder (e.g., `sample_images/`).

3.  **Execute Detection:**
    * Run the script from the command line inside the `Part_1_Glove_Detection/` directory.
    * You must provide the path to your model and input images.

    **Example Command:**
    ```bash
    python detection_script.py --model_path glove_detector_best_fold2.pt --input_dir sample_images/
    ```

4.  **Check Results:**
    * Annotated images will be saved in the `output/` folder.
    * JSON logs for each image will be saved in the `logs/` folder.