from ultralytics import YOLO
import os


def predict_image(
    image_path,
    model_path=r"runs/classify/train7/weights/best.pt",
    show=True,
    save_result=True,
    save_dir="predict_results",
):
    """
    Predict cattle breed from an input image using a trained YOLO classification model.
    """

    # Load the trained classification model
    model = YOLO(model_path)

    # Run prediction on the image(s)
    results_list = model(image_path)

    for results in results_list:
        # Show visualization (image + predicted label) if requested
        if show:
            results.show()

        # Save visualization if requested
        if save_result:
            os.makedirs(save_dir, exist_ok=True)
            results.save(save_dir)

        # Extract predicted class name and confidence
        names = results.names          # {class_id: class_name}
        probs = results.probs          # probability tensor
        pred_class_idx = int(probs.argmax())
        pred_class_name = names[pred_class_idx]
        pred_confidence = float(probs[pred_class_idx])

        print(f"Predicted Breed: {pred_class_name}, Confidence: {pred_confidence:.4f}")

    return results_list


if __name__ == "__main__":
    # Ask user for image path
    user_path = input("Enter full path to the image: ").strip()

    # Convert to raw-style path safely (just ensure backslashes work)
    user_path = user_path.replace("\\", "/")

    if not os.path.isfile(user_path):
        print("Error: file not found. Check the path and try again.")
    else:
        predict_image(user_path)
