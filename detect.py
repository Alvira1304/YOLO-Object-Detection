from ultralytics import YOLO
from collections import Counter

# 1. Load the pretrained YOLO model
model = YOLO("yolo11n.pt")

# 2. Detect objects in the image
results = model("test_image.png", conf=0.5)

# 3. Create a list to store detected object names
detected_objects = []

# 4. Print each detected object
for result in results:
    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        # Add object name to our list
        detected_objects.append(class_name)

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        print(
            f"Object: {class_name}, "
            f"Confidence: {confidence:.2f}, "
            f"Box: ({x1}, {y1}, {x2}, {y2})"
        )

# 5. Count each type of object
object_counts = Counter(detected_objects)

# 6. Display the object counts
print("\n----- OBJECT COUNT -----")

for object_name, count in object_counts.items():
    print(f"{object_name}: {count}")

# 7. Save the detected image
results[0].save(filename="result.jpg")

print("\nDetection completed!")
print("Result saved as result.jpg")