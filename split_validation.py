import os
import shutil
import random

test_dir = "Testing"
val_dir = "Validation"
classes = ["glioma", "meningioma", "notumor", "pituitary"]
#RANDOM_SEED = 42

#creating folders.
for class_name in classes:
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(val_class_dir, exist_ok=True)
    print(f"Created {val_class_dir}")

# Split each class 50/50
for class_name in classes:
    test_class_dir = os.path.join(test_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    images = os.listdir(test_class_dir)
    images = [img for img in images if os.path.isfile(os.path.join(test_class_dir, img))]
    
    #shuffling and split
    random.shuffle(images)
    split_point = len(images) // 2
    val_images = images[:split_point]
    
    print(f"\n{class_name}:")
    print(f"  Total images: {len(images)}")
    print(f"  Moving to Validation: {len(val_images)}")
    print(f"  Keeping in Testing: {len(images) - len(val_images)}")
    
    for img in val_images:
        src = os.path.join(test_class_dir, img)
        dst = os.path.join(val_class_dir, img)
        shutil.move(src, dst)

print("\nVerification of counts")
for class_name in classes:
    val_count = len(os.listdir(os.path.join(val_dir, class_name)))
    test_count = len(os.listdir(os.path.join(test_dir, class_name)))
    print(f"  {class_name}: Validation={val_count}, Testing={test_count}")
