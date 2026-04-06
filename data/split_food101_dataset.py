import os
import shutil

# 数据集路径
root_dir = os.path.expanduser("/home/toshi/LE/FOOD101/food-101")
images_dir = os.path.join(root_dir, "images")
train_txt = os.path.join(root_dir, "meta/train.txt")
test_txt = os.path.join(root_dir, "meta/test.txt")

# 目标目录
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")

# 创建目标目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 函数：根据划分文件复制数据
def copy_images(split_file, target_dir):
    with open(split_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        class_name, image_name = line.split("/")
        src_path = os.path.join(images_dir, class_name, f"{image_name}.jpg")
        dest_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)
        shutil.copy(src_path, dest_class_dir)

# 执行划分
print("Copying training images...")
copy_images(train_txt, train_dir)
print("Copying testing images...")
copy_images(test_txt, test_dir)
print("Dataset split completed!")
