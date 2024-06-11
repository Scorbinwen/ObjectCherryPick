from glob import glob
import shutil
import os
import hashlib
import PIL.Image as Image

# copy the contents of the demo.py file to  a new file called demo1.py
def Pick_Bad_Data():
    data_path = "data"
    raw_data_path = "meta"
    picked_data_path = "picked_data"
    txts = glob(os.path.join(data_path, raw_data_path, "*.txt"))
    count = 0
    for txt in txts:
        name = txt.split('/')[-1].split('.')[0]

        name_a = name+'_a.png'
        name_b = name+'_b.png'
        print(name_a)
        if os.path.exists(name_a) and os.path.exists(name_b):
            with open(txt, 'r') as f:
                tmp = f.readlines()
            ann = eval(tmp[0].strip())
            if ann == [1, 0]:
                # put name_a to picked_data
                shutil.copy(name_b, os.path.join(data_path, picked_data_path))
            elif ann == [0, 1]:
                shutil.copy(name_a, os.path.join(data_path, picked_data_path))
            else:
                shutil.copy(name_a, os.path.join(data_path, picked_data_path))
                shutil.copy(name_b, os.path.join(data_path, picked_data_path))
            print(ann)
            count += 1



def calculate_hash(image_path, hash_function=hashlib.md5):
    """
    Calculate the hash of an image.
    :param image_path: Path to the image file.
    :param hash_function: Hash function from hashlib (default is md5).
    :return: Hash of the image.
    """
    with Image.open(image_path) as img:
        # Convert image to a consistent format and size for hashing
        img = img.convert('RGB')
        # img = img.resize((256, 256))
        # Calculate the hash
        img_hash = hash_function(img.tobytes()).hexdigest()
    return img_hash

def find_and_remove_duplicates(folder_path):
    """
    Find and remove duplicate images in a folder.
    :param folder_path: Path to the folder containing images.
    """
    hashes = {}
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path):
            try:
                img_hash = calculate_hash(image_path)
                if img_hash in hashes:
                    print(f"Duplicate found: {filename} is a duplicate of {hashes[img_hash]}")
                    os.remove(image_path)
                    print(f"Removed: {filename}")
                else:
                    hashes[img_hash] = filename
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    Pick_Bad_Data()
    folder_path = "data/picked_data"
    find_and_remove_duplicates(folder_path)
