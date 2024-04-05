import os
import sys
import shutil
import numpy as np
import requests
import argparse
import json
import logging
import multiprocessing
from tqdm import tqdm

# Create a shared variable to store the number of images downloaded per class
class_img_counter = multiprocessing.Value('i', 0)

def get_image(base_dir, img_url, class_name, images_per_class, only_flickr:bool=True):
    
    with class_img_counter.get_lock():
        if class_img_counter.value >= images_per_class:
            return
    
    if only_flickr and not 'flickr' in img_url:
        return 

    try:
        img_resp = requests.get(img_url, timeout=1)
    except:
        return 

    if not 'content-type' in img_resp.headers:
        return 

    if not 'image' in img_resp.headers['content-type']:
        logging.debug("Not an image")
        return 

    if (len(img_resp.content) < 1000):
        return 

    img_name = img_url.split('/')[-1]
    img_name = class_name.replace(' ', '_').lower() + '_' + img_name.split("?")[0]

    if (len(img_name) <= 1):
        return 
    
    img_file_path = os.path.join(base_dir, img_name)

    with open(img_file_path, 'wb') as img_f:
        img_f.write(img_resp.content)
    
    with class_img_counter.get_lock():
        class_img_counter.value += 1
        
    return 
    
def create_dir(base_dir_path:str, folder_name:str) -> str:
    save_dir = os.path.join(base_dir_path, folder_name)
    if os.path.exists(save_dir):
        print(f"[WARNING] folder {save_dir} already exists!")
        user_input = input(
            "Delete existing images? (otherwise images are simply added to existing images) [y(es)|n(o)]: "
        ).lower().strip()
        if user_input in ['yes', 'y']:
            shutil.rmtree(save_dir)
            print(f"[INFO] Deleted existing images in folder: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    tqdm.write(f"[INFO] Directory {save_dir} is used to save images for {folder_name}")
    return save_dir

def main(args):
    
    # create saving directory (if not already exists)
    save_dir = os.path.normpath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Directory {os.path.join(os.getcwd(), save_dir)} is used to save images")

    # load ImageNet1K class information from json file 
    if os.path.exists(args.imagenet1k_class_info_dict):
        with open(args.imagenet1k_class_info_dict) as file:
            imagenet1k_class_info_dict = json.load(file)
    else:
        print(f'[ERROR] file {args.imagenet1k_class_info_dict} does not exists')
        exit()

    # Determine the classes to scrape images from

    # list to store classes to scrape in tuples (class_name, wnid)
    classes_to_scrape = []

    # if args.class_list is given, use these classes and add them to the list
    if args.class_list is not None:
        for class_name in args.class_list.split(','):
            cls_name = class_name.strip().replace(' ', '_')
            class_info = imagenet1k_class_info_dict.get(cls_name, None)
            if class_info is None:
                print(f'Class {cls_name} not found in ImageNet-1K')
                exit()
            classes_to_scrape.append((cls_name, class_info['wnid']))
                
    # if args.class_list is not given, randomly select args.number_of_classes classes and for
    # each of these classes, download args.images_per_class images
    elif args.class_list is None:
        for class_name, class_info in imagenet1k_class_info_dict.items():
            # check whether the class is expected to yield enough images
            if args.scrape_only_flickr:
                if int(class_info['flickr_img_url_count']) * 0.9 > args.images_per_class:
                    classes_to_scrape.append((class_name, class_info['wnid']))
            else:
                if int(class_info['img_url_count']) * 0.8 > args.images_per_class:
                    classes_to_scrape.append((class_name, class_info['wnid']))
        # check whether there are enough classes with at least args.number_of_classes images
        if (len(classes_to_scrape) < args.number_of_classes):
            print(
                f"[ERROR] With {args.images_per_class} images per class there are only {len(classes_to_scrape)} to choose from.",
                f"Decrease number of classes or decrease images per class."
            )
            exit()
        # randomly select classes from the potential class pool
        random_indices = np.random.choice(len(classes_to_scrape), args.number_of_classes, replace=False)
        classes_to_scrape = np.array(classes_to_scrape)[random_indices]

    # if args.class_list is not given, store all images in a folder called 'random' (if folder 
    # already exists, user can choose between adding images or deleting existing images)
    if args.class_list is None:
        save_dir = create_dir(args.save_dir, 'random')
    
    # iterate over all classes to scrape and download images (using multiprocessing)
    for (class_name, class_wnid) in (pbar := tqdm(classes_to_scrape)):
        # update progress bar
        pbar.set_description(f'[INFO] downloading images for class {class_name}')
        # receive the image-urls
        resp = requests.get(f'http://www.image-net.org/api/imagenet.synset.geturls?wnid={class_wnid}')
        img_urls = [url.decode('utf-8') for url in resp.content.splitlines()]
        # variable to make sure only args.images_per_class images are downloaded for each class
        # note that when using multiprocessing, the actual number of images downloaded is likely a bit
        # higher than this value, because some processes will still finish even though this value is exceeded
        with class_img_counter.get_lock():
            class_img_counter.value = 0
        
        # if args.class_list is given, create one folder (named after the class) for each class
        if args.class_list is not None:
            save_dir = create_dir(args.save_dir, class_name)
        
        # defined arguments for multiprocessing
        pool_args = [
            (
                save_dir,
                img_url,
                class_name,
                args.images_per_class,
                args.scrape_only_flickr
            ) for img_url in img_urls
        ]
        
        # Create a pool of processes
        with multiprocessing.Pool(processes=args.number_of_processes) as pool:
            # (try to) download images for each url in list img_urls
            pool.starmap(get_image, pool_args)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--save_dir',
        type=str,
        help='base directory where the image are saved (in corresponding folders)', 
        default='./sourceDir'
    )
    parser.add_argument(
        '--imagenet1k_class_info_dict',
        type=str,
        help='maps each class in ImageNet-1K to class index, wnid and url count', 
        default='./imagenet1k_class_info.json'
    )
    parser.add_argument(
        '--scrape_only_flickr',  
        type=bool,
        help='whether to only scrape images from FlickR', 
        default=True,
    )
    parser.add_argument(
        '--class_list', 
        type=str,
        help='class names to download images from (seperated by ",")',
        default=None
    )
    parser.add_argument(
        '--number_of_classes', 
        type=int,
        help='if class_list is not specified, random classes are selected from ImageNet1K',
        default=500, 
    )
    parser.add_argument(
        '--images_per_class', 
        type=int,
        help='number of images to download for each class',
        default=1, 
    )
    parser.add_argument(
        '--number_of_processes', 
        type=int,
        help='number of processes used for multiprocessing',
        default=8, 
    )
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))