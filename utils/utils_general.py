from torch.utils.data.dataloader import default_collate
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.amg import remove_small_regions
import torch
import re, hashlib
import timm
import numpy as np
from scipy.linalg import subspace_angles, orth, solve, norm
import json
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
import scipy.stats as stats
from sklearn.decomposition import sparse_encode
from tqdm import tqdm
from contextlib import ExitStack
from scipy import sparse
from joblib import Parallel, delayed
import random
from skimage.morphology import binary_dilation, square, binary_erosion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_model(model_name:str)->torch.nn.Module:
    
    model = timm.create_model(model_name, pretrained=True).to(DEVICE)
    model.eval()
    
    return model

def load_sam_mask_generator(sam_type='vit_h', sam_checkpoint='sam_vit_h_4b8939.pth', **kwargs):
  sam_model = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
  sam_model.to(device=DEVICE)

  return SamAutomaticMaskGenerator(
      model=sam_model,
      **kwargs
  )

def expand_mask(mask, filter_size:int=14):
    return binary_dilation(
        mask, square(filter_size)
    )
    
def shrink_mask(mask, filter_size:int=14):
    return binary_erosion(
        mask, square(filter_size)
    )
    
def get_unique_concept_masks(sorted_masks:np.ndarray, min_area_ration:float=0.01):
    concept_masks_by_idxs = np.zeros_like(sorted_masks[0])
    for mask in sorted_masks:
        unqiue_concept_idx = random.randint(0, 100000)
        tmp_concept_masks = (concept_masks_by_idxs + mask * unqiue_concept_idx).copy()
        for region_idx in np.unique(tmp_concept_masks):
            region_mask = tmp_concept_masks == region_idx
            # first check if the region was previously deleted
            if np.sum(region_mask) == 0 or region_idx == 0:
                continue
            # next, check whether the region is too small
            if np.mean(region_mask) < min_area_ration:
                # check whether the region mask is part of the current concept mask
                if np.all(np.logical_or(np.logical_not(region_mask), mask)):
                    # if that is the case, assign the region back to the original mask
                    tmp_concept_masks -= region_mask * unqiue_concept_idx
                # otherwise, the region was newly created in another mask
                else:
                    # thus, we reset the region to the original mask
                    tmp_concept_masks[concept_masks_by_idxs==unqiue_concept_idx] = unqiue_concept_idx
            
        concept_masks_by_idxs = tmp_concept_masks.copy()
    # as postprocessing step, check whether there are large regions in the mask not covered
    uncovered_area = np.logical_not(expand_mask(concept_masks_by_idxs > 0, 8))
    # if that is the case, assign this region a unique concept idx
    if np.mean(uncovered_area) > min_area_ration:
        concept_masks_by_idxs[uncovered_area] = random.randint(0, 100000)
    # concept_masks_by_idxs[expand_mask(concept_masks_by_idxs == 0, 1)] = 0
    # finally return a list of masks that uniquly cover the whole image
    return_masks:list[np.ndarray] = []
    for concept_idx in np.unique(concept_masks_by_idxs):
        # remove small disconnected regions (islands) from the concept mask
        concept_mask, modified = remove_small_regions(
            concept_masks_by_idxs == concept_idx, 
            np.prod(concept_masks_by_idxs.shape)*min_area_ration/2, 
            mode='islands'
        )
        # if np.mean(concept_mask) > 0.25:
        #     concept_mask = shrink_mask(concept_mask.copy())
        if np.mean(concept_mask) > min_area_ration and concept_idx != 0:
            return_masks.append(concept_mask.astype(np.float32))
    return return_masks

def get_activation_hook(layer, cls_idx:int=None):
    activations = []
    def hook(model, input, output):
        if isinstance(output, tuple): # in case of backward hook
            output = output[0]
        activations.append(output.detach().cpu().numpy())
    if cls_idx is None:
        handle = layer.register_forward_hook(hook)
    else:
        handle = layer.register_full_backward_hook(hook)
    return activations, handle

def compute_activations(model, layer_name, dataloader, cls_idx:int=None)->tuple[np.ndarray,np.ndarray]:
    
    # Attach the hook to the target layer
    activations, handle = get_activation_hook(
        model.get_submodule(layer_name),
        cls_idx
    )
    logits= []

    with ExitStack() as stack:
        # Only enable torch.no_grad() if cls_idx is not set
        if cls_idx is None:
            stack.enter_context(torch.no_grad())
        
        # Forward pass through the model to compute activations
        for inputs in tqdm(dataloader, desc="[INFO] Computing activations"):
            output = model(inputs)
            logits.extend(output.detach().clone().cpu().numpy())
            if cls_idx != None:
                model.zero_grad()
                target = torch.full((output.size(0),), cls_idx, dtype=torch.long, device=DEVICE)
                # based on the TCAV implementation
                loss = torch.nn.functional.cross_entropy(output, target) #, reduction='mean')
                loss.backward()
        
    # Remove the hook after computation to clean up
    handle.remove()
    
    return np.concatenate(activations, axis=0), np.array(logits)

def custom_collate(batch):
    if isinstance(batch[0], tuple):
        batch_segments, batch_masks, masking_modes = zip(*batch)
        return (
            default_collate(batch_segments).to(DEVICE), 
            default_collate(batch_masks).to(DEVICE), 
            masking_modes[0]
        )
    else:
        return default_collate(batch).to(DEVICE)
        

def create_filename(*args, delimiter='_', extension=''):
    """
    Creates a filename based on the values of the passed variables.
    
    Parameters:
        *args: Arbitrary number of arguments from which to create the filename.
        delimiter (str): Delimiter to use between words in the filename.
        extension (str): File extension to append to the filename.

    Returns:
        str: A sanitized and concatenated filename.
    """

    # Convert all arguments to strings and replace spaces with underscores
    filename_parts = [str(arg).replace(' ', '_') for arg in args]

    # Sanitize each part to remove any characters not allowed in filenames
    # This regex will keep only alphanumeric characters, underscores, hyphens, and periods
    sanitized_parts = [re.sub(r'[^a-zA-Z0-9_\-\.]', '', part) for part in filename_parts]

    # Concatenate the parts with the specified delimiter
    filename = delimiter.join(sanitized_parts) + extension

    return filename

def create_hashsum(filenames:list[str]) -> str:
    # concatenate all stirngs into one
    combined_filenames = '+'.join(filenames)
    # Create a hash object
    hash_obj = hashlib.sha256()
    # Update the hash object with the bytes of the combined string
    hash_obj.update(combined_filenames.encode('utf-8'))
    # Get the hexadecimal representation of the hash
    hash_hex = hash_obj.hexdigest()

    return hash_hex

def get_top_concept_segms(concept_activations, batch_sizes, n_examples):
    top_concept_segms = []
    for concept_idx in range(concept_activations.shape[1]):
        indices_sorted = np.argsort(concept_activations[:, concept_idx])[::-1]
        top_segms = []
        covered_img_idxs = set()
        for idx in indices_sorted:
            # Find which array the index belongs to
            cumsum_batches = np.cumsum(batch_sizes)
            img_idx = np.searchsorted(cumsum_batches, idx, side='right')
            if img_idx not in covered_img_idxs:
                covered_img_idxs.add(img_idx)
                segm_idx = idx - (cumsum_batches[img_idx - 1] if img_idx > 0 else 0)
                top_segms.append((img_idx, segm_idx))
            if len(top_segms) == 10:
                break
        top_concept_segms.append(top_segms)
    return top_concept_segms

def get_imagenet_class_index(class_name:str, labelpath:str='imagenet1k_class_info.json'):
    with open(labelpath, 'r') as json_file:
      imagenet1k_class_info = json.load(json_file)
    class_idx = imagenet1k_class_info[class_name.replace(' ', '_')]['class_index']
    return class_idx