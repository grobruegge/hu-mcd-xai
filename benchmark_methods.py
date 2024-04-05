import os, sys, argparse
import numpy as np
import classes
from concept_explainer import ConceptExplainer
from utils import utils_general, utils_mcd
from matplotlib import pyplot as plt
import pickle
from skimage.transform import resize
from sklearn.preprocessing import normalize
import random
 
# python test_concept_faithfulness.py --cls_list airliner beach_wagon zebra container_ship police_van hummingbird Siamese_cat ox golden_retriever tailed_frog --n_val_imgs_per_cls 50 --settings_to_compare ace mcd mine --mode sdc --model_name resnet50 --batch_size 128

def iter_mask_imgs_humcd(expl_obj:ConceptExplainer, val_imgs:list[classes.ImageClass],
                            mode:str) -> list[classes.ImageClass]:
    n_concepts = len(expl_obj.concept_bases)
    segm_acts = []
    batch_sizes = []
    for v_img in val_imgs:
        batch_sizes.append(len(v_img.segments))
        for v_segm in v_img.segments:
            segm_acts.append(v_segm.model_act) 
    segm_acts = np.stack(segm_acts, axis=0)
    if len(segm_acts.shape) != 2:
        raise ValueError("[ERROR] Something went wrong.")
    
    concept_activations = expl_obj.concept_activations(
        acts=segm_acts,
        batch_sizes=batch_sizes,
        norm_batch=True,
    )
    concept_relevances = expl_obj.concept_relevances(segm_acts)

    val_imgs_masked:list[classes.ImageClass] = []
    
    b_start = 0
    for idx_img, v_img in enumerate(val_imgs):
        v_img_shape = v_img.img_numpy.shape[:2]
        # initialize a dictionary to store the segments corresponding to each concept
        concept_segms = {idx_concept:[] for idx_concept in range(n_concepts)}
        concept_activation_map = concept_activations[b_start:b_start+batch_sizes[idx_img]]
        local_relevance_map = concept_relevances[b_start:b_start+batch_sizes[idx_img]]
        
        # take the argmax over all concepts for each segment (assign each segment a concept)
        hard_concept_assign = concept_activation_map.argmax(axis=-1)
        # initialize an array to store the importance scores (exclude orth compl)
        importance_scores = np.zeros(shape=(n_concepts,)) 
        for idx_concept in range(n_concepts): # discard orth compl
            # mask for which segments are assigned this concept
            segms_mask = hard_concept_assign == idx_concept
            # based on the mask we can get the corresponding indices of the segments
            segm_idxs = np.where(segms_mask == 1)[0]
            # check whether there are segments that are assigned to the current concepts
            if len(segm_idxs) > 0:
                # we calculate the local concept importance by mean pooling over the relevance scores
                importance_scores[idx_concept] = np.mean(local_relevance_map[segm_idxs, idx_concept])
                # if that is the case, 
                concept_segms[idx_concept] = [
                    classes.SegmentClass(
                        # SAM postprocesses the mask to be slightly smaller than the object, which we revert here
                        mask=utils_general.expand_mask(v_img.segments[idx_segm].mask, filter_size=8),
                        original_image=v_img
                    ) for idx_segm in segm_idxs
                ]
        # create an image which stores the accumulated masks by iteratively deleting/adding the concept
        v_img_masked = classes.ImageClass(v_img.filename, expl_obj.max_shortest_side)
        # define the starting condition depeinding on the mode
        if mode in ['sdc', 'sdc_single']:
            # in case of Smallest Destroying Concept or Single Destroying Concept we start with the unmasked image
            mask = np.ones(shape=v_img_shape)
        elif mode in ['ssc', 'ssc_single']:
            # in case of Smallest Sufficent Concept or Signle Sufficent Concept we start with the completely masked image
            mask = np.zeros(shape=v_img_shape)
        v_img_masked.segments.append(classes.SegmentClass(mask.copy(), v_img_masked))
        # because each segments uniquly covers a certain area, we do not need to keep track of the covered area (contrary to ACE)
        # iterate over the concept in descending order of relevance
        for idx_concept in np.argsort(importance_scores)[::-1]:
        # for idx in np.argsort(scores)[::-1]:
            # filter out the concepts which are not present in the image
            if len(concept_segms[idx_concept]) > 0:
                # only consider single concept mask 
                if mode == 'sdc_single':
                    mask = np.ones_like(mask)
                if mode == 'ssc_single':
                    mask = np.zeros_like(mask)
                # iterate over all segments belonging to the current concepts 
                for segm in concept_segms[idx_concept]:
                    # segm = segms[idx]
                    if mode in ['sdc', 'sdc_single']:
                        # cut the current concept mask out of the image
                        mask *= (1 - segm.mask)
                    elif mode in ['ssc', 'ssc_single']:
                        # include the current concept mask in the image
                        mask = np.logical_or(mask, segm.mask)
                # check whether the complete image is (un-)masked
                if (np.sum(mask) == 0 and mode == 'sdc') or (np.sum(1-mask) == 0 and mode == 'ssc'):
                    # if so, break the loop
                    break
                else:
                    # append the current concept mask as segment 
                    v_img_masked.segments.append(
                        classes.SegmentClass(mask.copy(), v_img_masked)
                    )
        
        val_imgs_masked.append(v_img_masked)
        b_start += batch_sizes[idx_img]
        
    return val_imgs_masked

def iter_mask_imgs_ace(expl_obj:ConceptExplainer, val_imgs:list[classes.ImageClass],
                       mode:str) -> list[classes.ImageClass]:
    # importance scores are computed globally, no measure for local importance
    tcav_scores, _, p_values = expl_obj.get_tcav_scores()
    
    val_imgs_masked:list[classes.ImageClass] = []
    n_concepts = len(tcav_scores)

    for v_img in val_imgs:
        # shape of the current validation images
        v_img_shape = v_img.img_numpy.shape[:2]
        # dictionary that contains the segments corresponding to each concept
        concept_segm = {idx_concept:[] for idx_concept in range(n_concepts)}
        
        # iterate over all segments of the val. image (last one is the unmasked image)
        for segm in v_img.segments: 
            # get the nearest cluster of the segment activation and check whether it is a concept
            idx_concept = expl_obj.get_matching_concept_idx(segm.model_act)
            if idx_concept != None: # in case cluster was filtered
                concept_segm[idx_concept].append(segm)
                
        # create a copy of the validation image, where the segments represent the iter. masking precedure
        v_img_masked = classes.ImageClass(v_img.filename, expl_obj.max_shortest_side)
        if mode in ['sdc', 'sdc_single']:
            mask = np.ones(shape=v_img_shape)
        elif mode in ['ssc', 'ssc_single']:
            mask = np.zeros(shape=v_img_shape)
        if mode in ['sdc_single', 'ssc_single']:
            # keep track of the uncovered area
            uncovered_area = np.ones_like(mask)
        # first segment is the unmasked image
        v_img_masked.segments.append(classes.SegmentClass(mask.copy(), v_img_masked))

        # iterate over the concept in descending order of their TCAV scores
        for idx_concept in np.argsort(tcav_scores)[::-1]:
            # check whether statistical testing compared to random concept exceeds threshold (see ACE implementation)
            if p_values[idx_concept] < 0.01:
                # check whether the concept has any segments associated to it
                if len(concept_segm[idx_concept]) > 0:
                    # only consider the current mask
                    if mode == 'sdc_single':
                        mask = np.ones_like(mask)
                    if mode == 'ssc_single':
                        mask = np.zeros_like(mask)
                    # iterate over all segments belonging to the concept
                    for segm in concept_segm[idx_concept]:
                        if mode == 'sdc':
                            # cut the current concept mask out of the image
                            mask *= (1 - segm.mask)
                        elif mode == 'sdc_single':
                            # only consider the area of the current segment that has not yet been covered
                            mask *= (1 - np.logical_and(segm.mask, uncovered_area)) 
                            # update the covered area
                            uncovered_area *= (1 - segm.mask)
                        elif mode == 'ssc':
                            # include the current concept mask in the image
                            mask = np.logical_or(mask, segm.mask)
                        elif mode == 'ssc_single':
                             # only consider the area of the current segment that has not yet been covered
                            mask = np.logical_or(mask, np.logical_and(segm.mask, uncovered_area))
                            # update the covered area
                            uncovered_area *= (1 - segm.mask)
                    # check whether the complete image is (un-)masked
                    if (np.sum(mask) == 0 and mode == 'sdc') or (np.sum(1-mask) == 0 and mode == 'ssc'):
                        # if so, break the loop
                        break
                    else:
                        # otherwise, append the current accumulated mask to the masked validation images
                        v_img_masked.segments.append(classes.SegmentClass(mask.copy(), v_img_masked))
                    
        val_imgs_masked.append(v_img_masked)
    
    return val_imgs_masked

def prepare_imgs_for_mcd(images:list[classes.ImageClass]):
    for img in images:
        if len(img.segments) > 1:
            raise ValueError("Should only be one segment")
        img_act = img.segments[0].model_act
        n_channels = img_act.shape[0]
        w,h, = img_act.shape[1:]
        if len(img_act.shape) != 3:
            raise ValueError("Expected feature map of following shape (n_channels, height, width)")
        pixel_acts = img_act.transpose(1,2,0).reshape(-1, n_channels)
        pixel_acts = normalize(pixel_acts, norm='l2', axis=1)
        img.segments = []
        for p_idx, p_act in enumerate(pixel_acts):
            if not np.all(p_act == 0):
                position_mask = np.zeros((w, h))
                position_mask[p_idx // w, p_idx % h] = 1
                pixel_segm = classes.SegmentClass(
                    mask=position_mask, # mask showing the position in the feature map
                    original_image=img
                )
                pixel_segm.model_act = p_act
                img.segments.append(pixel_segm)
            
def iter_mask_imgs_mcd(expl_obj:ConceptExplainer, val_imgs:list[classes.ImageClass], 
                       mode:str) -> list[classes.ImageClass]:
    n_concepts = len(expl_obj.concept_bases)
    model_input_shape = expl_obj.model.default_cfg['input_size'][1:]
    feature_maps = []
    for v_img in val_imgs:
        feature_maps.append(v_img.segments[0].model_act)
    feature_maps = np.stack(feature_maps, axis=0)
    n_imgs = feature_maps.shape[0]
    if n_imgs != len(val_imgs):
        raise ValueError("[ERROR] Something went wrong.")
    n_channels = feature_maps.shape[1]
    w,h = feature_maps.shape[2:]
    acts_flat = feature_maps.transpose(0, 2, 3, 1).reshape(n_imgs*w*h, n_channels)
    
    concept_activations = expl_obj.concept_activations(
        acts=acts_flat,
        batch_sizes=w*h,
        norm_batch=True,
    )
    concept_activation_maps = resize(
        concept_activations.reshape(n_imgs,w,h,n_concepts+1).transpose(0,3,1,2),
        (n_imgs, n_concepts+1, *model_input_shape)
    )
    
    concept_relevances = expl_obj.concept_relevances(acts_flat)
    local_relevance_maps = resize(
        concept_relevances.reshape(n_imgs,w,h,n_concepts+1).transpose(0,3,1,2),
        (n_imgs, n_concepts+1, *model_input_shape)
    )
    
    val_imgs_masked:list[classes.ImageClass] = []
    
    for idx_img, v_img in enumerate(val_imgs):
        concept_masks = {idx_concept:None for idx_concept in range(n_concepts)}
        
        # take the argmax over all concepts for each feature vector
        hard_concept_assign = concept_activation_maps[idx_img].argmax(axis=0)

        importance_scores = np.zeros(shape=(n_concepts,)) # exclude orth compl
        for idx_concept in range(n_concepts): # discard orth compl
            mask = hard_concept_assign == idx_concept
            if np.sum(mask) > 0:
                concept_masks[idx_concept] = classes.SegmentClass(
                    mask=mask.copy(),
                    original_image=v_img
                )
            # mean pool over the local relevance scores where the concept is present
            importance_scores[idx_concept] = np.mean(local_relevance_maps[idx_img, idx_concept, mask])
        
        v_img_masked = classes.ImageClass(v_img.filename, expl_obj.max_shortest_side)
        if mode in ['sdc', 'sdc_single']:
            mask = np.ones(shape=model_input_shape)
        elif mode in ['ssc', 'ssc_single']:
            mask = np.zeros(shape=model_input_shape)
        # first segment is the unmasked image
        v_img_masked.segments.append(classes.SegmentClass(mask.copy(), v_img_masked))
        
        for idx_concept in np.argsort(importance_scores)[::-1]:
            # only consider single concept mask 
            if mode == 'sdc_single':
                mask = np.ones_like(mask)
            if mode == 'ssc_single':
                mask = np.zeros_like(mask)
            if concept_masks[idx_concept] != None:
                if mode in ['sdc', 'sdc_single']:
                    # cut the current concept mask out of the image
                    mask *= (1 - concept_masks[idx_concept].mask)
                elif mode in ['ssc', 'ssc_single']:
                    # include the current concept mask in the image
                    mask = np.logical_or(mask, concept_masks[idx_concept].mask)
                # check whether the complete image is (un-)masked
                if (np.sum(mask) == 0 and mode == 'sdc') or (np.sum(1-mask) == 0 and mode == 'ssc'):
                    # if so, break the loop
                    break
                else:
                    v_img_masked.segments.append(
                        classes.SegmentClass(mask.copy(), v_img_masked)
                    )
        
        val_imgs_masked.append(v_img_masked)
        
    return val_imgs_masked

def iter_mask_imgs_random(val_imgs:list[classes.ImageClass], mode:str, 
                          grid_size:tuple=(10, 10)) -> list[classes.ImageClass]:
    val_imgs_masked:list[classes.ImageClass] = []
    
    for v_img in val_imgs:
        # get the image shape
        img_h, img_w = v_img.img_numpy.shape[:2]
        # Calculate grid cell size
        cell_height = img_h // grid_size[0]
        cell_width = img_w // grid_size[1]
        # Initialize list to store masks
        grid_masks = []
        # Iterate over each grid cell
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Calculate coordinates of current cell
                top = i * cell_height
                bottom = min((i + 1) * cell_height, img_h)
                left = j * cell_width
                right = min((j + 1) * cell_width, img_w)
                # Create mask for current cell
                g_mask = np.zeros_like(v_img.img_numpy)
                g_mask[top:bottom, left:right] = 1
                # Append mask to list
                grid_masks.append(g_mask)
        # randomly shuffle the grid masks
        random.shuffle(grid_masks)   
        
        # create an image which stores the accumulated masks by iteratively deleting/adding the concept
        v_img_masked = classes.ImageClass(v_img.filename, min((img_h, img_w)))
        # define the starting condition depeinding on the mode
        if mode in ['sdc', 'sdc_single']:
            # in case of Smallest Destroying Concept or Single Destroying Concept we start with the unmasked image
            mask = np.ones(shape=(img_h, img_w))
        elif mode in ['ssc', 'ssc_single']:
            # in case of Smallest Sufficent Concept or Signle Sufficent Concept we start with the completely masked image
            mask = np.zeros(shape=(img_h, img_w))
        # first segment is the (un)masked image
        v_img_masked.segments.append(classes.SegmentClass(mask.copy(), v_img_masked))
        
        for grid_mask in grid_masks:
            # only consider single concept mask 
            if mode == 'sdc_single':
                mask = np.ones_like(mask)
            if mode == 'ssc_single':
                mask = np.zeros_like(mask)
            if mode in ['sdc', 'sdc_single']:
                # cut the current concept mask out of the image
                mask *= (1 - grid_mask)
            elif mode in ['ssc', 'ssc_single']:
                # include the current concept mask in the image
                mask = np.logical_or(mask, grid_mask)
            
            v_img_masked.segments.append(
                classes.SegmentClass(mask.copy(), v_img_masked)
            )
            
        val_imgs_masked.append(v_img_masked)
        
    return val_imgs_masked
             
def compute_sdc_scores(expl_obj:ConceptExplainer, num_test_imgs:int, 
                       val_dir:str, setting:str, mode:str):
    val_imgs = expl_obj._load_images(
        folderpath=os.path.join(
            expl_obj.source_dir, val_dir, f'{expl_obj.target_class}_val'
        ),
        num_imgs=num_test_imgs,
        max_shortest_side=expl_obj.max_shortest_side,
        shuffle=False
    )
    if setting != 'mcd':
        expl_obj._segment_images(
            images=val_imgs,
            segm_algo=expl_obj.segm_algo,
            cache_dir='segms_val'
        )
    # little trick to get the activation of the whole image
    else:
        for v_img in val_imgs:
            v_img.segments = [
                classes.SegmentClass(
                    mask=np.ones(v_img.img_numpy.shape[:2]),
                    original_image=v_img
                )
            ]
    if setting != 'rdm':
        expl_obj._load_or_calc_acts(
            cache_dir=None,
            identifier='segm_val',
            images=val_imgs,
            cropping_mode=expl_obj.cropping_mode,
            use_masks=expl_obj.use_masks,
            masking_mode=expl_obj.masking_mode,
            erosion_threshold=expl_obj.erosion_threshold,
            batch_size=expl_obj.batch_size,
            norm_acts=False,
            delete_zeros=False,
        )
        
    if setting == 'humcd':
        val_imgs_masked = iter_mask_imgs_humcd(expl_obj, val_imgs, mode)
    elif setting == 'ace':
        val_imgs_masked = iter_mask_imgs_ace(expl_obj, val_imgs, mode)
    elif setting == 'mcd':
        val_imgs_masked = iter_mask_imgs_mcd(expl_obj, val_imgs, mode)
    elif setting == 'random':
        val_imgs_masked = iter_mask_imgs_random(val_imgs, mode)
    
    expl_obj._load_or_calc_acts(
        cache_dir=None,
        identifier='segm_val_masked',
        images=val_imgs_masked,
        cropping_mode=0, # original scale
        use_masks=True, # use Input Masking
        masking_mode=1, # neighborhood padding
        erosion_threshold=1.0, # not used
        batch_size=expl_obj.batch_size,
        norm_acts=False,
        delete_zeros=False
    )
    
    cls_corr_preds:list = []
    cls_pixel_perc:list = []
    for img in val_imgs_masked:
        imgs_corr_pred = []
        imgs_pixel_perc = []
        for segm_idx, segm in enumerate(img.segments):
            imgs_corr_pred.append(
                segm.model_pred.argmax(-1) == utils_general.get_imagenet_class_index(expl_obj.target_class)
            )
            # Calculate the percentage of pixels covered by the mask
            current_percentage = 100 * np.mean(img.segments[segm_idx].mask)
            imgs_pixel_perc.append(current_percentage)
        cls_corr_preds.append(imgs_corr_pred)
        cls_pixel_perc.append(imgs_pixel_perc)
        
    return cls_corr_preds, cls_pixel_perc

def calc_avg_and_std(lists:list[list], sample_size:int) -> tuple[np.ndarray,np.ndarray]:
    values_at_index = []  # To store all the values at each index for standard deviation calculation

    # Iterate over each list
    for lst in lists:
        # Ensure values_at_index is large enough to hold all values
        while len(values_at_index) < len(lst):
            values_at_index.append([])

        # Update the running_sum and values_at_index for each index in the current list
        for i, value in enumerate(lst):
            values_at_index[i].append(value)

    # Calculate the average and standard deviation for each index
    averages = np.array([sum(values) / len(values) for values in values_at_index if len(values)>0.75*sample_size])
    std_devs = np.array([np.std(values) for values in values_at_index if len(values)>0.75*sample_size])

    return averages, std_devs

def plot_results(setting_data:list, mode:str, show_std:bool=True):
    
    plt.figure(figsize=(10, 7.5))
    
    # Define a list of colors to cycle through
    colors = ['blue', 'green', 'red']
    for i, (
        setting_name, 
        setting_avg_corr_preds, 
        setting_std_corr_preds, 
        setting_avg_pixel_perc, 
        setting_std_pixel_perc
    ) in enumerate(setting_data):
        color = colors[i % len(colors)]  # Cycle through colors

        if mode in ['sdc', 'ssc']:
            x_data = 1 - (0.01 * setting_avg_pixel_perc)
            x_label = 'percentage of deleted pixels'
        elif mode in ['sdc_single', 'ssc_single']:
            x_data = np.arange(len(setting_avg_pixel_perc))
            x_label = 'concepts in descending order of importance'
            
        # Plot the main line for average correct predictions
        plt.plot(x_data, setting_avg_corr_preds, label=setting_name, color=color, marker='o')

        if mode in ['sdc', 'ssc'] and show_std:
            # Add a shade for the standard deviation of accuracy
            plt.fill_between(x_data, 
                            np.clip(np.subtract(setting_avg_corr_preds, setting_std_corr_preds), 0, 1), 
                            np.clip(np.add(setting_avg_corr_preds, setting_std_corr_preds), 0, 1), 
                            color=color, alpha=0.1)

            # Use errorbar to add horizontal lines with caps for the standard deviation of pixel deletion percentage
            for x, y, std in zip(x_data, setting_avg_corr_preds, setting_std_pixel_perc * 0.01):
                plt.errorbar(x, y, xerr=std, fmt='o', color='black', ecolor='black', capsize=5, elinewidth=2, markeredgewidth=2)

    plt.xlabel(x_label)
    plt.ylabel('average prediction accuracy')
    plt.title('Effect of Concept Flipping on Prediction Accuracy')
    plt.legend()

    # Set limits with a small padding
    plt.ylim(-0.05, 1.05)
    if mode in ['sdc', 'ssc']:
        plt.xlim(-0.05, 1.05)
    
    plt.grid(True)
    if mode == 'ssc':
        plt.gca().invert_xaxis() 

    # Save the figure to a file
    plt.savefig(f'{mode}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
def get_setting_params(setting_name:str):
    if setting_name == 'ace':
        return {
            'name': 'ACE',
            'layer_name': 'global_pool', 
            'segm_algo': 'slic',
            'n_imgs_per_cls': 50,
            'cropping_mode': 1,
            'use_masks': False,
            'masking_mode': None,
            'erosion_threshold': 0.25,
            'cluster_algo': 'kmeans',
            'norm_clustering':False,
            'num_clusters': 25,
            'min_cluster_size': 50,
            'min_coverage_ratio': 0.5,
            'max_samples_per_cluster': 50
        }
    elif setting_name == 'humcd':
        return {
            'name': 'HU-MCD',
            'layer_name': 'global_pool',
            'segm_algo': 'sam',
            'n_imgs_per_cls': 400,
            'cropping_mode': 0,
            'use_masks': True,
            'masking_mode': -1,
            'erosion_threshold': 1.0,
            'norm_clustering':True,
            'cluster_algo': 'sparse_subspace_clustering',
            'num_clusters': None,
            'min_cluster_size': 50,
            'min_coverage_ratio': 0.0,
            'max_samples_per_cluster': None
        }
    elif setting_name == 'mcd':
        return {
            'name': 'MCD',
            'layer_name': 'layer4',
            'segm_algo': None,
            'n_imgs_per_cls': 400,  # they took 384 imgs/class
            'cropping_mode': 0,
            'use_masks': False,
            'masking_mode': None,
            'erosion_threshold': 1.0,
            'norm_clustering':True, 
            'cluster_algo': 'sparse_subspace_clustering',
            'num_clusters': None,
            'min_cluster_size': 0,
            'min_coverage_ratio': 0.0,
            'max_samples_per_cluster': None,
            'completeness_threshold': 0.5
        }
    elif setting_name == 'rdm':
        return {
            'name': 'Random',
            'layer_name': 'layer4',
            'segm_algo': None,
            'n_imgs_per_cls': 0,
        }
    else:
        raise ValueError("Setting not recognized")

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    setting_data = []
    
    for setting_name in args.settings_to_compare:
        setting_params = get_setting_params(setting_name)
        print(f"[INFO] Running setting: {setting_params['name']}")
        
        setting_corr_preds = []
        setting_pixel_perc = []
                
        for cls_name in args.cls_list:
            print(f"[INFO] Computing {args.mode} for class {cls_name}")
            filepath = os.path.join(
                args.save_dir,
                utils_general.create_filename(
                    args.mode, cls_name, args.n_val_imgs_per_cls, args.model_name, 
                    setting_params['layer_name'], setting_name, extension='.pkl'
                )
            )
            if os.path.exists(filepath):
                with open(filepath, 'rb') as file:
                    cls_corr_pred, cls_pixel_perc = pickle.load(file)
                setting_corr_preds.extend(cls_corr_pred)
                setting_pixel_perc.extend(cls_pixel_perc)
                print(f"[INFO] Loaded cached cached {args.mode} values for class {cls_name} from file: {filepath}")
                continue
            expl_obj = ConceptExplainer(
                source_dir=args.source_dir,
                target_class=cls_name,
                model_name=args.model_name,
                layer_name=setting_params['layer_name']
            )
            expl_obj.load_class_images(
                num_imgs=setting_params['n_imgs_per_cls'],
                max_shortest_side=300
            )
            if setting_name in ['humcd', 'ace']:
                # apply segmentation algorithm on images
                expl_obj.segment_class_images(setting_params['segm_algo'])
            elif setting_name == 'mcd':
                for img in expl_obj.class_imgs:
                    # little trick to get the acts from the images
                    img.segments = [classes.SegmentClass(
                        mask=np.ones(img.img_numpy.shape[:2]),
                        original_image=img
                    )]
                expl_obj.segm_algo = 'full_imgs'
                
            if setting_name != 'rdm':
                expl_obj.compute_cls_segms_acts(
                    cache_dir=None,
                    cropping_mode=setting_params['cropping_mode'],
                    use_masks=setting_params['use_masks'],
                    masking_mode=setting_params['masking_mode'],
                    erosion_threshold=setting_params['erosion_threshold'],
                    batch_size=args.batch_size,
                    norm_acts=False
                )
            else:
                expl_obj.batch_size = args.batch_size
                
            if setting_name == 'mcd':  
                prepare_imgs_for_mcd(expl_obj.class_imgs)
                for n_clusters in range(3, 20): # 3 was the minimal number of clusters
                    expl_obj.create_concepts(
                        cluster_algo=setting_params['cluster_algo'],
                        n_clusters=n_clusters,
                        min_size=setting_params['min_cluster_size'],
                        min_coverage=setting_params['min_coverage_ratio'],
                        max_samples=setting_params['max_samples_per_cluster'],
                        outlier_percentile=0.75,
                        folderpath='self_repr_matrices_mcd'
                    )
                    expl_obj.compute_concept_subspace_bases(
                        subspace_dimensionality=None,
                        est_mode='FO',
                        compt_princ_angl=False
                    )
                    class_idx = utils_general.get_imagenet_class_index(class_name=cls_name)
                    weight_vector = expl_obj.model.fc.weight.data.detach().clone()[class_idx].cpu().numpy()
                    completeness = utils_mcd.calc_completeness(weight_vector, expl_obj.concept_bases)
                    print(f"[INFO] Class {cls_name} with {n_clusters} clusters, Completeness: {completeness:.2f}")
                    if completeness > setting_params['completeness_threshold']:
                        break
            elif setting_name != 'rdm':            
                expl_obj.create_concepts(
                    cluster_algo=setting_params['cluster_algo'],
                    n_clusters=setting_params['num_clusters'], 
                    norm_acts=setting_params['norm_clustering'],
                    min_size=setting_params['min_cluster_size'], 
                    min_coverage=setting_params['min_coverage_ratio'],
                    max_samples=setting_params['max_samples_per_cluster'],
                    outlier_percentile=1.0,
                    folderpath='self_repr_matrices_humcd'
                )
                
            if setting_name == 'humcd':
                expl_obj.compute_concept_subspace_bases(
                    subspace_dimensionality=None,
                    est_mode='ratio',
                    compt_princ_angl=False
                )
            elif setting_name == 'ace':
                expl_obj.compute_tcav_scores(
                    rdm_imgs_folder = 'random',
                    mimic_segm = False,
                    segm_cache_dir = None,
                    num_rdm_runs = 20,
                    acts_cache_dir = 'acts',
                    training_mode = 'sklearn',
                    num_epochs = None,
                    num_test_images = 50
                )
        
            cls_corr_pred, cls_pixel_perc = compute_sdc_scores(
                expl_obj, args.n_val_imgs_per_cls, 'imagenet_val_imgs', setting_name, args.mode
            )
            with open(filepath, 'wb') as f:
                pickle.dump((cls_corr_pred, cls_pixel_perc), f)
            print(f"[INFO] Cached cached {args.mode} values for class {cls_name} in file: {filepath}")
            
            setting_corr_preds.extend(cls_corr_pred)
            setting_pixel_perc.extend(cls_pixel_perc)

        avg_corr_pred, std_corr_pred = calc_avg_and_std(setting_corr_preds, len(args.cls_list)*args.n_val_imgs_per_cls)
        avg_pixel_per, std_pixel_per = calc_avg_and_std(setting_pixel_perc, len(args.cls_list)*args.n_val_imgs_per_cls)
        
        setting_data.append(
            (setting_params['name'], avg_corr_pred, std_corr_pred, avg_pixel_per, std_pixel_per)
        )
    
    print("[INFO] Plotting data")
    plot_results(setting_data, args.mode)
        
    return 0
    
def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()
    
    # save/load images, cached values and results and define target class
    parser.add_argument(
        '--mode', 
        type=str,
        help='One of Smallest Destroying Concepts (sdc), Single Destroying Concepts (sdc_single) Smallest Sufficent Concept (ssc) or Single Sufficent Concept (ssc_single)',
        choices=['sdc', 'sdc_single', 'ssc', 'ssc_single'],
        default='sdc'
    )
    parser.add_argument(
        '--settings_to_compare', 
        nargs='+', 
        help='setting names seperated by a space which should be compared',
        choices=['ace', 'mcd', 'humcd', 'rdm'],
        default=['ace', 'mcd', 'humcd', 'rdm']
    )
    parser.add_argument(
        '--cls_list', 
        nargs='+', 
        help='list of class names',
        default=['airliner', 'beach_wagon', 'zebra', 'container_ship', 'police_van', 'hummingbird', 'Siamese_cat', 'ox', 'golden_retriever', 'tailed_frog']
    )
    parser.add_argument(
        '--source_dir', 
        type=str,
        help='directory where the class- and random image datasets are saved', 
        default='sourceDir'
    )
    parser.add_argument(
        '--save_dir', 
        type=str,
        help='directory where the results are saved', 
        default='./sdc_data'
    )
    parser.add_argument(
        '--model_name', 
        type=str,
        help='name of pytorch model to be explained', 
        default='resnet50'
    )
    parser.add_argument(
        '--batch_size', 
        type=int,
        help="batch size used to calculate the activations from the PyTorch model",
        default=64
    )
    parser.add_argument(
        '--n_val_imgs_per_cls', 
        type=int, 
        help='',
        default=50
    )

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))