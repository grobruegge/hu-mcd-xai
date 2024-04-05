import os, sys, argparse
import classes
from concept_explainer import ConceptExplainer
import shutil
import numpy as np
import skimage.segmentation as segmentation
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from skimage.transform import resize
from utils import utils_general, utils_mcd

def check_folder(folderpath:str) -> str:
    if os.path.exists(folderpath):
        print(f"[WARNING] folder {folderpath} already exists!")
        user_input = input(
            "[QUESTION] Overwrite existing concepts for this class? (otherwise class is skipped) [y(es)|n(o)]: "
        ).lower().strip()
        if user_input in ['yes', 'y']:
            shutil.rmtree(folderpath)
            print(f"[INFO] Deleted existing images in folder: {folderpath}")
        else: 
            print("[INFO] Continue with next class")
            return None
    os.makedirs(folderpath, exist_ok=False)
    return folderpath

def get_top_concept_segms(concept_activations:np.ndarray, n_segms_per_img:list, 
                          n_prototypes:int):
    """for each concept, get the segments in which the concepts is maximally activated (with
    maximal similarities to the concept)

    Args:
        concept_activations (np.ndarray): _description_
        batch_sizes (list): list containing the number of segments for each image
        n_prototypes (int): number of prototypes to get for each concept

    Returns:
        list[list]: list containing lists that store the maximally activating segments for each concept
    """
    # list to store the top segments for each concept
    top_concept_segms = []
    concept_assignments = np.argmax(concept_activations, axis=1)
    # iterate over all concepts (including the orthogonal complement)
    for concept_idx in range(concept_activations.shape[1]):
        concept_segms_idxs = np.where(concept_assignments == concept_idx)[0]
        # Sort these indices based on activation values in descending order
        segm_idxs_sorted = concept_segms_idxs[np.argsort(concept_activations[concept_segms_idxs, concept_idx])[::-1]]
        # list to store the top segments for each concept
        top_segms = []
        # set that stores the images that were already considered
        covered_img_idxs = set()
        # iterate over the image indices in decreasing order of maximal concept activations
        for segm_idx in segm_idxs_sorted:
            # compute the start indices for each image by accumulating the #segms/img
            cumsum_batches = np.cumsum(n_segms_per_img)
            # find the image index the current segment
            img_idx = np.searchsorted(cumsum_batches, segm_idx, side='right')
            # check whether the image was already considered
            if img_idx not in covered_img_idxs:
                # if that is not the case, add the current image to the set
                covered_img_idxs.add(img_idx)
                # compute the relative segment index (within the current image)
                rel_segm_idx = segm_idx - (cumsum_batches[img_idx - 1] if img_idx > 0 else 0)
                # add the tuple consisting of image and relative segment index to the top segments
                top_segms.append((img_idx, rel_segm_idx))
            # stop when n_examples are reached
            if len(top_segms) >= n_prototypes:
                break
        top_concept_segms.append(top_segms)
    return top_concept_segms

def save_concepts(top_concept_segms:list[tuple], imgs:list[classes.ImageClass], 
                  concept_scores:np.ndarray, n_concepts:int, n_imgs_per_concepts:int,
                  folderpath:str):

    # create a figure to plot the top prototypes for each concept
    fig = plt.figure(figsize=(n_imgs_per_concepts * 2, 4 * n_concepts))
    outer = gridspec.GridSpec(n_concepts, 1, wspace=0., hspace=0.3)
    # order the concept based on their importance (exclude orthogonal component)
    concept_order = concept_scores[:-1].argsort()[::-1]
    # iterate over the concept in decreasing order of global importance (last on is the orthogonal complement)
    print("[INFO] Saving concepts...")
    for row, idx_concept in enumerate(list(concept_order[:n_concepts-1])+[len(concept_order)]):
        top_segms = top_concept_segms[idx_concept]
        inner = gridspec.GridSpecFromSubplotSpec(
        2, n_imgs_per_concepts, subplot_spec=outer[row], wspace=0, hspace=0.1
        )
        concept_folderpath = os.path.join(folderpath, str(idx_concept))
        # Ensure concept directory exists
        if not os.path.exists(concept_folderpath) and (idx_concept != len(concept_order)):
            os.makedirs(concept_folderpath)
        for column, (img_idx, segm_idx) in enumerate(top_segms):
            ax = plt.Subplot(fig, inner[column])
            segm = imgs[img_idx].segments[segm_idx]
            ax.imshow(resize(
                segm.get_padded_img(
                    segm.get_cropped_segment(), 
                    segm.get_cropped_segment_mask(),
                    fill_value=0.4588
                ), 
                (224,224)
            ))
            ax.set_xticks([])
            ax.set_yticks([])
            if column == int(n_imgs_per_concepts / 2):
                ax.set_title(f"Concept: {idx_concept} (Score: {concept_scores[idx_concept]:.2f})")
            ax.grid(False)
            # exclude orthogonal complement
            if idx_concept != len(concept_order):
                # Save concept example
                np.save(
                    os.path.join(concept_folderpath, f"{os.path.basename(segm.org_img.filename)}_resized.npy"), 
                    resize(segm.org_img.img_numpy, (224, 224))
                )
                np.save(
                    os.path.join(concept_folderpath, f"{os.path.basename(segm.org_img.filename)}_mask_resized.npy"), 
                    resize(segm.mask, (224, 224))
                )
            fig.add_subplot(ax)
            ax = plt.Subplot(fig, inner[column + n_imgs_per_concepts])
            ax.imshow(
                resize(
                    segmentation.mark_boundaries(
                        segm.org_img.img_numpy, 
                        segm.mask, 
                        color=(1, 1, 0), 
                        mode='thick'
                    ),
                    (224,224)
                )
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(segm.org_img.filename[-10:])
            ax.grid(False)
            fig.add_subplot(ax)
            
    plt.savefig(os.path.join(folderpath, 'overview.png'))
    plt.close(fig) 
    
def main(args:argparse.Namespace):
    for cls_name in args.cls_list:
        print(f"[INFO] Running framework for class {cls_name}")
        # folderpath to save the concepts in
        cls_folderpath = check_folder(os.path.join(args.save_dir_results, 'humcd', cls_name))
        # in case concepts already exists, skip to next class
        if cls_folderpath == None:
            continue
        # create ConceptExplainer object
        explainer = ConceptExplainer(
            source_dir=args.source_dir,
            target_class=cls_name,
            model_name=args.model_name,
            layer_name=args.layer_name,
        )
        # load class images
        explainer.load_class_images(
            num_imgs=args.n_cls_imgs,
            max_shortest_side=300
        )
        # segment images using SAM
        explainer.segment_class_images(
            segm_algo='sam',
            cache_dir='segms'
        )
        # load activations for image segments
        explainer.compute_cls_segms_acts(
            cache_dir='acts',
            cropping_mode=0, # use the complete image
            use_masks=True, # mask out the respective segments
            masking_mode=-1, # reveal the direct sourrounding of the segment
            erosion_threshold=0.25, # apply erosion on large masks to prevent class leakage
            batch_size=args.batch_size,
            norm_acts=False, # normalize activations
        )
        # create concepts by clustering segmentation activations
        explainer.create_concepts(
            cluster_algo='sparse_subspace_clustering', # use sparse Subspace Clustering (SC)
            n_clusters=None, # determined automatically based on the average #segms/img
            norm_acts=True, # use the normalized activations for clustering
            min_size=50, # only consider clusters with at least 50 members as concepts 
            min_coverage=0.0, # not used
            max_samples=None, # filter outlier by only using the top 200 examples per cluster,
            outlier_percentile=1.0, # percentage of outliers filterd for sparse SC
            folderpath=args.cache_dir_self_repr_matrices,
        )
        # compute subspace bases for each concept
        explainer.compute_concept_subspace_bases(
            subspace_dimensionality=None, # use heuristic to determine subspace dimensionality
            est_mode='ratio',
            compt_princ_angl=False
        )
        # get the ImageNet1k class index based on the class name
        imagenet_cls_idx = utils_general.get_imagenet_class_index(cls_name)
         # get the class weight vector of the final linear classification layer
        weight_vector = explainer.model.fc.weight.data.detach()[imagenet_cls_idx].cpu().numpy()
        # calculate concept scores as described in MCD paper
        concept_scores, _ = explainer.concept_quantification(
            weight_vector
        )
        # calculate completeness as described in MCD paper
        completeness = 1 - concept_scores[-1]
        # calculate completeness following MCD implementation
        completeness_mcd = utils_mcd.calc_completeness(weight_vector, explainer.concept_bases)
        print(f"[INFO] Completeness: {completeness:.2f} (MCD: {completeness_mcd:.2f})")
        # load validation class images to get concept prototypes
        val_cls_imgs = explainer._load_images(
            folderpath=os.path.join(
                args.source_dir, 
                args.val_dir,
                f'{cls_name}_val'
            ),
            num_imgs=50, # use 50 validation images 
            max_shortest_side=300,
            shuffle=False
        )
        # segment validation images
        explainer._segment_images(
            images=val_cls_imgs,
            segm_algo='sam',
            cache_dir='segms_val'
        )
        # calculate activations for image segments
        explainer._load_or_calc_acts(
            cache_dir=None,
            identifier=cls_name,
            images=val_cls_imgs,
            cropping_mode=0, # use the complete image (no cropping)
            use_masks=True, # mask out the respective segments
            masking_mode=-1, # reveal the direct sourrounding of the segment
            erosion_threshold=0.25, # apply erosion on large masks to prevent class leakage
            batch_size=args.batch_size,
            norm_acts=False, # normalize activations
            delete_zeros=True
        )
        
        val_segms_acts = []
        n_segms_per_img = []
        for v_img in val_cls_imgs:
            n_segms_per_img.append(len(v_img.segments))
            for v_segm in v_img.segments:
                val_segms_acts.append(v_segm.model_act)
        val_segms_acts = np.stack(val_segms_acts, axis=0)
        # compute the concept similarities (activations) for each segment
        concept_activations = explainer.concept_activations(
            acts=val_segms_acts,
            batch_sizes=n_segms_per_img,
            norm_batch=False,
        )
        # get the maximally activating segments for each concept
        top_concepts_segms = get_top_concept_segms(
            concept_activations=concept_activations, 
            n_segms_per_img=n_segms_per_img, 
            n_prototypes=args.n_imgs_per_concept
        )
        save_concepts(
            top_concept_segms=top_concepts_segms,
            imgs=val_cls_imgs,
            concept_scores=concept_scores,
            n_concepts=args.n_concepts_to_plot,
            n_imgs_per_concepts=args.n_imgs_per_concept,
            folderpath=cls_folderpath          
        )
            
def parse_arguments(argv) -> argparse.Namespace:
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--cls_list', 
        nargs='+', 
        help='list of class names',
        default=['airliner', 'beach_wagon', 'zebra', 'container_ship', 'police_van', 'hummingbird', 'Siamese_cat', 'ox', 'golden_retriever', 'tailed_frog']
    )
    parser.add_argument(
        '--source_dir', 
        type=str,
        help='directory where the class images are saved (in folders named after the class)', 
        default='sourceDir'
    )
    parser.add_argument(
        '--model_name', 
        type=str,
        help='name of model to be explained (tested for ResNet18 and ResNet50)', 
        default='resnet50'
    )
    parser.add_argument(
        '--batch_size', 
        type=int,
        help="batch size used to calculate the activations from the PyTorch model",
        default=64
    )
    parser.add_argument(
        '--layer_name', 
        type=str,
        help='layer from which to calculate the activations from', 
        default='global_pool'
    )
    parser.add_argument(
        '--n_cls_imgs', 
        type=int,
        help="number of class images to be segmented",
        default=400
    )
    parser.add_argument(
        '--cache_dir_self_repr_matrices', 
        type=str,
        help='directory where the results are saved', 
        default='./self_repr_matrices_humcd'
    )
    parser.add_argument(
        '--val_dir', 
        type=str,
        help='directory where the test class images are saved (in folders "[class name]_val")', 
        default='./val_imgs'
    )
    parser.add_argument(
        '--save_dir_results', 
        type=str,
        help='directory where the results are saved', 
        default='./concept_examples'
    )
    parser.add_argument(
        '--n_concepts_to_plot', 
        type=int,
        help="number of concepts to be saved (in decreasing order of importance)",
        default=10
    )
    parser.add_argument(
        '--n_imgs_per_concept', 
        type=int,
        help="number of maximally activating examples (prototypes) to be saved for each concepts",
        default=10
    )
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))