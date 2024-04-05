import os, sys, argparse
import classes
from concept_explainer import ConceptExplainer
import numpy as np
from sklearn.preprocessing import normalize
from skimage.transform import resize
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from run_humcd import check_folder
from utils import utils_general, utils_mcd
                
def prepare_imgs_for_mcd(images:list[classes.ImageClass]):
    # iterate over all images
    for img in images:
        # each image should only contain one segment representing the unmasked image
        if len(img.segments) > 1:
            raise ValueError("[ERROR] Should only be one segment")
        img_act = img.segments[0].model_act
        n_channels = img_act.shape[0]
        w,h, = img_act.shape[1:]
        if len(img_act.shape) != 3:
            raise ValueError("[ERROR] Expected feature map of following shape (n_channels, height, width)")
        pixel_acts = img_act.transpose(1,2,0).reshape(-1, n_channels)
        # normalize the spatial feature vectors to have unit lenght
        pixel_acts = normalize(pixel_acts, norm='l2', axis=1)
        # rearrange the image segments to represent the spatial activations
        img.segments = []
        for p_idx, p_act in enumerate(pixel_acts):
            # check whether a spatial activation contains all-zero values
            if np.all(p_act == 0):
                print(f"[WANRNING] Activation with all zeros encountered") 
                # skip that activation (leads to single-element clusters)
                continue
            # create a position mask that showing the position of the spatial activation in the feature map
            position_mask = np.zeros((w, h))
            position_mask[p_idx // w, p_idx % h] = 1
            pixel_segm = classes.SegmentClass(
                mask=position_mask, 
                original_image=img
            )
            pixel_segm.model_act = p_act
            img.segments.append(pixel_segm)

def save_concepts(sorted_prototype_idxs, images:list[classes.ImageClass], 
                  concept_act_maps:np.ndarray, concept_importance:np.ndarray,
                  n_concepts:int=10, n_imgs_per_concepts:int=10, 
                  folderpath:str=None):
        
    fig = plt.figure(figsize=(n_imgs_per_concepts * 2, 4 * n_concepts))
    outer = gridspec.GridSpec(n_concepts, 1, wspace=0., hspace=0.3)
    # order the concept based on their importance (exclude orthogonal component)
    concept_order = concept_importance[:-1].argsort()[::-1]
    # iterate over the concept in decreasing order of global importance (last on is the orthogonal complement)
    print("[INFO] Saving concepts...")
    for row, idx_concept in enumerate(list(concept_order[:n_concepts-1]) + [len(concept_order)]):
        inner = gridspec.GridSpecFromSubplotSpec(
        2, n_imgs_per_concepts, subplot_spec=outer[row], wspace=0, hspace=0.1
        )
        # concept_segms = concept.get_segments(mode=mode, num=n_imgs_per_concepts)
        concept_folderpath = os.path.join(folderpath, str(idx_concept))
        # Ensure concept directory exists
        if not os.path.exists(concept_folderpath) and (idx_concept != len(concept_order)):
            os.makedirs(concept_folderpath)
        for idx_sample in range(n_imgs_per_concepts):
            ax = plt.Subplot(fig, inner[idx_sample])
            # get the image index with the maximum concept similarity
            idx_img = sorted_prototype_idxs[idx_sample, idx_concept]
            # resize the image to match the model input size
            img = resize(images[idx_img].img_numpy, (224,224))
            # get the concept activation map
            cam = resize(concept_act_maps[idx_img, idx_concept], (224,224)) 
            mask = cam > 0.4
            ones = np.where(mask == 1)
            if ones[0].size > 0 and ones[1].size > 0:
                ymin, ymax, xmin, xmax = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
            else:
                ymin, ymax, xmin, xmax = 0, cam.shape[0]-1, 0, cam.shape[1]-1
            ax.imshow(resize(
                classes.SegmentClass.get_padded_img(
                    img[ymin:ymax, xmin:xmax], 
                    mask[ymin:ymax, xmin:xmax],
                    fill_value=0.4588
                ), 
                (224,224)
            ))
            ax.set_xticks([])
            ax.set_yticks([])
            if idx_sample == int(n_imgs_per_concepts / 2):
                ax.set_title(f"Concept {idx_concept}")
            ax.grid(False)
            # exclude orthogonal complement
            if idx_concept != len(concept_order):
                # Save concept example
                np.save(
                    os.path.join(concept_folderpath, f"{os.path.basename(images[idx_img].filename)}_resized.npy"), 
                    resize(img, (224, 224))
                )
                np.save(
                    os.path.join(concept_folderpath, f"{os.path.basename(images[idx_img].filename)}_mask_resized.npy"), 
                    resize(mask, (224, 224))
                )
            fig.add_subplot(ax)
            ax = plt.Subplot(fig, inner[idx_sample + n_imgs_per_concepts])
            ax.imshow(img)
            ax.contour(cam, levels=[0.5,], colors='yellow', linewidths=1)
            ax.contour(cam, levels=[0.4,], colors='white', linewidths=1)
            ax.imshow(np.ones_like(cam)*0.2,  cmap="binary", vmin=0, vmax=1, alpha=np.clip(1-cam,0,1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(images[idx_img].filename[-10:])
            ax.grid(False)
            fig.add_subplot(ax)
            
    plt.savefig(os.path.join(folderpath, 'overview.png'))
    plt.close(fig) 
             
def main(args):
    for cls_name in args.cls_list:
        print(f"[INFO] Running MCD for class {cls_name}")
        # folderpath to save the concepts in
        cls_folderpath = check_folder(os.path.join(args.save_dir, 'mcd', cls_name))
        # in case concepts already exists, skip to next class
        if cls_folderpath == None:
            continue
        # initialize ConceptExplainer object
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
        # instead of segmenting the images, get the feature map for the complete images
        for img in explainer.class_imgs:
            # define the segment as unmasked 
            img.segments = [classes.SegmentClass(
                mask=np.ones(img.img_numpy.shape[:2]),
                original_image=img
            )]
        # no segmentation is applied
        explainer.segm_algo = 'full_imgs'
        # compute the feature maps from the unmasked images 
        explainer.compute_cls_segms_acts(
            cache_dir='acts',
            cropping_mode=0, # use the complete images (no cropping)
            use_masks=False, # no masking is applied
            masking_mode=None, # no masking is applied
            erosion_threshold=1.0, # no erosion is applied 
            batch_size=args.batch_size,
            norm_acts=False # normalization is applied later
        )
        # functions are not designed for MCD framework, some rearranging is necessary
        prepare_imgs_for_mcd(explainer.class_imgs)
        # find the minimal number of clusters to reach a completeness of 50%
        # 3 clusters were the minimum in MCD paper, we choose 5 for test consistency
        for n_clusters in range(5, 20): 
            explainer.create_concepts(
                cluster_algo='sparse_subspace_clustering',
                n_clusters=n_clusters,
                min_size=0, # no postprocessing of clusters
                min_coverage=0.0, # no postprocessing of clusters
                max_samples=None, # no postprocessing of clusters
                outlier_percentile=0.75,
                folderpath=args.cache_dir_self_repr_matrices
            )
            # compute subspace bases for each concept
            explainer.compute_concept_subspace_bases(
                subspace_dimensionality=None, # use heuristic to determine subspace dimensionality
                est_mode='FO',
                compt_princ_angl=False
            )
            # get the ImageNet1k class index based on the class name
            class_idx = utils_general.get_imagenet_class_index(cls_name)
            # get the class weight vector of the final linear classification layer
            weight_vector = explainer.model.fc.weight.data.detach()[class_idx].cpu().numpy()
            # calculate the completeness
            completeness = utils_mcd.calc_completeness(weight_vector, explainer.concept_bases)
            print(f"[INFO] Completeness score with {n_clusters} clusters: {completeness:.2f}")
            # check whether completeness threshold is reached
            if completeness > 0.5:
                break
        # compute global importance of concepts
        global_importance, _ = explainer.concept_quantification(
            weight_vector
        )
        # load test class images to get concept prototypes
        val_cls_imgs = explainer._load_images(
            folderpath=os.path.join(
                args.source_dir, 
                args.val_dir,
                f'{cls_name}_val'
            ),
            num_imgs=50, # use 50 validation images (see MCD implementation)
            max_shortest_side=300,
            shuffle=True
        )
        # same trick as used above to get the feature maps of the images
        for img in val_cls_imgs:
            img.segments = [classes.SegmentClass(
                mask=np.ones(img.img_numpy.shape[:2]),
                original_image=img
            )]
            
        explainer._load_or_calc_acts(
            cache_dir=None,
            identifier='val_cls_imgs',
            images=val_cls_imgs,
            cropping_mode=0, # use the complete images (no cropping)
            use_masks=False, # no masking is applied
            masking_mode=None, # no masking is applied
            erosion_threshold=1.0, # no erosion is applied 
            batch_size=args.batch_size,
            norm_acts=False,
            delete_zeros=False
        )
        n_concepts = len(explainer.concept_bases)
        feature_maps = []
        for test_img in val_cls_imgs:
            feature_maps.append(test_img.segments[0].model_act)
        feature_maps = np.stack(feature_maps, axis=0)
        n_imgs = feature_maps.shape[0]
        if n_imgs != len(val_cls_imgs):
            raise ValueError("[ERROR] Something went wrong.")
        n_channels = feature_maps.shape[1]
        w,h = feature_maps.shape[2:]
        acts_flat = feature_maps.transpose(0, 2, 3, 1).reshape(n_imgs*w*h, n_channels)
        # compute the concept similarities for each spatial activation
        concept_activations = explainer.concept_activations(
            acts=acts_flat,
            batch_sizes=w*h,
            norm_batch=True,
        )
        concept_activation_maps = concept_activations.reshape(
            n_imgs,w,h,n_concepts+1
        ).transpose(0,3,1,2)
        # take the maximum over the spatial activations 
        max_concept_activations = concept_activation_maps.max(axis=(2,3))
        # for each concept (axis=1), sort the images by their maximum spatial activations
        prototypes_idxs = (-1.0*max_concept_activations).argsort(axis=0)
        
        save_concepts(
            prototypes_idxs, 
            val_cls_imgs, 
            concept_activation_maps, 
            concept_importance=global_importance,
            n_concepts=args.n_concepts_to_plot,
            n_imgs_per_concepts=args.n_imgs_per_concept,
            folderpath=cls_folderpath
        )
        
def parse_arguments(argv):
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
        '--layer_name', 
        type=str,
        help='layer from which to calculate the activations from', 
        default='layer4'
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
        help='directory where the sparse representation matrices are cached', 
        default='self_repr_matrices_mcd'
    )
    parser.add_argument(
        '--val_dir', 
        type=str,
        help='directory where the test class images are saved (in folders named after the class)', 
        default='val_imgs'
    )
    parser.add_argument(
        '--save_dir', 
        type=str,
        help='directory where the results are saved', 
        default='./concept_examples'
    )
    parser.add_argument(
        '--n_concepts_to_plot', 
        type=int,
        help="number of (top) concepts to be saved",
        default=10
    )
    parser.add_argument(
        '--n_imgs_per_concept', 
        type=int,
        help="number of (top) images to be saved for each concepts",
        default=10
    )
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))