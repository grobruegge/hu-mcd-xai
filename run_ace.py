import os, sys, argparse
import classes
from concept_explainer import ConceptExplainer
import numpy as np    
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import skimage.segmentation as segmentation
from tqdm import tqdm
from skimage.transform import resize
from run_humcd import check_folder

def save_concepts(sorted_concepts:list[classes.ConceptClass]=None, 
                  n_concepts:int=10, n_imgs_per_concepts:int=10, mode:str='diverse', 
                  folderpath:str=None):
        
    fig = plt.figure(figsize=(n_imgs_per_concepts * 2, 4 * n_concepts))
    outer = gridspec.GridSpec(n_concepts, 1, wspace=0., hspace=0.3)
    
    for idx_concept, concept in enumerate(tqdm(sorted_concepts[:n_concepts], desc='Plotting concepts')):
        inner = gridspec.GridSpecFromSubplotSpec(
        2, n_imgs_per_concepts, subplot_spec=outer[idx_concept], wspace=0, hspace=0.1
        )
        concept_segms = concept.get_segments(mode=mode, num=n_imgs_per_concepts)
        concept_folderpath = os.path.join(folderpath, str(concept.label))
        # Ensure concept directory exists
        if not os.path.exists(concept_folderpath):
            os.makedirs(concept_folderpath)
        for segm_idx, segm in enumerate(concept_segms):
            ax = plt.Subplot(fig, inner[segm_idx])
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
            if segm_idx == int(n_imgs_per_concepts / 2):
                ax.set_title(f"Cluster {concept.label} (size: {len(concept.segments)}) (TCAV score: {np.mean(concept.tcav_scores):.2f})")
            ax.grid(False)
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
            ax = plt.Subplot(fig, inner[segm_idx + n_imgs_per_concepts])
            mask = segm.mask
            image = segm.org_img.img_numpy
            ax.imshow(
                resize(
                    segmentation.mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'),
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

def main(args):
    # iterate over all classes provided in class list
    for cls_name in args.cls_list:
        # folderpath to save the concepts in
        cls_folderpath = check_folder(os.path.join(args.save_dir, 'ace', cls_name))
        # in case concepts already exists, skip to next class
        if cls_folderpath == None:
            continue
        # create explainer object 
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
        # segment images using SLIC
        explainer.segment_class_images(
            segm_algo='slic',
            cache_dir='segms'
        )
        # calculate activations for image segmentations
        explainer.compute_cls_segms_acts(
            cache_dir='acts',
            cropping_mode=1, # fitting rectangle, rescaled
            use_masks=False, # use mean padding
            masking_mode=None, # not relevant, because use_mask=False
            erosion_threshold=1.0, 
            batch_size=args.batch_size,
            norm_acts=False
        )
        # create concepts by clustering segmentation activations
        explainer.create_concepts(
            cluster_algo='kmeans',
            n_clusters=25, 
            norm_acts=False,
            min_size=50, # only consider clusters with more than 50 segments as concepts 
            min_coverage=0.5, # only consider clusters with segments belonging to more than 50 % of the class images as concepts
            max_samples=50 # only use 50 segments whose activations are closest to the centroid
        )
        # compute TCAV scores
        print("[INFO] Calculating TCAV scores")
        explainer.compute_tcav_scores(
            rdm_imgs_folder=args.rdm_img_folder,
            mimic_segm=False, # random concepts are represented by complete random images
            segm_cache_dir='rdm_segms',
            num_rdm_runs=20, # how many times CAV are trained against different random concepts
            acts_cache_dir='acts',
            training_mode='sklearn', # train SVM
            num_epochs=None, # only relevant if training_mode='pytorch'
            num_test_images=50 # number of class images to calculate TCAV scores
        )
        # get TCAV scores
        mean_tcav_scores, std_tcav_scores, p_values = explainer.get_tcav_scores()

        sorted_concepts:list[classes.ConceptClass] = []
        # iterate over the concept in descending order of relevance
        for concept_idx in np.argsort(mean_tcav_scores)[::-1]:
            # check whether the concept pass statistical testing
            if p_values[concept_idx] < 0.01:
                sorted_concepts.append(explainer.concepts[concept_idx])

        # save overview plot and examples for the top concepts that passed statistical testing
        save_concepts(
            sorted_concepts=sorted_concepts,
            n_concepts=args.n_concepts_to_plot,
            n_imgs_per_concepts=args.n_imgs_per_concept,
            mode='diverse',
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
        default='resnet18'
    )
    parser.add_argument(
        '--layer_name', 
        type=str,
        help='layer from which to calculate the activations from', 
        default='global_pool'
    )
    parser.add_argument(
        '--batch_size', 
        type=int,
        help="batch size used to calculate the activations from the PyTorch model",
        default=64
    )
    parser.add_argument(
        '--n_cls_imgs', 
        type=int,
        help="number of class images to be segmented",
        default=50
    )
    parser.add_argument(
        '--rdm_img_folder', 
        type=str,
        help='folder which contains random images within source_dir (recommoned at least 1000)', 
        default='random'
    )
    parser.add_argument(
        '--save_dir', 
        type=str,
        help='directory where the results are saved', 
        default='concept_examples'
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