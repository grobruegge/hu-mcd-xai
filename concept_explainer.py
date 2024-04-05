import numpy as np
import os
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import utils_general, utils_mcd
import classes
from scipy.linalg import norm
import random
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
from itertools import accumulate


class ConceptExplainer():
    def __init__(self, source_dir:str, target_class:str, model_name:str, layer_name:str):
        self.source_dir:str = source_dir
        self.target_class:str = target_class
        self.layer_name:str = layer_name
        self.model = utils_general.make_model(model_name)
        self.class_imgs:list[classes.ImageClass] = None
        self.max_shortest_side:int = None
        self.segm_algo:str = None
        self.cropping_mode:bool = None
        self.use_masks:bool = None 
        self.masking_mode:int = None
        self.erosion_threshold:float = None
        self.batch_size:int = None
        self.norm_acts:bool = None
        self.clustering:classes.ClusterSpaceClass = None
        self.concepts:list[classes.ConceptClass] = None
        self.max_samples_per_concept:int = None
        self.concept_bases:list[np.ndarray] = None
        self.compl_basis:np.ndarray = None
        
    @staticmethod
    def _load_images(folderpath:str, num_imgs:int, max_shortest_side:int,
                     shuffle:bool=False) -> list[classes.ImageClass]:
        imgs_list = []
        filenames_list = sorted(os.listdir(folderpath))
        if shuffle:
            random.shuffle(filenames_list)
        print(f"[INFO] Loading {os.path.basename(folderpath)} images...")
        for filename in filenames_list:
            try:
                img = classes.ImageClass(
                    filename=os.path.join(folderpath, filename),
                    max_shortest_side=max_shortest_side,
                )
            except Exception as e:
                print(f"[WARNING] Cannot load image {filename}: {e} (will be skipped)")
                continue
            imgs_list.append(img)
            if len(imgs_list) >= num_imgs:
                break
        print(f"[INFO] Loaded {len(imgs_list)} images of class {os.path.basename(folderpath)}")
        return imgs_list
    
    def load_class_images(self, num_imgs:int=100, max_shortest_side:int=300):
        self.class_imgs = self._load_images(
            folderpath=os.path.join(self.source_dir, self.target_class),
            num_imgs=num_imgs,
            max_shortest_side=max_shortest_side
        )
        self.max_shortest_side = max_shortest_side
    
    @staticmethod
    def _segment_images(images:list[classes.ImageClass], segm_algo:str, cache_dir:str):
        if segm_algo == 'sam':
            sam_model = utils_general.load_sam_mask_generator(
                points_per_side=32,
                min_mask_region_area=256 #very small regions max 16 x 16 pixels
            )
        else:
            sam_model = None
        for img in tqdm(images, desc="[INFO] Performing segmentations"):
            img.load_segments(cache_dir=cache_dir, sam_model=sam_model)

    def segment_class_images(self, segm_algo:str='sam', cache_dir:str='segms'):
        self._segment_images(self.class_imgs, segm_algo, cache_dir)
        self.segm_algo = segm_algo

    @staticmethod
    def _link_acts_to_segms(images:list[classes.ImageClass], segments_activations:np.ndarray, 
                        segments_logits:np.ndarray, norm_acts:bool, delete_zeros:bool=True):
        if norm_acts:
            segments_activations = normalize(segments_activations, norm='l2', axis=1)
        counter = 0
        for img in images:
            for segm in img.segments[:]:
                if delete_zeros and np.all(segments_activations[counter] == 0):
                    img.segments.remove(segm)
                else:
                    segm.model_act = segments_activations[counter]
                    segm.model_pred = segments_logits[counter]
                counter += 1
        if counter != len(segments_activations):
            raise ValueError("Not all activations could be linked to segments")
        
    def _load_or_calc_acts(self, cache_dir:str, identifier:str, images:list[classes.ImageClass],
                           cropping_mode:int, use_masks:bool, masking_mode:int, 
                           erosion_threshold:float=1.0, batch_size:int=64, 
                           norm_acts:bool=False, delete_zeros:bool=True)->tuple[np.ndarray,np.ndarray]:
        
        if cache_dir != None:
            filename = utils_general.create_filename(
                identifier,
                self.model.default_cfg['architecture'],
                self.layer_name,
                self.segm_algo,
                'org_scale' if cropping_mode == 0 else (
                    'tight_cropped_segms' if cropping_mode == 1 else 'quadr_cropped_segms'
                ),
                'masked_conv' if use_masks else 'padded',
                '' if not use_masks else (
                    'soft_border' if masking_mode > 0 else (
                        'hard_border' if masking_mode == 0 else 'leakage'
                    )
                ),
                f'erosion_{erosion_threshold}',
                utils_general.create_hashsum([img.filename for img in images])
            )
            fp_acts = os.path.join(cache_dir, filename + '_acts.npy')
            fp_logits = os.path.join(cache_dir, filename + '_logits.npy')
            
            if all([os.path.exists(fp) for fp in [fp_acts, fp_logits]]):
                segm_acts = np.load(fp_acts, allow_pickle=False)
                segm_logits = np.load(fp_logits, allow_pickle=False)
                self._link_acts_to_segms(images, segm_acts, segm_logits, norm_acts, delete_zeros)
                print(f"[INFO] Loaded activations from file: {fp_acts}")
                return
            
        # Prepare dataset
        dataset = classes.ConceptDatasetClass(
            images, 
            self.model.default_cfg, 
            cropping_mode, 
            use_masks,
            masking_mode,
            erosion_threshold
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=utils_general.custom_collate)

        segm_acts, segm_logits = utils_general.compute_activations(
            self.model,
            self.layer_name,
            dataloader
        )
        self._link_acts_to_segms(images, segm_acts, segm_logits, norm_acts, delete_zeros)
        
        if cache_dir != None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # Save computed activations
            np.save(fp_acts, segm_acts, allow_pickle=False)
            np.save(fp_logits, segm_logits, allow_pickle=False)
            print(f"[INFO] saved activations of segmentations in file: {fp_acts}")
                
    def compute_cls_segms_acts(self, cache_dir:str='acts', cropping_mode:int=0, 
                               use_masks:bool=True, masking_mode:int=-1, erosion_threshold:float=1.0,
                               batch_size:int=64, norm_acts:bool=False):
        if self.segm_algo is None:
            raise ValueError("first, compute image segmentation")
        
        self._load_or_calc_acts(
            cache_dir=cache_dir,
            identifier=self.target_class,
            images=self.class_imgs,
            cropping_mode=cropping_mode,
            use_masks=use_masks,
            masking_mode=masking_mode,
            erosion_threshold=erosion_threshold,
            batch_size=batch_size,
            norm_acts=norm_acts
        )
            
        # update class attributes
        self.cropping_mode = cropping_mode
        self.use_masks = use_masks
        self.masking_mode = masking_mode
        self.erosion_threshold = erosion_threshold
        self.batch_size = batch_size
        self.norm_acts = norm_acts
        
    def _check_cluster(self, cluster_segms:list[classes.SegmentClass], min_size:int, 
                       min_coverage:float) -> bool:
        
        cluster_size = len(cluster_segms)
        if cluster_size < min_size:
            return False
        
        cluster_imgs = set([segm.org_img.filename for segm in cluster_segms])

        if len(cluster_imgs) / len(self.class_imgs) < min_coverage:
            return False
                
        return True
    
    def create_concepts(self, cluster_algo:str='kmeans', n_clusters:int=None, norm_acts:bool=True,
                        min_size:int=0, min_coverage:float=0.0, max_samples:int=None, 
                        outlier_percentile:float=1.0, folderpath:str='self_repr_matrices'):
        self.clustering = classes.ClusterSpaceClass(
            images = self.class_imgs,
            norm_acts=norm_acts
        ) 
        if cluster_algo == 'kmeans':
            self.clustering.k_means_clustering(n_clusters=n_clusters)
        elif cluster_algo == 'sparse_subspace_clustering':
            # Ensure  directory exists
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            filepath = os.path.join(
                folderpath, 
                utils_general.create_filename(
                    'sparse_repr_matrix', self.target_class, len(self.class_imgs),
                    extension='.npz'
                )
            )
            self.clustering.sparse_subspace_clustering(filepath, outlier_percentile, n_clusters, max_samples)
        else:
            raise ValueError("[ERROR] Clustering algorithm not recognized")
        
        self.concepts = []
        for cluster in self.clustering.clusters:
            if self._check_cluster(cluster.segments, min_size, min_coverage):
                self.concepts.append(
                    classes.ConceptClass(
                        label=cluster.label,
                        segments=cluster.segments,
                        norm_acts=cluster.norm_acts,
                        max_samples=max_samples
                    )
                )
                
        self.max_samples_per_concept = max_samples
        print(f"[INFO] Considered {len(self.concepts)}/{len(self.clustering.clusters)} clusters as concepts")
        
    def compute_concept_subspace_bases(self, subspace_dimensionality:int=None, est_mode:str='ratio',
                                        compt_princ_angl:bool=False):
        self.concept_bases = []
        for concept in self.concepts:
            concept.compute_pca_basis(num_concepts=len(self.concepts), dim=subspace_dimensionality, est_mode=est_mode)
            self.concept_bases.append(concept.basis)
        self.compl_basis = utils_mcd.basis_of_ortho_complement(self.concept_bases)

        if compt_princ_angl:
            self.subspace_pricipal_angles = np.empty(
                (len(self.concept_bases), len(self.concept_bases)), 
                dtype=np.dtype([('min_distance', np.float64), ('max_distance', np.float64)])
            )

            for i, subspace_i in enumerate(self.concept_bases):
                for j, subspace_j in enumerate(self.concept_bases):
                    pas = utils_mcd.principle_angles_subspace_subspace(subspace_i, subspace_j)
                    self.subspace_pricipal_angles[i, j] = (pas.min(), pas.max())
            
    def concept_quantification(self, vector:np.ndarray):
        if not hasattr(self, 'concept_bases'):
            raise ValueError("First compute the subspace bases for concepts.")
        
        if vector.ndim > 1 and vector.shape[0] != 1:
            raise ValueError("Only one-dimensional vectors allowed")
        
        # decompose weight vector into concepts
        concept_proj = utils_mcd.subspace_projection(
            subspace_bases=self.concept_bases+[self.compl_basis],
            vector=vector
        )
        norm_concept_proj = norm(concept_proj,axis=1)

        # calculate the importance score of each concept as the squared norm of the projection of 
        # the weight vector in the concept subspace scaled by the sqaured norm of the weight vector
        importance_scores = norm_concept_proj**2 / norm(vector)**2
        
        if hasattr(self, 'subspace_pricipal_angles'):
            lower_bound = []
            upper_bound = []
            for i, subspace_i in enumerate(self.concept_bases): 
                for j, subspace_j in enumerate(self.concept_bases):
                    if i != j:
                        lower_bound.append(
                            norm_concept_proj[i]*norm_concept_proj[j] * np.cos(self.subspace_pricipal_angles[i,j]['max_distance'])
                        )
                        upper_bound.append(
                            norm_concept_proj[i]*norm_concept_proj[j] * np.cos(self.subspace_pricipal_angles[i,j]['min_distance'])
                        )
            importance_bounds = (sum(norm_concept_proj ** 2) + sum(lower_bound), sum(norm_concept_proj ** 2) + sum(upper_bound))

            return importance_scores, importance_bounds

        return importance_scores, None
    
    def concept_activations(self, acts:np.ndarray, batch_sizes=1, norm_batch:bool=True, n_jobs=-1):
        n_acts = acts.shape[0]
        if not isinstance(batch_sizes, list):
            batch_sizes = [batch_sizes] * np.ceil(n_acts / batch_sizes).astype(int)
        start_indices = [0] + list(accumulate(batch_sizes))[:-1]
        similarities = np.zeros((n_acts, len(self.concept_bases)+1))  # +1 for complement

        results = Parallel(n_jobs=n_jobs)(
            delayed(utils_mcd.batch_concept_activations)(
                acts[b_start:b_start+b_size], self.concept_bases + [self.compl_basis], norm_batch
            ) for b_start, b_size in tqdm(
                zip(start_indices, batch_sizes), 
                desc='[INFO] Computing concept activations',
                total=len(batch_sizes)
            )
        )

        for b_start, b_size, b_result in zip(start_indices, batch_sizes, results):
            similarities[b_start:b_start+b_size] = b_result

        return similarities

    def concept_relevances(self, acts:np.ndarray, n_jobs=-1):
        if len(acts.shape) != 2:
            raise ValueError("Expect feature vectors to be flattened")
        if not hasattr(self, 'concept_bases'):
            raise ValueError("First compute the subspace bases for concepts.")
                
        concept_projs = np.empty((acts.shape[0], len(self.concept_bases)+1, acts.shape[1]))

        results = Parallel(n_jobs=n_jobs)(
            delayed(utils_mcd.subspace_projection)(
                self.concept_bases + [self.compl_basis], feature_vec
            ) for feature_vec in tqdm(
                acts, 
                desc='[INFO] Computing concept relevance scores'
            )
        )
        
        for idx, proj in enumerate(results):
            concept_projs[idx] = proj

        class_idx = utils_general.get_imagenet_class_index(self.target_class)
        weight_vec = self.model.fc.weight.data.detach()[class_idx].cpu().numpy()
        
        return concept_projs@weight_vec
    
    def compute_tcav_scores(self, rdm_imgs_folder:str='random', mimic_segm:bool=False, segm_cache_dir:str='rdm_segms', 
                                    num_rdm_runs:int=20, acts_cache_dir:str='acts', training_mode:str='sklearn',
                                    num_epochs:int=50, num_test_images:int=100):
        # load random images 
        random_imgs = self._load_images(
            folderpath=os.path.join(self.source_dir, rdm_imgs_folder),
            num_imgs=len(self.class_imgs) if mimic_segm else 2*num_rdm_runs*self.max_samples_per_concept,
            max_shortest_side=self.max_shortest_side
        )
        if mimic_segm:
            # create segmentations from random images which are using for training the CAVs
            self._segment_images(random_imgs, self.segm_algo, segm_cache_dir)
        else:
            # represent each image by one segments that covers the whole image (effectively considering the whole image)
            for rdm_img in random_imgs:
                rdm_img.segments = [classes.SegmentClass(
                    mask=np.ones(rdm_img.img_numpy.shape[:2]),
                    original_image=rdm_img
                )]
        # load/calculate activations for random image segments
        self._load_or_calc_acts(
            cache_dir=acts_cache_dir,
            identifier=rdm_imgs_folder + ('_segms' if mimic_segm else '_imgs'),
            images=random_imgs,
            cropping_mode=self.cropping_mode if mimic_segm else 0,
            use_masks=self.use_masks if mimic_segm else False,
            masking_mode=self.masking_mode if mimic_segm else None,
            batch_size=self.batch_size
        )
        random_segms_acts = []
        for rdm_img in random_imgs:
            for segm in rdm_img.segments:
                random_segms_acts.append(segm.model_act)
        random_segms_acts = np.stack(random_segms_acts)
        
        # create a random concept using random segmentations from the target class images
        rdm_segments = []
        if mimic_segm:
            # take random segments from the same class
            for cls_img in self.class_imgs:
                rdm_segments.extend(cls_img.segments)
        else:
            # take random (full) images (as in ACE)
            for rdm_img in random_imgs:
                rdm_segments.extend(rdm_img.segments)

        self.rdm_concept=classes.ConceptClass(
            label=-1,
            segments=random.sample(rdm_segments, self.max_samples_per_concept),
            norm_acts=False,
            max_samples=self.max_samples_per_concept
        )
        
        # train CAV for each concept (incl. random concept)
        for concept in tqdm(self.concepts + [self.rdm_concept], desc='[INFO] Training CAV'):
            concept.train_cavs(
                rdm_acts=random_segms_acts,
                mode=training_mode,
                n_runs=num_rdm_runs,
                n_epochs=num_epochs,
                batch_size=self.batch_size
            )
            
        # load random class images to calcualte TCAV scores
        rdm_cls_imgs = self._load_images(
            folderpath=os.path.join(self.source_dir, self.target_class),
            num_imgs=num_test_images,
            max_shortest_side=self.max_shortest_side,
            shuffle=True
        )
        
        # little trick to use the same functions: Each image is represented as unmasked 
        # 'segment' (= whole image)
        for cls_img in rdm_cls_imgs:
            cls_img.segments = [classes.SegmentClass(
                mask=np.ones(cls_img.img_numpy.shape[:2]),
                original_image=cls_img
            )]

        # Prepare dataset and create dataloader
        dataset = classes.ConceptDatasetClass(
            rdm_cls_imgs, 
            self.model.default_cfg, 
            cropping_mode=0, 
            use_masks=False,
            masking_mode=None,
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                collate_fn=utils_general.custom_collate)
        
        # get the gradients w.r.t. target class for the class images of the specified layer
        cls_gradients, cls_pred = utils_general.compute_activations(
            self.model,
            self.layer_name,
            dataloader=dataloader,
            cls_idx=utils_general.get_imagenet_class_index(
                self.target_class
            )
        )
        
        # calculate TCAV scores of random concepts later used for statistical testing
        self.rdm_concept.calc_tcav_scores(cls_gradients)
        
        for concept in self.concepts:
            # calculate TCAV scores
            concept.calc_tcav_scores(cls_gradients)
            # perform statistical testing
            concept.calc_p_value(self.rdm_concept.tcav_scores)

    def get_tcav_scores(self):
        mean_tcav_scores = []
        std_tcav_scores = []
        p_values = []
        
        for concept in self.concepts:
            # calculate TCAV scores
            mean_tcav_scores.append(np.mean(concept.tcav_scores))
            std_tcav_scores.append(np.std(concept.tcav_scores))
            # perform statistical testing
            p_values.append(concept.p_value)
    
        return mean_tcav_scores, std_tcav_scores, p_values
    
    def get_matching_concept_idx(self, activation:np.ndarray):
        if norm(activation) == 0:
            return None
        matching_cluster = self.clustering.get_clostest_cluster(activation)

        # Iterate over the second list and find the object with the matching label
        for idx_c, concept in enumerate(self.concepts):
            if concept.label == matching_cluster.label:
                return idx_c
        return None
    