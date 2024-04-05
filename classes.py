import numpy as np
from skimage.segmentation import slic
import os
from PIL import Image
import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean
from sklearn.cluster import k_means
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skdim.id import lPCA
from sklearn.decomposition import PCA
from scipy.linalg import norm
from matplotlib import gridspec as gridspec
import random
from sklearn.preprocessing import normalize
from scipy import sparse
from sklearn.metrics import pairwise_distances_argmin_min
from utils import utils_general, utils_mcd, utils_ace

class ImageClass:
    def __init__(self, filename:str, max_shortest_side:int):
        self.filename = filename
        self.img_pil = None
        self.load_image(max_shortest_side)
        self.segments = []
    
    def load_image(self, max_shortest_side:int):
        self.img_pil = Image.open(self.filename)
        w, h = self.img_pil.size
        shortest_side = min(w, h)

        # Resize the image if the shortest side is larger than max_short_side
        if shortest_side > max_shortest_side:
            if h > w:
                new_height = int(max_shortest_side * (h / w))
                new_size = (max_shortest_side, new_height)
            else:
                new_width = int(max_shortest_side * (w / h))
                new_size = (new_width, max_shortest_side)

            self.img_pil = self.img_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        self.img_numpy = np.array(self.img_pil, dtype=np.float32) / 255.0
        if self.img_numpy.shape[2] != 3:
            raise Exception("Loaded image does not have 3 channels.")
    
    def _segment_slic(self, jaccard_threshold:float=0.5):
        self.segments = []
        for n_segms in [15, 50, 80]:    
            img_masked = slic(self.img_numpy, n_segments=n_segms, sigma=1, compactness=50)
            for segm_idx in range(img_masked.max()):
                mask = (img_masked == segm_idx).astype(np.float32)
                if np.mean(mask) > 0.001:
                    unique = True
                    for segm in self.segments:
                        jaccard = np.sum(segm.mask * mask) / np.sum((segm.mask + mask) > 0)
                        if jaccard > jaccard_threshold:
                            unique = False
                            break
                    if unique:
                        self.segments.append(
                            SegmentClass(
                                mask=mask,
                                original_image=self
                            )
                        )
    
    def _segment_sam(self, mask_generator, min_area_ration:float=0.01, jaccard_threshold:float=0.5): 
        masks = mask_generator.generate(
            np.array(self.img_pil)
        )
        
        # sort masks by their predicted IoU
        sorted_masks = [mask['segmentation'] for mask in sorted(masks, key=(lambda x: x['predicted_iou']), reverse=True)]
        self.segments = [
            SegmentClass(
                mask.astype(np.float32), self
            ) for mask in utils_general.get_unique_concept_masks(
                sorted_masks,
                min_area_ration
            )
        ]
    
    def load_segments(self, cache_dir:str='segms', sam_model=None):
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Construct cache filename based on image filename and segmentation algorithm
        algo = 'slic' if sam_model == None else 'sam'
        cache_filename = os.path.join(cache_dir, f"{os.path.basename(self.filename)}_{algo}.npy")

        # Check if cached segments exist
        if os.path.isfile(cache_filename):
            # Load segments from cache
            self.segments = [
                SegmentClass(
                    mask=mask, original_image=self
                ) for mask in np.load(cache_filename, allow_pickle=False)
            ]
        else:
            # Compute segments using the specified algorithm
            if sam_model == None:
                self._segment_slic()
            else:
                self._segment_sam(sam_model)
            
            # Save computed segments to cache
            np.save(cache_filename, np.array([segm.mask for segm in self.segments], dtype=np.float32), allow_pickle=False)

class SegmentClass:
    def __init__(self, mask:np.ndarray, original_image:ImageClass):
        self.mask:np.ndarray = mask
        if self.mask.ndim != 2:
            raise ValueError("Mask must be a 2D array")
        self.org_img:ImageClass = original_image
        self.model_act:np.ndarray = None
        self.model_pred:np.ndarray = None
        
    @staticmethod
    def get_padded_img(img:np.ndarray, mask:np.ndarray, fill_value:float):
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        return img * mask + (1 - mask) * fill_value
    
    @staticmethod
    def get_quadratic_bounding_box(mask:np.ndarray) -> tuple:
        """
        Find the minimum quadratic bounding box for a mask.
    
        Parameters:
        mask (numpy.ndarray): A 2D array representing the mask.
    
        Returns:
        tuple: Coordinates of the top-left and bottom-right corners of the bounding box.
        """
    
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Determine the dimensions of the bounding box
        width = xmax - xmin
        height = ymax - ymin
        
        # Make the bounding box quadratic
        if width > height:
            diff = width - height
            ymin = max(0, ymin - diff // 2)
            ymax = ymin + width  # Make height equal to width
        
            # Ensure ymax doesn't exceed mask boundaries
            if ymax > mask.shape[0] - 1:
                ymax = mask.shape[0] - 1
                ymin = max(0, ymax - width)
        
        elif height > width:
            diff = height - width
            xmin = max(0, xmin - diff // 2)
            xmax = xmin + height  # Make width equal to height
        
            # Ensure xmax doesn't exceed mask boundaries
            if xmax > mask.shape[1] - 1:
                xmax = mask.shape[1] - 1
                xmin = max(0, xmax - height)
    
        return (ymin, ymax, xmin, xmax)
    
    def get_cropped_segment(self, quadratic:bool=True):
        if quadratic:
            ymin, ymax, xmin, xmax = self.get_quadratic_bounding_box(self.mask)
        else:
            ones = np.where(self.mask == 1)
            ymin, ymax, xmin, xmax = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        return self.org_img.img_numpy[ymin:ymax, xmin:xmax]
    
    def get_cropped_segment_mask(self, quadratic:bool=True):
        if quadratic:
            ymin, ymax, xmin, xmax = self.get_quadratic_bounding_box(self.mask)
        else:
            ones = np.where(self.mask == 1)
            ymin, ymax, xmin, xmax = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        return self.mask[ymin:ymax, xmin:xmax]
    
class ConceptDatasetClass(Dataset):
    def __init__(self, images:list[ImageClass], model_cfg:dict, cropping_mode:int=0, 
                 use_masks:bool=True, masking_mode:int=-1, erosion_threshold:float=1.0,
                 fill_value:float=0.4588):
        self.segments = []
        self.masks = []
        self.use_masks = use_masks
        self.masking_mode = masking_mode
        self.erosion_threshold = erosion_threshold
        for img in images:
            for segm in img.segments:
                if cropping_mode > 0:
                    if use_masks:
                        self.segments.append(segm.get_cropped_segment(quadratic=(cropping_mode==2)))
                        self.masks.append(segm.get_cropped_segment_mask(quadratic=(cropping_mode==2)))
                    else:
                        self.segments.append(
                            segm.get_padded_img(
                                segm.get_cropped_segment(quadratic=(cropping_mode==2)), 
                                segm.get_cropped_segment_mask(quadratic=(cropping_mode==2)), 
                                fill_value
                            )
                        )
                else:
                    if use_masks:
                        self.segments.append(segm.org_img.img_numpy)
                        mask = segm.mask
                        if np.mean(mask) > self.erosion_threshold and self.masking_mode == -1:
                            mask = utils_general.shrink_mask(mask)
                        self.masks.append(mask)
                    else:
                        self.segments.append(
                            segm.get_padded_img(
                                segm.org_img.img_numpy, segm.mask, fill_value
                            )
                        )
            
        self.resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(model_cfg['input_size'][1:]),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Compose([
            transforms.Normalize(
                mean=model_cfg['mean'], std=model_cfg['std']
            )
        ])

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]

        if segment.max() <= 1:
            segment = (segment * 255).astype(np.uint8)
        segment = self.normalize(self.resize(segment))
        
        if self.use_masks:
            mask = self.masks[idx]
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
            mask = self.resize(mask.astype(np.uint8))
            mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
            return (segment, mask, self.masking_mode)
        else:
            return segment

class ClusterClass():
    def __init__(self, label:int, segments:list[SegmentClass], norm_acts:bool):
        self.label = label
        self.norm_acts:bool = norm_acts
        self.segments = segments
        self.centroid = self.embeddings.mean(axis=0)
        self._sort_segms()
    
    @property
    def embeddings(self):
        cluster_acts = np.array([segm.model_act for segm in self.segments])
        if self.norm_acts:
            cluster_acts = normalize(cluster_acts, norm='l2', axis=1)
        return cluster_acts

    def _sort_segms(self):
        self.segments = [
            segm for _, segm in sorted(
                zip(self.embeddings, self.segments), 
                key=lambda pair: euclidean(pair[0], self.centroid)
            )
        ]
        
    def get_segments(self, mode:str='max', num:int=None):
        if num is None or num > len(self.segments):
            num = len(self.segments)
        if mode == 'max':
            concept_segms = self.segments[:num]
        elif mode == 'random':
            concept_segms = random.sample(self.segments, num)
        elif mode == 'diverse':
            concept_segms = []
            while True:
                seen = set()
                for segm in self.segments:
                    segm_img_name = segm.org_img.filename
                    if segm_img_name not in seen and segm not in concept_segms:
                        seen.add(segm_img_name)
                        concept_segms.append(segm)
                if len(concept_segms) >= num:
                    break
        else:
            raise ValueError('Invalid mode!')

        return concept_segms[:num]
        
class ClusterSpaceClass():
    def __init__(self, images:list[ImageClass], norm_acts:bool=True):
        self.segms:list[SegmentClass] = []
        for img in images:
            self.segms.extend(img.segments)
        self.norm_acts:bool = norm_acts
        self.labels:np.ndarray = None
        self.clusters:list[ClusterClass] = None
    
    def get_clostest_cluster(self, activation:np.ndarray):
        if len(activation.shape) > 1:
            raise ValueError("[ERROR] activation expected to be 1-dim vector")
        if self.norm_acts:
            activation = activation / norm(activation)
        cluster_score = []
        for cluster in self.clusters:
            cluster_score.append(euclidean(activation, cluster.centroid))
        matching_cluster_idx = np.argmin(cluster_score)
        return self.clusters[matching_cluster_idx]
    
    def _create_clusters(self, labels_to_exclude:list=[]):
        self.clusters:list[ClusterClass] = []
        unique_labels = np.unique(self.labels)
        for label_idx in unique_labels:
            if label_idx not in labels_to_exclude:
                cluster_idxs = np.where(self.labels == label_idx)[0]
                self.clusters.append(
                    ClusterClass(
                        label=label_idx,
                        segments=[self.segms[segm_idx] for segm_idx in cluster_idxs],
                        norm_acts=self.norm_acts
                    )
                )

    def _approximate_num_clusters(self, method:str='segments_per_image')->int:
        if method == 'segments_per_image':
            _, segms_per_img = np.unique(
                [segm.org_img.filename for segm in self.segms],
                return_counts=True
            ) 
            num_clusters = int(segms_per_img.mean() + segms_per_img.std())
            print(f"[INFO] Performing clustering with {num_clusters} clusters")
            return num_clusters
            
    def k_means_clustering(self, n_clusters:int=None):
        if n_clusters == None:
            n_clusters = self._approximate_num_clusters()
            
        segm_acts:np.ndarray = np.stack([segm.model_act for segm in self.segms])
        if self.norm_acts:
            segm_acts = normalize(segm_acts, axis=1, norm='l2')
        centroids, self.labels, cost = k_means(
            segm_acts, n_clusters=n_clusters, random_state=43,
        )
        self._create_clusters()

    def load_or_comp_sparse_repr_matrix(self, filepath:str, outlier_percentile:float=0.75):
        # filepath = os.path.join(filepath if '.npz' in filepath else filepath + '.npz') 
        if os.path.exists(filepath):
            self.sparse_repr_matrix = sparse.load_npz(filepath)
            self.outlier_mask = np.load(filepath.replace(".npz","_outlier.npy"))
            print(f"[INFO] Used cached sparse representation matrix: {filepath}")
            return 

        segm_acts = np.stack([segm.model_act for segm in self.segms])
        if self.norm_acts:
            segm_acts = normalize(segm_acts, axis=1, norm='l2')

        self.sparse_repr_matrix = utils_mcd.compute_sparse_repr_matrix(
            acts=segm_acts
        )
        self.outlier_mask = utils_mcd.get_outlier_mask(self.sparse_repr_matrix, outlier_percentile)
        
        if outlier_percentile < 1.0:
            self.sparse_repr_matrix = utils_mcd.compute_sparse_repr_matrix(
                acts=segm_acts[~self.outlier_mask],
            )
            
        sparse.save_npz(filepath, self.sparse_repr_matrix)
        np.save(filepath.replace(".npz","_outlier.npy"), self.outlier_mask)
        print(f"[INFO] Saved sparse representation matrix: {filepath}")

    def _spectral_clustering(self, affinity_matrix, n_clusters:int=None, max_samples:int=None):
        if n_clusters == None:
            n_clusters = self._approximate_num_clusters()
        # compute laplacian matrix
        laplacian = sparse.csgraph.laplacian(affinity_matrix, normed=True)
        _, vec = sparse.linalg.eigsh(
            sparse.identity(laplacian.shape[0]) - laplacian, 
            k=n_clusters,
            sigma=None,
            which='LA'
        )
        embedding = normalize(vec)

        centroids, labels, _ = k_means(
            embedding, n_clusters, random_state=43, n_init='auto'
        )
        
        if max_samples != None:
            # Retain only the top n_points closest to every cluster centroid
            closest_points_indices = []
            for i in range(n_clusters):
                cluster_points_indices = np.where(labels == i)[0]
                distances_to_centroid = pairwise_distances_argmin_min(embedding[cluster_points_indices], [centroids[i]])[1]
                closest_points_indices.extend(cluster_points_indices[np.argsort(distances_to_centroid)[:max_samples]])
        
            labels = np.full(embedding.shape[0], -1)
            labels[closest_points_indices] = labels[closest_points_indices]
        return labels
    
    def sparse_subspace_clustering(self, filepath, outlier_percentile:float, n_clusters:int, max_samples:int):
        if not hasattr(self, 'sparse_repr_matrix'):
            self.load_or_comp_sparse_repr_matrix(filepath, outlier_percentile)
        # normalize sparse self representation matrix using l2-norm
        sparse_repr_matrix_normed = normalize(self.sparse_repr_matrix, 'l2')
        # compute the affinity matrx
        affinity_matrix = 0.5 * (np.absolute(sparse_repr_matrix_normed) + np.absolute(sparse_repr_matrix_normed.T))
        # compute the labels of inlying datapoints
        # inlier_labels = spectral_clustering(affinity=self.affinity_matrix, n_clusters=n_clusters, random_state=43)
        inlier_labels = self._spectral_clustering(affinity_matrix, n_clusters, max_samples)
    
        # initialize the labels of all datapoints as -1 (outlier)
        self.labels = -1*np.ones_like(self.outlier_mask)
        # only set the labels of the inlying datapoints based on the results of spectral clustering
        self.labels[~self.outlier_mask] = inlier_labels
    
        self._create_clusters(labels_to_exclude=[-1])
        
class ConceptClass(ClusterClass):
    def __init__(self, label:int, segments:list[SegmentClass], norm_acts:bool, max_samples:int=None):
        super().__init__(label, segments, norm_acts)
        self.max_samples:int = max_samples
        self.basis:np.ndarray = None
        self.mcd_scores:list[float] = []
        self.mcd_scores_bounds:tuple[float,float] = None
        self.cavs:dict = {}
        self.tcav_scores:list[float] = []
        self.p_value:float=None

    def _estimate_dim(self, acts:np.ndarray, mode:str) -> int:
        """
        Estimates the instrinsic dimensionality of concept subspaces bases on PCA.
        For hyperparameters refer to https://scikit-dimension.readthedocs.io/en/latest/skdim.id.lPCA.html
        """
        if mode=='FO':
            lpca = lPCA(ver='FO')
        elif mode=='ratio':
            # use top principal components to preserve 80 % of the variance
            lpca = lPCA(ver='ratio', alphaRatio=0.8)
        else:
            raise ValueError("[ERROR] Intrinsic dimensionality estimation mode not recognized")
        return lpca.fit_transform(acts)
    
    def compute_pca_basis(self, num_concepts:int, dim:int=None, est_mode:str='ratio') -> dict:
        cluster_acts = np.stack([segm.model_act for segm in self.get_segments(mode='diverse', num=self.max_samples)])
        # determine dimensionality of concept subspace
        est_dim = self._estimate_dim(cluster_acts, est_mode) if dim is None else dim
        # ensure that the union of concept subspaces does not exceed the feature space dimensionality
        est_dim = min(
            (est_dim, cluster_acts.shape[1] // num_concepts)
        )
        # apply PCA on concept activations
        pca = PCA(n_components=None, random_state=42)
        pca.fit(cluster_acts)
        # take top principal components
        self.basis = pca.components_.copy()[:est_dim]

    def train_cavs(self, rdm_acts:np.ndarray, mode:str='sklearn', n_runs:int=20, 
                   n_epochs:int=50, batch_size:int=64):
        self.cavs = {}
        concept_acts = np.stack(
            [segm.model_act for segm in self.get_segments(mode='max', num=self.max_samples)]
        )
        for idx_run in range(n_runs):
            rdm_acts = rdm_acts[np.random.choice(len(rdm_acts), size=self.max_samples, replace=False)]
            
            X_data = np.concatenate(
                [
                concept_acts.reshape(concept_acts.shape[0], -1),
                rdm_acts.reshape(rdm_acts.shape[0], -1)
                ], 
                axis=0
            )
            # all concept activations have label 1, and all 
            # random activations have label 0
            y_labels = np.concatenate(
                # [np.ones(min_num_acts), np.zeros(min_num_acts)], 
                [np.ones(concept_acts.shape[0]), np.zeros(rdm_acts.shape[0])], 
                axis=0
            )

            if mode=='pytorch':
                cav_dict = utils_ace.cav_pytorch_training(
                    X=X_data,
                    y=y_labels,
                    batch_size=batch_size,
                    num_epochs=n_epochs
                )
            elif mode=='sklearn':
                cav_dict = utils_ace.cav_sklearn_training(
                    X=X_data,
                    y=y_labels
                )
            
            self.cavs[idx_run] = cav_dict

    def calc_tcav_scores(self, cls_gradients:np.ndarray):
        self.tcav_scores = []
        for idx_run, cav_dict in self.cavs.items():
            cav_normed = cav_dict['weight_vector'] / norm(cav_dict['weight_vector'])
            prod = cls_gradients @ cav_normed
            self.tcav_scores.append(np.mean(prod < 0))
    
    def calc_p_value(self, rdm_concept_tcav_scores:list[float]):
        self.p_value = utils_ace.statistical_testings(
            self.tcav_scores, rdm_concept_tcav_scores
        )