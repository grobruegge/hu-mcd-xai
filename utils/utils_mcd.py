import numpy as np
from scipy.linalg import subspace_angles, orth, solve, norm
from sklearn.decomposition import sparse_encode
from tqdm import tqdm
from scipy import sparse
from joblib import Parallel, delayed

def is_orthonormal(matrix:np.ndarray, tol:float=1e-5):
    """Verifies that the vector in a matrix of shape (n_vecs, dim) are
    orthonormalized by checking whether matrix^T @ matrix = identity

    Args:
        matrix (np.ndarray): matrix contaning vectors
        tol (float, optional): tolerance to address small inaccuracies introduced by
            floating-point arithmetic in computers. Defaults to 1e-6.

    Returns:
        bool: True if vectors are orthonormalized, False otherwise
    """
    dot_product = matrix @ matrix.T
    identity_matrix = np.eye(dot_product.shape[1])
    return np.allclose(dot_product, identity_matrix, atol=tol)

def comp_projection_matrix(basis:np.ndarray):
    """ Compute projection matrix of a given basis. 

    Args:
        basis (np.ndarray): basis vectors as matrix of shape (num_vecs, dim)

    Returns:
        Tuple: (projection matrix, projection matrix onto orthogonal complement)
    """
    # compute projection matrix
    if is_orthonormal(basis):
        # formula simplifies
        proj_matrix = basis.T@basis
    else:
        print("[WARNING] basis vectors are not orthonormalized")
        proj_matrix = basis.T @ np.linalg.inv(basis @ basis.T) @ basis
    
    # compute projection matrix onto the orthogonal complement
    identity_matrix = np.eye(*proj_matrix.shape)
    proj_matrix_compl = identity_matrix - proj_matrix
    
    return proj_matrix, proj_matrix_compl
    
def othonormal_basis_of_union(bases:list) -> np.ndarray:
    """Computes orthonormal basis given the basis vectors in the list

    Args:
        bases (list): list of np.ndarrays

    Returns:
        np.ndarray: basis vectors of orthonormal basis
    """
    union = np.concatenate(bases)
    orthonormal_basis = orth(union.T).T
    return orthonormal_basis

def calc_completeness(weight_vector:np.ndarray, concept_bases:list):
    """Calculate completness of concept based on the projection of the weight
    vector of the linear classification layer of the target model to be explained
    into the union of all concept subspaces
    """
    weight_vector = weight_vector / norm(weight_vector)
    # create a basis orthonormal feature space basis using all concept bases
    orthonormal_conceptspace_basis = othonormal_basis_of_union(
        concept_bases
    )
    P, P_c = comp_projection_matrix(orthonormal_conceptspace_basis)
    return norm(P@weight_vector) #, norm(P_c@self.weight_vec_normed)
    
        
def basis_of_ortho_complement(concept_bases:list[np.ndarray]):
    # calc orthonormalized basis spanned by concept subspaces
    orthonormal_concept_basis = othonormal_basis_of_union(
        concept_bases
    )
    
    if orthonormal_concept_basis.shape[0] == orthonormal_concept_basis.shape[1]:
        # print("[INFO] subspace spanned by all concept already coveres the entire feature space")
        return np.empty(shape=(1,orthonormal_concept_basis.shape[1]))
    
    _, P_compl = comp_projection_matrix(orthonormal_concept_basis)
    
    try:
        complement_basis = orth(P_compl.T, rcond=1e-5).T
        assert (
            (complement_basis.shape[0] + orthonormal_concept_basis.shape[0]) == orthonormal_concept_basis.shape[1], 
            "basis does not cover orhogonal complement"
        )
    except AssertionError:
        complement_basis = orth(P_compl.T, rcond=1e-4).T
        assert (
            (complement_basis.shape[0] + orthonormal_concept_basis.shape[0]) == orthonormal_concept_basis.shape[1], 
            "basis does not cover orhogonal complement"
        )
        
    return complement_basis

def decompose_vector(vec:np.ndarray, basis_vecs:np.ndarray):
    # check whether the union of concept bases form a square matrix 
    if basis_vecs.shape[0] == basis_vecs.shape[1]:
        coeffs = solve(basis_vecs.T, vec)
    else:
        print("[WARNING] concept space union does not match the feature space dimensionality")
        coeffs, residuals, rank, s = np.linalg.lstsq(basis_vecs.T, vec, rcond=None)
    
    return coeffs

def get_subspace_ranges(subspace_bases:list[np.ndarray]):
    d_i = np.array([basis.shape[0] for basis in subspace_bases])
    start_idx = np.insert(d_i.cumsum(), 0, 0)[:-1]
    end_idx = d_i.cumsum()
    return start_idx, end_idx

def subspace_projection(subspace_bases:list[np.ndarray], vector:np.ndarray):
    # create a matrix which contains the union of all concept bases (incl. compl. basis)
    union_matrix = np.concatenate(subspace_bases) # C

    # decompose the vector given all concept basis vectors
    coeffs = decompose_vector(vector, union_matrix) # w_i^l
    
    # project vector into the union of all concept basis vectors
    vec_proj = coeffs[:,np.newaxis] * union_matrix # w_i^l*c_i^l 
    
    # calculate the lenght of the sum of each concept subspace projection
    start_idxs, end_idxs = get_subspace_ranges(subspace_bases)
    concept_subspace_proj = np.array(
        [vec_proj[s:e].sum(axis=0) for s, e in zip(start_idxs, end_idxs)]
    ) # w^l 
    
    return concept_subspace_proj
    
def principle_angles_subspace_subspace(A,B):
    """
    Use scipy implementation for subspace angles here which computes SVD for cosine and sine.
    This is good for precision but slow.

    Parameters
    -----------   
    A, B: np.ndarray, matrix with basis vectors for subspace of dimensionality p in R^d of shape (p, n)
    """
    if(len(A.shape)==1):
        A= np.expand_dims(A,axis=0)
    if(len(B.shape)==1):
        B = np.expand_dims(B,axis=0)
    try:
        pas = np.array(subspace_angles(A.T,B.T))
    except:
        pas = None
    return pas

def grassmann_distance_from_principal_angles(pas):
    return np.sqrt(np.sum(np.power(pas,2)))

def subspac_distance_matrix(subspaces):
    distance_matrix = np.empty((len(subspaces), len(subspaces)))
    for i,A in enumerate(subspaces):
        for j,B in enumerate(subspaces):
            pas = principle_angles_subspace_subspace(A,B)
            distance_matrix[i,j] = grassmann_distance_from_principal_angles(pas)
    return distance_matrix

def active_support_elastic_net(X, y, alpha, tau=1.0, support_size=100, maxiter=40):

    n_samples = X.shape[0]

    if n_samples <= support_size:  # skip active support search for small scale data
        supp = np.arange(n_samples, dtype=int)  # this results in the following iteration to converge in 1 iteration
    else:    
        supp = np.argpartition(-np.abs(np.dot(y, X.T)[0]), support_size)[0:support_size]

    curr_obj = float("inf")
    for _ in range(maxiter):
        Xs = X[supp, :]
        
        cs = sparse_encode(y, Xs, algorithm='lasso_lars', alpha=alpha)
      
        delta = (y - np.dot(cs, Xs)) / alpha
		
        obj = tau * np.sum(np.abs(cs[0])) + (1.0 - tau)/2.0 * np.sum(np.power(cs[0], 2.0)) + alpha/2.0 * np.sum(np.power(delta, 2.0))
        if curr_obj - obj < 1.0e-10 * curr_obj:
            break
        curr_obj = obj
			
        coherence = np.abs(np.dot(delta, X.T))[0]
        coherence[supp] = 0
        addedsupp = np.nonzero(coherence > tau + 1.0e-10)[0]
        
        if addedsupp.size == 0:  # converged
            break

        # Find the set of nonzero entries of cs.
        activesupp = supp[np.abs(cs[0]) > 1.0e-10]  
        
        if activesupp.size > 0.8 * support_size:  # this suggests that support_size is too small and needs to be increased
            support_size = min([round(max([activesupp.size, support_size]) * 1.1), n_samples])
        
        if addedsupp.size + activesupp.size > support_size:
            ord = np.argpartition(-coherence[addedsupp], support_size - activesupp.size)[0:support_size - activesupp.size]
            addedsupp = addedsupp[ord]
        
        supp = np.concatenate([activesupp, addedsupp])
    
    c = np.zeros(n_samples)
    c[supp] = cs
    return c

def compute_sparse_repr_matrix(acts: np.ndarray, tau: float = 1.0, gamma: float = 10.0, n_jobs=-1):
    n_acts = acts.shape[0]
    coh_matrix = np.absolute(np.dot(acts, acts.T))

    def compute_row(idx):
        y = acts[idx, :].copy().reshape(1, -1)
        X = np.delete(acts, idx, axis=0)

        coh = np.delete(coh_matrix[idx], idx)
        alpha0 = np.amax(coh) / tau  
        alpha = alpha0 / gamma
        
        c = active_support_elastic_net(X, y, alpha=alpha, tau=tau)
        idxs_non_zero = np.flatnonzero(c)
        # Adjust indices to account for the removed row
        idxs_non_zero_adjusted = [i + 1 if i >= idx else i for i in idxs_non_zero]
        
        return idx, idxs_non_zero_adjusted, c[idxs_non_zero]

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_row)(i) for i in tqdm(
            range(n_acts), 
            desc='[INFO] Computing Self-Representation Matrix'
        )
    )

    rows, cols, vals = [], [], []
    for idx, idxs_non_zero, c_vals in results:
        rows.extend([idx] * len(idxs_non_zero))
        cols.extend(idxs_non_zero)
        vals.extend(c_vals)

    sparse_repr_matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n_acts, n_acts))
    return sparse_repr_matrix

def matrix_to_array(matrix):
    return np.squeeze(np.array(matrix.ravel()))

def get_outlier_mask(sparse_repr_matrix, percentile):
    W_l1 = matrix_to_array(np.abs(sparse_repr_matrix).sum(axis=1))
    return W_l1 > np.quantile(W_l1, percentile) 

def batch_concept_activations(batch_acts, featurespace_bases, norm_batch):
    if norm_batch:
        batch_acts = batch_acts / norm(batch_acts, axis=1).max()
    batch_similarities = np.empty(batch_acts.shape[0], len(featurespace_bases))
    for idx, act in enumerate(batch_acts):
        batch_similarities[idx] = norm(subspace_projection(subspace_bases=featurespace_bases, vector=act), axis=1)
        if not norm_batch:
            batch_similarities[idx] /= norm(act)
            
    return batch_similarities