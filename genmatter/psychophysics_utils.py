import genjax
import jax
import numpy as np
import jax.numpy as jnp
from genjax import Const, gen, Pytree
from .datatypes import Precomputed_DiscreteDistribution, StaticJnp, HDGMM_Gibbs_TraceWrapper, f_, pytree_stack, snp, inverse_wishart, discrete_categorical, truncate_eigenval_ratio
from .trace_wrappers import Super_Pytree, __hdgmm_TraceWrapper__
from genjax import gen, Pytree
from genjax import ChoiceMapBuilder as C
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import cKDTree
import cv2
import os

def generate_rotation_grid_2d(angle_max_deg=45, angle_step_deg=5):
    angles = jnp.deg2rad(jnp.arange(-angle_max_deg, angle_max_deg + angle_step_deg, angle_step_deg))

    cos_angles = jnp.cos(angles)
    sin_angles = jnp.sin(angles)

    rotation_matrices = jnp.stack([
        jnp.stack([cos_angles, -sin_angles], axis=-1),
        jnp.stack([sin_angles, cos_angles], axis=-1)
    ], axis=-2)

    return angles, rotation_matrices

@Pytree.dataclass
class HDGMM_Hyperparams(Super_Pytree):

    outlier_prob: jnp.float32 = Super_Pytree.field()
    outlier_velocity_gamma_shape: jnp.float32 = Super_Pytree.field()
    outlier_velocity_gamma_rate: jnp.float32 = Super_Pytree.field()

    alpha: jnp.float32 = Super_Pytree.field()
    beta: jnp.float32 = Super_Pytree.field()

    mu_H: jnp.ndarray = Super_Pytree.field()

    sigma_H: jnp.float32 = Super_Pytree.field()

    nu_H: jnp.float32 = Super_Pytree.field()
    Psi_H: jnp.ndarray = Super_Pytree.field()

    nu_B: jnp.float32 = Super_Pytree.field()
    Psi_B: jnp.ndarray = Super_Pytree.field()

    sigma_V: jnp.float32 = Super_Pytree.field()

    nu_V: jnp.float32 = Super_Pytree.field()
    Psi_V: jnp.ndarray = Super_Pytree.field()

    translation_max_radius: StaticJnp = Super_Pytree.static()
    translation_num_radii_cells: StaticJnp = Super_Pytree.static()
    translation_theta_step_deg: StaticJnp = Super_Pytree.static()
    translation_gaussian_scale: StaticJnp = Super_Pytree.static()
    rotation_vmf_kappa: StaticJnp = Super_Pytree.static()
    rotation_angle_max_deg: StaticJnp = Super_Pytree.static()
    rotation_angle_step_deg: StaticJnp = Super_Pytree.static()

    n_hyperblobs: jnp.int32 = Super_Pytree.static()
    n_blobs: jnp.int32 = Super_Pytree.static()
    n_datapoints: jnp.int32 = Super_Pytree.static()

    discrete_translation: Precomputed_DiscreteDistribution = Super_Pytree.static()
    discrete_rotation: Precomputed_DiscreteDistribution = Super_Pytree.static()

    def __eq__(self, other):
        if jax.tree_util.tree_structure(self) != jax.tree_util.tree_structure(other):
            return False
        leaves1 = jax.tree_util.tree_leaves(self)
        leaves2 = jax.tree_util.tree_leaves(other)
        bools = [jnp.all(l1 == l2) for l1, l2 in zip(leaves1, leaves2)]
        return jnp.all(jnp.array(bools))

    @classmethod
    def create(cls, **kwargs):
        
        max_radius = kwargs["translation_max_radius"].v
        num_radii = kwargs["translation_num_radii_cells"].v
        theta_step_deg = kwargs["translation_theta_step_deg"].v
        scale = kwargs["translation_gaussian_scale"].v
        angle_max_deg = kwargs["rotation_angle_max_deg"].v
        angle_step_deg = kwargs["rotation_angle_step_deg"].v
        kappa = kwargs["rotation_vmf_kappa"].v

        radii_positive = jnp.linspace(0.0, max_radius, num_radii)[1:]
        
        angle_step = jnp.deg2rad(theta_step_deg)
        angles = jnp.arange(0, 2*jnp.pi, angle_step)
        
        directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        
        translations_zero = jnp.zeros((1, 2))

        translations_positive = jnp.concatenate(
            [r * directions for r in radii_positive],
            axis=0
        )

        translation_support = jnp.concatenate([translations_zero, translations_positive], axis=0)
        
        translation_logprobs = jnp.sum(jax.scipy.stats.norm.logpdf(translation_support, 0.0, scale), axis=-1)

        angles, rotation_matrices = generate_rotation_grid_2d(angle_max_deg=angle_max_deg, angle_step_deg=angle_step_deg)
        
        rotation_logprobs = jax.scipy.stats.vonmises.logpdf(angles, kappa)
            
        return cls(
            **kwargs,
            discrete_translation=Precomputed_DiscreteDistribution(support=translation_support, logprobs=translation_logprobs),
            discrete_rotation=Precomputed_DiscreteDistribution(support=jnp.array(rotation_matrices), logprobs=rotation_logprobs)
        )


@Pytree.dataclass
class HDGMM_Hyperblobs_State(Super_Pytree):
    hyperblob_weights: jnp.ndarray
    hyperblob_means: jnp.ndarray
    hyperblob_covs: jnp.ndarray
    hyperblob_trans_vels: jnp.ndarray
    hyperblob_rot_vels: jnp.ndarray

@Pytree.dataclass
class HDGMM_Blobs_State(Super_Pytree):
    hyperblob_assignments: jnp.ndarray
    blob_weights: jnp.ndarray
    blob_means: jnp.ndarray
    blob_covs: jnp.ndarray
    blob_vel_means: jnp.ndarray
    blob_vel_covs: jnp.ndarray

@Pytree.dataclass
class HDGMM_Datapoints_State(Super_Pytree):
    blob_assignments: jnp.ndarray
    datapoint_positions: jnp.ndarray
    datapoint_vels: jnp.ndarray

@Pytree.dataclass
class HDGMM_State(Super_Pytree):
    hypers: HDGMM_Hyperparams
    hyperblobs_state: HDGMM_Hyperblobs_State
    blobs_state: HDGMM_Blobs_State
    datapoints_state: HDGMM_Datapoints_State

@gen
def HDGMM_model(hypers : HDGMM_Hyperparams):
    hyperblobs_state = HDGMM_hyperblobs_model(hypers) @ 'hyperblobs'
    blobs_state = HDGMM_blobs_model(hypers, hyperblobs_state) @ 'blobs'
    datapoints_state = HDGMM_datapoints_model(hypers, blobs_state) @ 'datapoints'

    return HDGMM_State(
        hypers = hypers,
        hyperblobs_state=hyperblobs_state, 
        blobs_state=blobs_state, 
        datapoints_state=datapoints_state
    )

@gen
def HDGMM_hyperblobs_model(hypers : HDGMM_Hyperparams):
    sample_shape = Const((hypers.n_hyperblobs,))
    hyperblob_weights = genjax.dirichlet(jnp.repeat(hypers.alpha, hypers.n_hyperblobs)) @ 'hyperblob_weights'
    hyperblob_covs = inverse_wishart(hypers.nu_H, hypers.Psi_H, sample_shape=sample_shape) @ 'hyperblob_covs'
    hyperblob_means = genjax.normal(hypers.mu_H, jnp.sqrt(hypers.sigma_H), sample_shape=sample_shape) @ 'hyperblob_means' 
    hyperblob_trans_vels = discrete_categorical(hypers.discrete_translation, sample_shape=sample_shape) @ 'hyperblob_trans_vels'
    hyperblob_rot_vels = discrete_categorical(hypers.discrete_rotation, sample_shape=sample_shape) @ 'hyperblob_rot_vels'

    return HDGMM_Hyperblobs_State(
        hyperblob_weights = hyperblob_weights,
        hyperblob_means = hyperblob_means,
        hyperblob_covs = hyperblob_covs,
        hyperblob_trans_vels = hyperblob_trans_vels,
        hyperblob_rot_vels = hyperblob_rot_vels,
    )

@gen
def HDGMM_blobs_model(hypers : HDGMM_Hyperparams, hyperblobs_state : HDGMM_Hyperblobs_State):
    sample_shape = Const((hypers.n_blobs,))
    hyperblob_assignments = genjax.categorical(probs = hyperblobs_state.hyperblob_weights, sample_shape=sample_shape) @ 'hyperblob_assignments'
    blob_weights = genjax.dirichlet(jnp.repeat(hypers.beta, hypers.n_blobs)) @ 'blob_weights'
    assigned_hyperblob_per_blob = hyperblobs_state[hyperblob_assignments]
    blob_covs = inverse_wishart(hypers.nu_B, hypers.Psi_B, sample_shape=sample_shape) @ 'blob_covs'
    blob_means = genjax.mv_normal(assigned_hyperblob_per_blob.hyperblob_means, assigned_hyperblob_per_blob.hyperblob_covs) @ 'blob_means'    
    blob_vel_means_ = assigned_hyperblob_per_blob.hyperblob_trans_vels + jnp.einsum('nij,nj->ni', 
                                                                                    assigned_hyperblob_per_blob.hyperblob_rot_vels - jnp.repeat(jnp.eye(blob_means.shape[-1])[None, ...], hypers.n_blobs, axis=0), 
                                                                                    blob_means - assigned_hyperblob_per_blob.hyperblob_means)
    blob_vel_means = genjax.normal(blob_vel_means_, jnp.sqrt(hypers.sigma_V)) @ 'blob_vel_means'
    blob_vel_covs = inverse_wishart(hypers.nu_V, hypers.Psi_V, sample_shape=sample_shape) @ 'blob_vel_covs'
    
    return HDGMM_Blobs_State(
        hyperblob_assignments = hyperblob_assignments,
        blob_weights = blob_weights,
        blob_means = blob_means,
        blob_covs = blob_covs,
        blob_vel_means = blob_vel_means,
        blob_vel_covs = blob_vel_covs,
    )

@gen
def HDGMM_datapoints_model(hypers : HDGMM_Hyperparams, blobs_state : HDGMM_Blobs_State):
    blob_assignments = genjax.categorical(probs = blobs_state.blob_weights, sample_shape=Const((hypers.n_datapoints,))) @ 'blob_assignments'
    assigned_blob_per_datapoint = blobs_state[blob_assignments]
    datapoint_positions = genjax.mv_normal(assigned_blob_per_datapoint.blob_means, assigned_blob_per_datapoint.blob_covs) @ 'datapoint_positions'
    datapoint_vels = genjax.mv_normal(assigned_blob_per_datapoint.blob_vel_means, assigned_blob_per_datapoint.blob_vel_covs) @ 'datapoint_vels'

    return HDGMM_Datapoints_State(
        blob_assignments = blob_assignments,
        datapoint_positions = datapoint_positions,
        datapoint_vels = datapoint_vels,
    )

model_jimportance = jax.jit(HDGMM_model.importance)

@Pytree.dataclass(init=True, has_implicitly_inherited_fields=True)
class hdgmm_TraceWrapper(__hdgmm_TraceWrapper__):
    def __init__(self, trace = None, force_retval=None, force_log_likelihood=None):
        super().__init__(trace, force_retval, force_log_likelihood)
        self.log_likelihood_ = self.extract_log_likelihood(trace) if force_log_likelihood is None else force_log_likelihood
    
    def extract_log_likelihood(self, trace):
        return jnp.array(0)
    
    # Hyperblob level methods
    def hyperblob_means(self, hyperblob_id: int):
        # Check if this is the outlier hyperblob
        if hyperblob_id == self.num_hyperblobs:
            raise ValueError("Cannot get means for outlier hyperblob")
        return np.array(self.retval.hyperblobs_state.hyperblob_means)[hyperblob_id]
    
    def hyperblob_covariance(self, hyperblob_id: int):
        if hyperblob_id == self.num_hyperblobs:
            raise ValueError("Cannot get covariance for outlier hyperblob")
        return np.array(self.retval.hyperblobs_state.hyperblob_covs)[hyperblob_id]
    
    def hyperblob_mixture_weight(self, hyperblob_id: int):
        if hyperblob_id == self.num_hyperblobs:
            # Return the outlier probability
            return float(self.retval.hypers.outlier_prob)
        # For regular hyperblobs, adjust the weight to account for outlier probability
        weights = np.array(self.retval.hyperblobs_state.hyperblob_weights)
        return float(weights[hyperblob_id] * (1 - self.retval.hypers.outlier_prob))
    
    def hyperblob_trans_vel(self, hyperblob_id: int):
        if hyperblob_id == self.num_hyperblobs:
            raise ValueError("Cannot get translation velocity for outlier hyperblob")
        return np.array(self.retval.hyperblobs_state.hyperblob_trans_vels)[hyperblob_id]
    
    def hyperblob_rot_vel(self, hyperblob_id: int):
        if hyperblob_id == self.num_hyperblobs:
            raise ValueError("Cannot get rotation velocity for outlier hyperblob")
        return np.array(self.retval.hyperblobs_state.hyperblob_rot_vels)[hyperblob_id]
    
    def hyperblob_blobs(self, hyperblob_id: int):
        assignments = np.array(self.retval.blobs_state.hyperblob_assignments)
        return np.where(assignments == hyperblob_id)[0].tolist()
    
    def all_hyperblob_means(self):
        return np.array(self.retval.hyperblobs_state.hyperblob_means)
    
    def all_hyperblob_covariances(self):
        return np.array(self.retval.hyperblobs_state.hyperblob_covs)
    
    def all_hyperblob_mixture_weights(self):
        # Include the outlier probability in the weights
        weights = np.array(self.retval.hyperblobs_state.hyperblob_weights)
        adjusted_weights = weights * (1 - self.retval.hypers.outlier_prob)
        return np.append(adjusted_weights, self.retval.hypers.outlier_prob)
    
    def all_hyperblob_trans_vels(self):
        return np.array(self.retval.hyperblobs_state.hyperblob_trans_vels)
    
    def all_hyperblob_rot_vels(self):
        return np.array(self.retval.hyperblobs_state.hyperblob_rot_vels)
    
    # Blob level methods
    def blob_mean(self, blob_id: int):
        if blob_id == self.num_blobs:
            raise ValueError("Cannot get mean for outlier blob")
        return np.array(self.retval.blobs_state.blob_means)[blob_id]
    
    def blob_covariance(self, blob_id: int):
        if blob_id == self.num_blobs:
            raise ValueError("Cannot get covariance for outlier blob")
        return np.array(self.retval.blobs_state.blob_covs)[blob_id]
    
    def blob_velocity_mean(self, blob_id: int):
        if blob_id == self.num_blobs:
            raise ValueError("Cannot get velocity mean for outlier blob")
        return np.array(self.retval.blobs_state.blob_vel_means)[blob_id]
    
    def blob_velocity_covariance(self, blob_id: int):
        if blob_id == self.num_blobs:
            raise ValueError("Cannot get velocity covariance for outlier blob")
        return np.array(self.retval.blobs_state.blob_vel_covs)[blob_id]
    
    def blob_mixture_weight(self, blob_id: int):
        if blob_id == self.num_blobs:
            # Return the outlier probability
            return float(self.retval.hypers.outlier_prob)
        # For regular blobs, adjust the weight to account for outlier probability
        weights = np.array(self.retval.blobs_state.blob_weights)
        return float(weights[blob_id] * (1 - self.retval.hypers.outlier_prob))
    
    def blob_hyperblob(self, blob_id: int):
        if blob_id == self.num_blobs:
            # Outlier blob belongs to outlier hyperblob
            return self.num_hyperblobs
        return int(np.array(self.retval.blobs_state.hyperblob_assignments)[blob_id])
    
    def blob_datapoints(self, blob_id: int):
        return np.array(self.retval.datapoints_state.datapoint_positions)[self.blob_datapoints_indices(blob_id)]
    
    def blob_datapoints_indices(self, blob_id: int):
        blob_assignments = np.array(self.retval.datapoints_state.blob_assignments)
        return np.where(blob_assignments == blob_id)[0]
    
    def all_blob_means(self):
        return np.array(self.retval.blobs_state.blob_means)
    
    def all_blob_covariances(self):
        return np.array(self.retval.blobs_state.blob_covs)
    
    def all_blob_velocity_means(self):
        return np.array(self.retval.blobs_state.blob_vel_means)
    
    def all_blob_velocity_covariances(self):
        return np.array(self.retval.blobs_state.blob_vel_covs)
    
    def all_blob_mixture_weights(self):
        # Include the outlier probability in the weights
        weights = np.array(self.retval.blobs_state.blob_weights)
        adjusted_weights = weights * (1 - self.retval.hypers.outlier_prob)
        return np.append(adjusted_weights, self.retval.hypers.outlier_prob)
    
    def all_blob_hyperblob_assignments(self):
        # Regular blob assignments plus the outlier blob assigned to outlier hyperblob
        assignments = np.array(self.retval.blobs_state.hyperblob_assignments)
        return np.append(assignments, self.num_hyperblobs)
    
    # Datapoint level methods
    def datapoint_position(self, datapoint_id: int):
        return np.array(self.retval.datapoints_state.datapoint_positions)[datapoint_id]
    
    def datapoint_velocity(self, datapoint_id: int):
        return np.array(self.retval.datapoints_state.datapoint_vels)[datapoint_id]
    
    def datapoint_blob(self, datapoint_id: int):
        blob_assignment = int(np.array(self.retval.datapoints_state.blob_assignments)[datapoint_id])
        # The outlier will always be num_blobs, not -1
        return blob_assignment
    
    def datapoint_hyperblob(self, datapoint_id: int):
        blob_id = self.datapoint_blob(datapoint_id)
        return self.blob_hyperblob(blob_id)
    
    def all_datapoint_positions(self):
        return np.array(self.retval.datapoints_state.datapoint_positions)
    
    def all_datapoint_velocities(self):
        return np.array(self.retval.datapoints_state.datapoint_vels)
    
    def all_datapoint_blob_assignments(self):
        # Outliers are already represented as num_blobs
        return np.array(self.retval.datapoints_state.blob_assignments)
    
    def all_datapoint_hyperblob_assignments(self):
        blob_assignments = self.all_datapoint_blob_assignments()
        # For regular blobs, use the hyperblob assignments
        # For outlier blob (num_blobs), use outlier hyperblob (num_hyperblobs)
        result = np.zeros_like(blob_assignments)
        
        # Handle regular datapoints
        regular_mask = blob_assignments < self.num_blobs
        if np.any(regular_mask):
            hyperblob_assignments = np.array(self.retval.blobs_state.hyperblob_assignments)
            result[regular_mask] = hyperblob_assignments[blob_assignments[regular_mask]]
        
        # Handle outlier datapoints
        outlier_mask = blob_assignments == self.num_blobs
        if np.any(outlier_mask):
            result[outlier_mask] = self.num_hyperblobs
            
        return result
    
    # Properties
    @property
    def num_hyperblobs(self):
        return self.retval.hypers.n_hyperblobs
    
    @property
    def num_blobs(self):
        return self.retval.hypers.n_blobs
    
    @property
    def num_datapoints(self):
        return self.retval.hypers.n_datapoints
    
    @property
    def log_likelihood(self):
        log_likelihood_arr = self.log_likelihood_
        if log_likelihood_arr.ndim == 0:
            return log_likelihood_arr
        elif log_likelihood_arr.ndim == 1:
            return jnp.sum(log_likelihood_arr)
        elif log_likelihood_arr.ndim == 2:
            return jnp.sum(log_likelihood_arr, axis=1)
        else:
            raise ValueError("log_likelihood_arr has more than 2 dimensions, not accounted for")

def kabsch_algorithm_local_rotation_and_translation_2d(P, Q):
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.linalg.det(V @ U.T)
    diag = np.eye(2)
    if d < 0:
        diag[1, 1] = -1
    R = V @ diag @ U.T
    t = centroid_Q - centroid_P
    return R, t

def sample_covariance_matrix_numpy_2d(X):
    N = X.shape[0]
    if N <= 1:
        return np.eye(2) * 0.01
    return np.cov(X.T) * (N/(N-1))

def make_hierarchical_kmeans_chm_2d(tracked_points, num_blobs, num_hyperblobs, frame_idx=0, motion_vectors=None):
    positions = tracked_points[frame_idx]
    velocities = motion_vectors[frame_idx]

    features = np.concatenate([positions, velocities], axis=1)

    kmeans_blobs = KMeans(n_clusters=num_blobs, random_state=0)
    blob_labels = kmeans_blobs.fit_predict(features)
    blob_centroids = kmeans_blobs.cluster_centers_[:, :2]

    blob_empirical_covs = []
    blob_mixture_weights = []
    for i in range(num_blobs):
        pts = positions[blob_labels == i]
        if len(pts) > 1:
            cov = np.cov(pts.T)
        else:
            cov = np.eye(2) * 0.01
        blob_empirical_covs.append(cov)
        blob_mixture_weights.append(len(pts) / len(positions))
    blob_empirical_covs = np.stack(blob_empirical_covs)
    blob_mixture_weights = np.array(blob_mixture_weights)

    blob_vel_means = []
    for i in range(num_blobs):
        vels = velocities[blob_labels == i]
        if len(vels) > 0:
            mean_vel = np.mean(vels, axis=0)
        else:
            mean_vel = np.zeros(2)
        blob_vel_means.append(mean_vel)
    blob_vel_means = np.array(blob_vel_means)

    blob_position_motion_matrix = np.concatenate([blob_centroids, blob_vel_means], axis=1)
    kmeans_hyperblobs = KMeans(n_clusters=num_hyperblobs, random_state=0)
    hyperblob_labels = kmeans_hyperblobs.fit_predict(blob_position_motion_matrix)
    hyperblob_centroids = kmeans_hyperblobs.cluster_centers_[:, :2]

    hyperblob_empirical_covs = []
    hyperblob_mixture_weights = []
    for i in range(num_hyperblobs):
        blob_idxs = np.where(hyperblob_labels == i)[0]
        pts = []
        for bidx in blob_idxs:
            pts.append(positions[blob_labels == bidx])
        if pts:
            pts = np.vstack(pts)
            if len(pts) > 1:
                cov = np.cov(pts.T)
            else:
                cov = np.eye(2) * 0.01
            weight = len(pts) / len(positions)
        else:
            cov = np.eye(2) * 0.01
            weight = 0.0
        hyperblob_empirical_covs.append(cov)
        hyperblob_mixture_weights.append(weight)
    hyperblob_empirical_covs = np.stack(hyperblob_empirical_covs)
    hyperblob_mixture_weights = np.array(hyperblob_mixture_weights)

    hyperblob_rot_vels = []
    hyperblob_trans_vels = []
    for i in range(num_hyperblobs):
        blob_idxs = np.where(hyperblob_labels == i)[0]
        point_indices = np.concatenate([np.where(blob_labels == bidx)[0] for bidx in blob_idxs]) if len(blob_idxs) > 0 else np.array([], dtype=int)
        if len(point_indices) > 1:
            points_current = positions[point_indices]
            point_motions = velocities[point_indices]
            points_next = points_current + point_motions
            R, t = kabsch_algorithm_local_rotation_and_translation_2d(points_current, points_next)
        else:
            R = np.eye(2)
            t = np.zeros(2)
        hyperblob_rot_vels.append(R)
        hyperblob_trans_vels.append(t)
    hyperblob_rot_vels = np.array(hyperblob_rot_vels)
    hyperblob_trans_vels = np.array(hyperblob_trans_vels)

    blob_vel_covs = []
    for i in range(num_blobs):
        vels = velocities[blob_labels == i]
        if len(vels) > 1:
            cov = np.cov(vels.T)
        else:
            cov = np.eye(2) * 0.01
        blob_vel_covs.append(cov)
    blob_vel_covs = np.stack(blob_vel_covs)

    blob_expected_vels = []
    for i in range(num_blobs):
        hyperblob_idx = hyperblob_labels[i]
        R = hyperblob_rot_vels[hyperblob_idx]
        t = hyperblob_trans_vels[hyperblob_idx]
        x = blob_centroids[i]
        mu = hyperblob_centroids[hyperblob_idx]
        expected_vel = t + (R - np.eye(2)) @ (x - mu)
        blob_expected_vels.append(expected_vel)
    blob_expected_vels = np.array(blob_expected_vels)

    kmeans_chm = (
        C.n()
        | C['datapoints', 'blob_assignments'].set(jnp.array(blob_labels))
        | C['datapoints', 'datapoint_positions'].set(jnp.array(positions))
        | C['datapoints', 'datapoint_vels'].set(jnp.array(velocities))
        | C['blobs', 'blob_means'].set(jnp.array(blob_centroids))
        | C['blobs', 'blob_covs'].set(jnp.array(truncate_eigenval_ratio(blob_empirical_covs, threshold = 1e6)))
        | C['blobs', 'blob_weights'].set(jnp.array(blob_mixture_weights))
        | C['blobs', 'hyperblob_assignments'].set(jnp.array(hyperblob_labels))
        | C['blobs', 'blob_vel_means'].set(jnp.array(blob_vel_means))
        | C['blobs', 'blob_vel_covs'].set(jnp.array(truncate_eigenval_ratio(blob_vel_covs, threshold = 1e6)))
        | C['hyperblobs', 'hyperblob_means'].set(jnp.array(hyperblob_centroids))
        | C['hyperblobs', 'hyperblob_covs'].set(jnp.array(truncate_eigenval_ratio(hyperblob_empirical_covs, threshold = 1e6)))
        | C['hyperblobs', 'hyperblob_weights'].set(jnp.array(hyperblob_mixture_weights))
        | C['hyperblobs', 'hyperblob_rot_vels'].set(jnp.array(hyperblob_rot_vels))
        | C['hyperblobs', 'hyperblob_trans_vels'].set(jnp.array(hyperblob_trans_vels))
    )
    return kmeans_chm


def normal_inverse_wishart_posterior_by_cluster(X, cluster_ids, mu_k, nu0, Psi0, pseudocounts):
    T, d = X.shape
    K = mu_k.shape[0]

    residuals = X - mu_k[cluster_ids]
    scatter = jnp.einsum('ti,tj->tij', residuals, residuals)

    scatter_flat = scatter.reshape(T, d * d)

    scatter_sum_flat = jax.ops.segment_sum(scatter_flat * pseudocounts[:, None], cluster_ids, num_segments=K)
    S_k = scatter_sum_flat.reshape(K, d, d)

    N_k = jax.ops.segment_sum(jnp.ones(T, dtype=X.dtype) * pseudocounts, cluster_ids, num_segments=K)

    nu_posteriors = nu0 + N_k

    Psi_posteriors = Psi0 + S_k

    Psi_posteriors = truncate_eigenval_ratio(Psi_posteriors, threshold=1e6)

    return nu_posteriors, Psi_posteriors

def normal_normal_posterior_full_cov_batched_flexible_prior(x, cluster_ids, mu0_k, Sigma0, Sigma_k):
    T, d = x.shape
    K = mu0_k.shape[0]

    N_k = jax.ops.segment_sum(jnp.ones(T), cluster_ids, num_segments=K)

    sum_x_k = jax.ops.segment_sum(x, cluster_ids, num_segments=K)

    if Sigma0.ndim == 0:
        Sigma0_k = jnp.broadcast_to(Sigma0 * jnp.eye(d), (K, d, d))
    elif Sigma0.ndim == 2:
        Sigma0_k = jnp.broadcast_to(Sigma0, (K, d, d))
    else:
        Sigma0_k = Sigma0

    Sigma0_inv = jnp.linalg.inv(Sigma0_k)
    Sigma_inv = jnp.linalg.inv(Sigma_k)

    Sigma_n_inv = Sigma0_inv + N_k[:, None, None] * Sigma_inv
    Sigma_n_k = jnp.linalg.inv(Sigma_n_inv)

    term_prior = jnp.einsum('kij,kj->ki', Sigma0_inv, mu0_k)
    term_data  = jnp.einsum('kij,kj->ki', Sigma_inv, sum_x_k)
    mu_n_k = jnp.einsum('kij,kj->ki', Sigma_n_k, term_prior + term_data)

    return mu_n_k, Sigma_n_k

def empty_gibbs(key, hdgmm_state : HDGMM_State):
    return hdgmm_state

def empty_gibbs_weighted(key, hdgmm_state : HDGMM_State, use_weighted_blobs = False):
    return hdgmm_state

def gibbs_blob_weights(key, hdgmm_state : HDGMM_State): 
    posterior_key, _ = jax.random.split(key)

    num_blobs = hdgmm_state.hypers.n_blobs
    prior_beta = hdgmm_state.hypers.beta
    blob_idxs = hdgmm_state.datapoints_state.blob_assignments

    blob_counts = jax.ops.segment_sum(
        jnp.ones_like(blob_idxs),
        blob_idxs,
        num_segments=num_blobs
    )

    new_betas = prior_beta + blob_counts

    new_blob_weights = genjax.dirichlet.sample(posterior_key, new_betas)

    return hdgmm_state.replace({'blobs_state': {'blob_weights': new_blob_weights}})


def gibbs_hyperblob_weights(key, hdgmm_state : HDGMM_State, use_weighted_blobs = False):
    posterior_key, _ = jax.random.split(key)

    num_hyperblobs = hdgmm_state.hypers.n_hyperblobs
    prior_alpha = hdgmm_state.hypers.alpha
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments

    blob_weights = hdgmm_state.blobs_state.blob_weights
    num_blobs = hdgmm_state.hypers.n_blobs

    blob_pseudocounts = jnp.where(use_weighted_blobs, blob_weights * num_blobs, jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))

    hyperblob_counts = jax.ops.segment_sum(
        blob_pseudocounts,
        hyperblob_assignments,
        num_segments=num_hyperblobs
    )

    new_alphas = prior_alpha + hyperblob_counts

    new_hyperblob_weights = genjax.dirichlet.sample(posterior_key, new_alphas)

    return hdgmm_state.replace({'hyperblobs_state': {'hyperblob_weights': new_hyperblob_weights}})


@gen
def blob_datapoint_likelihood_model_no_assignment(blob_state : HDGMM_Blobs_State):
    datapoint_position = genjax.mv_normal(blob_state.blob_means, blob_state.blob_covs) @ 'datapoint_position'
    datapoint_vel = genjax.mv_normal(blob_state.blob_vel_means, blob_state.blob_vel_covs) @ 'datapoint_vel'
    return None

def gibbs_blob_assignments(key, hdgmm_state, position_only=False, velocity_only=False, disable_outlier_prob=False):
    posterior_key, _ = jax.random.split(key)

    hypers = hdgmm_state.hypers
    num_blobs = hypers.n_blobs
    num_datapoints = hypers.n_datapoints
    datapoint_positions = hdgmm_state.datapoints_state.datapoint_positions
    datapoint_vels = hdgmm_state.datapoints_state.datapoint_vels
    blobs_state = hdgmm_state.blobs_state

    gibbs_blob_vel_covs = jnp.where(position_only, 1e14 * blobs_state.blob_vel_covs, blobs_state.blob_vel_covs)
    gibbs_blob_covs = jnp.where(velocity_only, 1e14 * blobs_state.blob_covs, blobs_state.blob_covs)
    blobs_state = blobs_state.replace({'blob_vel_covs': gibbs_blob_vel_covs})
    blobs_state = blobs_state.replace({'blob_covs': gibbs_blob_covs})

    blob_weights = blobs_state.blob_weights
    outlier_prob = hypers.outlier_prob

    outlier_prob = jnp.where(disable_outlier_prob, 0.0, outlier_prob)
    extended_weights = jnp.concatenate([blob_weights, jnp.array([outlier_prob])])
    normalized_weights = extended_weights / jnp.sum(extended_weights)
    log_mixture_weights = jnp.log(normalized_weights)

    def compute_local_density(point_idx):
        chm = (C["datapoint_position"].set(datapoint_positions[point_idx]) |
               C["datapoint_vel"].set(datapoint_vels[point_idx]))

        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_no_assignment.assess(
                chm, (blobs_state[i],)
            )[0]
        )(jnp.arange(num_blobs))

        v = datapoint_vels[point_idx]
        speed = jnp.linalg.norm(v)

        alpha = hypers.outlier_velocity_gamma_shape
        beta = hypers.outlier_velocity_gamma_rate

        log_gamma_vel = (
            (alpha - 1) * jnp.log(speed + 1e-8)
            - beta * speed
            - alpha * jnp.log(1. / beta)
            - jax.lax.lgamma(alpha)
        )

        log_gamma_vel = jnp.where(disable_outlier_prob, 0.0, log_gamma_vel)

        raw_log_liks = jnp.concatenate([log_liks, jnp.array([log_gamma_vel])])
        return raw_log_liks + log_mixture_weights

    all_logprobs = jax.vmap(compute_local_density)(jnp.arange(num_datapoints))
    updated_assignments = genjax.categorical.sample(posterior_key, logits=all_logprobs)

    return hdgmm_state.replace({
        'datapoints_state': {
            'blob_assignments': updated_assignments
        }
    })

def gibbs_blob_assignments_batched(key, hdgmm_state, position_only=False, velocity_only=False, disable_outlier_prob=False):
    posterior_key, _ = jax.random.split(key)

    batch_size = 975

    hypers = hdgmm_state.hypers
    num_blobs = hypers.n_blobs
    num_datapoints = hypers.n_datapoints
    datapoint_positions = hdgmm_state.datapoints_state.datapoint_positions
    datapoint_vels = hdgmm_state.datapoints_state.datapoint_vels
    blobs_state = hdgmm_state.blobs_state

    # disable velocity updates if position_only is True
    gibbs_blob_vel_covs = jnp.where(position_only, 1e14 * blobs_state.blob_vel_covs, blobs_state.blob_vel_covs)
    # disable position updates if velocity_only is True
    gibbs_blob_covs = jnp.where(velocity_only, 1e14 * blobs_state.blob_covs, blobs_state.blob_covs)
    blobs_state = blobs_state.replace({'blob_vel_covs': gibbs_blob_vel_covs})
    blobs_state = blobs_state.replace({'blob_covs': gibbs_blob_covs})

    blob_weights = blobs_state.blob_weights  # [L]
    outlier_prob = hypers.outlier_prob  # scalar

    # if potions only or velocity only is true, set outlier probability to 0 using jnp where
    outlier_prob = jnp.where(disable_outlier_prob, 0.0, outlier_prob)
    # Normalize weights
    extended_weights = jnp.concatenate([blob_weights, jnp.array([outlier_prob])])  # [L+1]
    normalized_weights = extended_weights / jnp.sum(extended_weights)  # [L+1]
    log_mixture_weights = jnp.log(normalized_weights)

    def compute_local_density(point_idx):
        chm = (C["datapoint_position"].set(datapoint_positions[point_idx]) |
               C["datapoint_vel"].set(datapoint_vels[point_idx]))

        # Real blobs
        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_no_assignment.assess(
                chm, (blobs_state[i],)
            )[0]
        )(jnp.arange(num_blobs))  # [L]

        # --- Outlier: Gamma log-pdf over ||v||
        v = datapoint_vels[point_idx]
        speed = jnp.linalg.norm(v)

        alpha = hypers.outlier_velocity_gamma_shape
        beta = hypers.outlier_velocity_gamma_rate

        log_gamma_vel = (
            (alpha - 1) * jnp.log(speed + 1e-8)
            - beta * speed
            - alpha * jnp.log(1. / beta)
            - jax.lax.lgamma(alpha)
        )

        # jnp where outlier to be 0 if disable_outlier_prob is true
        log_gamma_vel = jnp.where(disable_outlier_prob, 0.0, log_gamma_vel)

        # Combine
        raw_log_liks = jnp.concatenate([log_liks, jnp.array([log_gamma_vel])])  # [L+1]
        return raw_log_liks + log_mixture_weights

    # Compute logprobs in batches using scan
    def batch_compute_logprobs(carry, batch_idx):
        batch_start = batch_idx * batch_size
        batch_indices = jnp.arange(batch_size) + batch_start
        batch_logprobs = jax.vmap(compute_local_density)(batch_indices)
        return carry, batch_logprobs
    
    # Calculate number of full batches
    num_full_batches = num_datapoints // batch_size
    
    # Run scan over full batches
    _, batched_logprobs = jax.lax.scan(
        batch_compute_logprobs,
        None,  # carry is unused
        jnp.arange(num_full_batches)
    )
    
    # Reshape to get the full array of logprobs
    all_logprobs = batched_logprobs.reshape(num_full_batches * batch_size, -1)
    
    # Sample from the computed logprobs
    updated_assignments = genjax.categorical.sample(posterior_key, logits=all_logprobs)

    return hdgmm_state.replace({
        'datapoints_state': {
            'blob_assignments': updated_assignments
        }
    })

# Likelihood model for hyperblob blobs used in Gibbs #4
@gen
def hyperblob_blob_likelihood_model(hyperblobs_state : HDGMM_Hyperblobs_State, hypers : HDGMM_Hyperparams):
    hyperblob_idx = genjax.categorical(probs = hyperblobs_state.hyperblob_weights) @ 'hyperblob_assignment'
    hyperblob_state = hyperblobs_state[hyperblob_idx]
    blob_mean = genjax.mv_normal(hyperblob_state.hyperblob_means, hyperblob_state.hyperblob_covs) @ 'blob_mean'
    blob_vel_mean_ = hyperblob_state.hyperblob_trans_vels + jnp.matmul(hyperblob_state.hyperblob_rot_vels - jnp.eye(blob_mean.shape[-1]), blob_mean - hyperblob_state.hyperblob_means)
    blob_vel_mean = genjax.normal(blob_vel_mean_, jnp.sqrt(hypers.sigma_V)) @ 'blob_vel'
    return None

# Gibbs #4: Update hyperblob assignments 
def gibbs_hyperblob_assignments(key, hdgmm_state : HDGMM_State):
    posterior_key, _ = jax.random.split(key)

    # get the data and parameters from the trace
    num_hyperblobs = hdgmm_state.hypers.n_hyperblobs
    num_blobs = hdgmm_state.hypers.n_blobs
    blob_means = hdgmm_state.blobs_state.blob_means
    blob_vel_means = hdgmm_state.blobs_state.blob_vel_means
    hyperblobs_state = hdgmm_state.hyperblobs_state
    hypers = hdgmm_state.hypers

    def compute_local_density(blob_idx):
        chm = (C["blob_mean"].set(blob_means[blob_idx]) 
              | C["blob_vel"].set(blob_vel_means[blob_idx]))

        return jax.vmap(
            lambda i: hyperblob_blob_likelihood_model.assess(
                chm.at["hyperblob_assignment"].set(i), (hyperblobs_state, hypers)
            )[0]
        )(jnp.arange(num_hyperblobs))

    local_densities = jax.vmap(compute_local_density)(jnp.arange(num_blobs))

    # Sample new assignments
    updated_hyperblob_assignments = genjax.categorical.sample(posterior_key, logits = local_densities)

    return hdgmm_state.replace({'blobs_state': {'hyperblob_assignments': updated_hyperblob_assignments}})

# Gibbs #5: Update hyperblob covariances 
def gibbs_hyperblob_covs(key, hdgmm_state, use_weighted_blobs = False):
    posterior_key, _ = jax.random.split(key)

    blob_means = hdgmm_state.blobs_state.blob_means
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments
    assumed_known_means = hdgmm_state.hyperblobs_state.hyperblob_means
    prior_nu_H = hdgmm_state.hypers.nu_H
    prior_Psi_H = hdgmm_state.hypers.Psi_H

    n_hyperblobs = hdgmm_state.hypers.n_hyperblobs

    # for weighted sums (i.e. weighted by number of datapoints assigned to each blob)
    blob_weights = hdgmm_state.blobs_state.blob_weights
    num_blobs = hdgmm_state.hypers.n_blobs

    blob_pseudocounts = jnp.where(use_weighted_blobs, blob_weights * num_blobs, jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))

    # --- 1. Get number of blobs assigned to each hyperblob ---
    N_k = jax.ops.segment_sum(
        blob_pseudocounts, 
        hyperblob_assignments,
        num_segments=n_hyperblobs
    )  # [K]

    # --- 2. Compute normal NIW posterior ---
    nu_posteriors, Psi_posteriors = normal_inverse_wishart_posterior_by_cluster(
        blob_means, hyperblob_assignments, assumed_known_means, prior_nu_H, prior_Psi_H, blob_pseudocounts
    )

    # --- 3. Sample new covariances from posterior ---
    updated_hyperblob_covs = inverse_wishart.sample(posterior_key, nu_posteriors, Psi_posteriors)

    # --- 4. Mask: retain previous covariances if N_k <= 3 (3 min points per hyperblob for scatter matrix to be valid in NIW --> full rank) ---
    current_hyperblob_covs = hdgmm_state.hyperblobs_state.hyperblob_covs  # [K, d, d]
    mask = (N_k >= 3)[:, None, None]  # [K,1,1] to broadcast over 3x3 matrices

    final_hyperblob_covs = jnp.where(mask, updated_hyperblob_covs, current_hyperblob_covs)

    return hdgmm_state.replace({'hyperblobs_state': {'hyperblob_covs': final_hyperblob_covs}})


# Gibbs #6: Update blob covariances 
def gibbs_blob_covs(key, hdgmm_state):
    posterior_key, _ = jax.random.split(key)

    datapoint_positions = hdgmm_state.datapoints_state.datapoint_positions
    blob_assignments = hdgmm_state.datapoints_state.blob_assignments
    assumed_known_means = hdgmm_state.blobs_state.blob_means
    prior_nu_B = hdgmm_state.hypers.nu_B
    prior_Psi_B = hdgmm_state.hypers.Psi_B

    n_blobs = hdgmm_state.hypers.n_blobs

    datapoint_pseudocounts = jnp.ones(datapoint_positions.shape[0], dtype=datapoint_positions.dtype)

    # --- 1. Get number of datapoints assigned to each blob ---
    N_l = jax.ops.segment_sum(
        datapoint_pseudocounts, 
        blob_assignments,
        num_segments=n_blobs
    )  # [L]


    # --- 2. Compute normal NIW posterior ---
    nu_posteriors, Psi_posteriors = normal_inverse_wishart_posterior_by_cluster(
        datapoint_positions, blob_assignments, assumed_known_means, prior_nu_B, prior_Psi_B, datapoint_pseudocounts
    )

    # --- 3. Sample new covariances from posterior ---
    updated_blob_covs = inverse_wishart.sample(posterior_key, nu_posteriors, Psi_posteriors)

    # --- 4. Mask: retain previous covariances if N_l <= 3 (3 min points per blob for scatter matrix to be valid in NIW --> full rank)---
    current_blob_covs = hdgmm_state.blobs_state.blob_covs  # [L, d, d]
    mask = (N_l >= 3)[:, None, None]  # [L,1,1] to broadcast over 3x3 matrices

    final_blob_covs = jnp.where(mask, updated_blob_covs, current_blob_covs)

    return hdgmm_state.replace({'blobs_state': {'blob_covs': final_blob_covs}})


# Gibbs #7: Update blob velocity covariances 
def gibbs_blob_vel_covs(key, hdgmm_state):
    posterior_key, _ = jax.random.split(key)

    datapoint_vels = hdgmm_state.datapoints_state.datapoint_vels
    blob_assignments = hdgmm_state.datapoints_state.blob_assignments
    assumed_known_vel_means = hdgmm_state.blobs_state.blob_vel_means
    prior_nu_V = hdgmm_state.hypers.nu_V
    prior_Psi_V = hdgmm_state.hypers.Psi_V

    n_blobs = hdgmm_state.hypers.n_blobs

    datapoint_pseudocounts = jnp.ones(datapoint_vels.shape[0], dtype=datapoint_vels.dtype)

    # --- 1. Get number of datapoints assigned to each blob ---
    N_l = jax.ops.segment_sum(
        datapoint_pseudocounts, 
        blob_assignments,
        num_segments=n_blobs
    )  # [L]

    # --- 2. Compute normal NIW posterior ---
    nu_posteriors, Psi_posteriors = normal_inverse_wishart_posterior_by_cluster(
        datapoint_vels, blob_assignments, assumed_known_vel_means, prior_nu_V, prior_Psi_V, datapoint_pseudocounts
    )

    # --- 3. Sample new covariances from posterior ---
    updated_blob_vel_covs = inverse_wishart.sample(posterior_key, nu_posteriors, Psi_posteriors)

    # --- 4. Mask: retain previous covariances if N_l <= 3 (3 min points per blob for scatter matrix to be valid in NIW --> full rank)---
    current_blob_vel_covs = hdgmm_state.blobs_state.blob_vel_covs  # [L, d, d]
    mask = (N_l >= 3)[:, None, None]  # [L,1,1] to broadcast over 3x3 matrices

    final_blob_vel_covs = jnp.where(mask, updated_blob_vel_covs, current_blob_vel_covs)

    return hdgmm_state.replace({'blobs_state': {'blob_vel_covs': final_blob_vel_covs}})

# Gibbs #8: Update blob velocity means 
def gibbs_blob_vel_means(key, hdgmm_state):
    posterior_key, _ = jax.random.split(key)

    datapoint_vels = hdgmm_state.datapoints_state.datapoint_vels
    blob_assignments = hdgmm_state.datapoints_state.blob_assignments
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments
    blob_means = hdgmm_state.blobs_state.blob_means
    assigned_hyperblob_per_blob = hdgmm_state.hyperblobs_state[hyperblob_assignments]
    likelihood_blob_vel_covs = hdgmm_state.blobs_state.blob_vel_covs
    current_blob_vel_means = hdgmm_state.blobs_state.blob_vel_means

    d = blob_means.shape[-1]

    prior_blob_vel_means_ = assigned_hyperblob_per_blob.hyperblob_trans_vels + jnp.einsum('nij,nj->ni', assigned_hyperblob_per_blob.hyperblob_rot_vels - jnp.repeat(jnp.eye(d)[None, ...], hdgmm_state.hypers.n_blobs, axis=0), blob_means - assigned_hyperblob_per_blob.hyperblob_means)

    prior_variance = hdgmm_state.hypers.sigma_V

    # Count datapoints per blob
    n_blobs = hdgmm_state.hypers.n_blobs
    N_l = jax.ops.segment_sum(
        jnp.ones(datapoint_vels.shape[0], dtype=datapoint_vels.dtype), 
        blob_assignments,
        num_segments=n_blobs
    )  # [L]
    
    # Get posterior parameters
    posterior_mus, posterior_covs = normal_normal_posterior_full_cov_batched_flexible_prior(datapoint_vels, blob_assignments, prior_blob_vel_means_, prior_variance, likelihood_blob_vel_covs)
    
    # For blobs with no datapoints, set velocity to zero
    has_points = N_l > 0
    
    # Sample new blob velocity means for blobs with datapoints
    sampled_vel_means = genjax.mv_normal.sample(posterior_key, posterior_mus, posterior_covs)
    
    # Set velocity to zero for empty blobs
    zero_velocities = jnp.zeros_like(posterior_mus)
    posterior_blob_vel_means = jnp.where(has_points[:, None], sampled_vel_means, zero_velocities)
    # posterior_blob_vel_means = jnp.where(has_points[:, None], sampled_vel_means, current_blob_vel_means)
    return hdgmm_state.replace({'blobs_state': {'blob_vel_means': posterior_blob_vel_means}})

# Gibbs #9: Update hyperblob means 
def gibbs_hyperblob_means(key, hdgmm_state, use_weighted_blobs=False):
    posterior_key, _ = jax.random.split(key)
    
    # Extract data
    blob_means = hdgmm_state.blobs_state.blob_means              # [L, d]
    blob_vel_means = hdgmm_state.blobs_state.blob_vel_means      # [L, d]
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments  # [L]
    blob_weights = hdgmm_state.blobs_state.blob_weights          # [L]

    hyperblob_covs = hdgmm_state.hyperblobs_state.hyperblob_covs          # [K, d, d]
    rot_vels = hdgmm_state.hyperblobs_state.hyperblob_rot_vels            # [K, d, d]
    trans_vels = hdgmm_state.hyperblobs_state.hyperblob_trans_vels        # [K, d]

    mu0 = hdgmm_state.hypers.mu_H
    sigmaH = hdgmm_state.hypers.sigma_H
    sigmaV = hdgmm_state.hypers.sigma_V
    K = hdgmm_state.hypers.n_hyperblobs
    num_blobs = hdgmm_state.hypers.n_blobs

    L = blob_means.shape[0]
    d = blob_means.shape[1]

    # Create pseudocounts based on blob weights if use_weighted_blobs is True
    blob_pseudocounts = jnp.where(use_weighted_blobs, 
                                 blob_weights * num_blobs, 
                                 jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))

    # Segment sums with pseudocounts
    N_k = jax.ops.segment_sum(blob_pseudocounts, hyperblob_assignments, num_segments=K) # [K] weighted num blobs in each hyperblob
    S_k = jax.ops.segment_sum(blob_means * blob_pseudocounts[:, None], hyperblob_assignments, num_segments=K) # [K, d] weighted sum of blob means

    hyperblob_covs_inv = jnp.linalg.inv(hyperblob_covs) # [K, d, d] inverse of hyperblob covariances

    # Compute transformed velocity means
    A_k = jnp.eye(d) - rot_vels  # [K, d, d] affine rotation matrix for each hyperblob
    b_l = trans_vels[hyperblob_assignments] - jnp.einsum("lij,lj->li", A_k[hyperblob_assignments], blob_means) # [L, d] affine translation vector for each blob
    v_l = blob_vel_means
    residuals = v_l - b_l  # [L, d]

    # Segment sum residuals with pseudocounts
    residual_sum = jax.ops.segment_sum(residuals * blob_pseudocounts[:, None], hyperblob_assignments, num_segments=K)  # [K, d]
    # Then apply A_k^T
    rhs_affine = jnp.einsum("kji,kj->ki", A_k, residual_sum)  # [K, d]
    lhs_affine = jnp.einsum('kpi,kpj->kij', A_k, A_k)  # [K, d, d]

    # Invert prior variance
    prior_precision = jnp.eye(d) / sigmaH  # [d,d] prior precision matrix

    # Posterior precision and mean
    posterior_precision = (
        prior_precision[None, ...] + # [1, d, d]
        N_k[:, None, None] * (hyperblob_covs_inv + lhs_affine / sigmaV) # [K, d, d]
    )  # [K, d, d]

    # Compute weighted mean with pseudocounts
    weighted_mean = (
        (mu0/sigmaH)[None, :] + # [1, d]
        jnp.einsum('kij,kj->ki', hyperblob_covs_inv, S_k) + # [K, d]
        rhs_affine / sigmaV # [K, d]
    )

    posterior_covs = jnp.linalg.inv(posterior_precision)
    posterior_means = jnp.einsum("kij,kj->ki", posterior_covs, weighted_mean)

    # Sample new means
    new_means = genjax.mv_normal.sample(posterior_key, posterior_means, posterior_covs)

    return hdgmm_state.replace({'hyperblobs_state': {'hyperblob_means': new_means}})

# Gibbs #10: Update blob means 
def gibbs_blob_means(key, hdgmm_state):
    posterior_key, _ = jax.random.split(key)

    # Data
    datapoint_positions = hdgmm_state.datapoints_state.datapoint_positions  # [N, d]
    blob_assignments = hdgmm_state.datapoints_state.blob_assignments        # [N]
    blob_vel_means = hdgmm_state.blobs_state.blob_vel_means                 # [L, d]
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments   # [L]
    blob_covs = hdgmm_state.blobs_state.blob_covs                           # [L, d, d]

    # Hyperblob info
    mu_H = hdgmm_state.hyperblobs_state.hyperblob_means                     # [K, d]
    trans_vels = hdgmm_state.hyperblobs_state.hyperblob_trans_vels         # [K, d]
    rot_vels = hdgmm_state.hyperblobs_state.hyperblob_rot_vels             # [K, d, d]

    # Hyperparams
    sigmaV = hdgmm_state.hypers.sigma_V
    L = hdgmm_state.hypers.n_blobs
    d = hdgmm_state.blobs_state.blob_means.shape[-1] # [L,3] --> [3]

    # Assign hyperblob means to blobs
    muH_l = mu_H[hyperblob_assignments]  # [L, d]

    # Prior precision
    hyperblob_cov_inv = jnp.linalg.inv(hdgmm_state.hyperblobs_state.hyperblob_covs[hyperblob_assignments])  # [L, d, d]

    # Data terms (from datapoints)
    N_l = jax.ops.segment_sum(jnp.ones(datapoint_positions.shape[0]), blob_assignments, num_segments=L) # [L]
    S_l = jax.ops.segment_sum(datapoint_positions, blob_assignments, num_segments=L) # [L, d]

    # Likelihood precision (from blob covariances)
    blob_cov_inv = jnp.linalg.inv(blob_covs)  # [L, d, d]

    # Velocity affine likelihood
    A_l = rot_vels[hyperblob_assignments] - jnp.eye(d)  # [L, d, d]
    b_l = trans_vels[hyperblob_assignments] - jnp.einsum("lij,lj->li", A_l, muH_l)  # [L, d]
    v_l = blob_vel_means # [L, d]
    residuals = v_l - b_l  # [L, d]

    lhs_affine = jnp.einsum("lij,lik->ljk", A_l, A_l)    # [L, d, d]
    rhs_affine = jnp.einsum("lij,li->lj", A_l, residuals)  # [L, d]

    # Combine into posterior precision and weighted mean
    P_post = (
        hyperblob_cov_inv +  # prior from hyperblob covariance [L, d, d]
        blob_cov_inv * N_l[:, None, None] +  # likelihood from datapoints [L, d, d]
        lhs_affine / sigmaV  # velocity model [L, d, d]
    )  # [L, d, d]

    weighted_mean = (
        jnp.einsum("lij,lj->li", hyperblob_cov_inv, muH_l) +  # prior with hyperblob covariance
        jnp.einsum("lij,lj->li", blob_cov_inv, S_l) +         # datapoint positions
        rhs_affine / sigmaV                                   # velocity affine
    )  # [L, d]

    # Solve for posterior mean
    cov_post = jnp.linalg.inv(P_post)  # [L, d, d]
    mean_post = jnp.einsum("lij,lj->li", cov_post, weighted_mean)  # [L, d]

    # Sample new blob means
    new_blob_means = genjax.mv_normal.sample(posterior_key, mean_post, cov_post)

    return hdgmm_state.replace({'blobs_state': {'blob_means': new_blob_means}})


# Gibbs #11: Update hyperblob rotation velocities
def gibbs_hyperblob_rot(key, hdgmm_state: HDGMM_State, use_weighted_blobs=False):
    posterior_key, _ = jax.random.split(key)

    hypers = hdgmm_state.hypers
    hyperblobs_state = hdgmm_state.hyperblobs_state
    blob_means = hdgmm_state.blobs_state.blob_means
    blob_vel_means = hdgmm_state.blobs_state.blob_vel_means
    blob_weights = hdgmm_state.blobs_state.blob_weights
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments
    hyperblob_trans_vels = hyperblobs_state.hyperblob_trans_vels

    d = blob_means.shape[-1]

    n_hyperblobs = hypers.n_hyperblobs
    num_blobs = hypers.n_blobs
    rot_support = hypers.discrete_rotation.support  # [M, d, d]
    rot_logprobs = hypers.discrete_rotation.logprobs  # [M] - prior logprobs
    sigma_V = hypers.sigma_V
    
    # Create pseudocounts based on blob weights if use_weighted_blobs is True
    blob_pseudocounts = jnp.where(use_weighted_blobs, 
                                 blob_weights * num_blobs, 
                                 jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))
    
    # Create a mask for each hyperblob indicating which blobs belong to it
    hyperblob_mask = (hyperblob_assignments[None, :] == jnp.arange(n_hyperblobs)[:, None])  # [K, n_blobs]

    # First compute the (R_k - I) term for each rotation candidate
    # Shape: [M, d, d]
    rot_minus_eye = rot_support - jnp.eye(d)
    
    # Pre-compute position differences (μ_ℓ - μ_k) for all blobs
    hyperblob_means_per_blob = hyperblobs_state.hyperblob_means[hyperblob_assignments]  # [n_blobs, d]
    position_difference = blob_means - hyperblob_means_per_blob  # [n_blobs, d]
    
    # Pre-compute translation component for each blob
    trans_per_blob = hyperblob_trans_vels[hyperblob_assignments]  # [n_blobs, d]
    
    # Function to compute log probabilities for all rotation candidates for a single hyperblob
    def compute_rotation_logprobs_for_hyperblob(hk, mask):
        # Reference all blobs (we'll use the mask later)
        all_blob_velocities = blob_vel_means  # [n_blobs, d]
        all_position_differences = position_difference  # [n_blobs, d]
        all_trans_components = trans_per_blob  # [n_blobs, d]
        
        # For each rotation candidate and each blob, compute the expected velocity
        # expected velocity = t_k + (R_k - I)(μ_ℓ - μ_k)
    
        
        # Now compute the rotation effect for each candidate and each blob
        # Shape: [M, n_blobs, d]
        rotation_effect = jnp.einsum('mij,nj->mni', rot_minus_eye, all_position_differences)
        
        # Compute full expected velocity for each candidate and each blob
        # Shape: [M, n_blobs, d]
        expected_velocities = all_trans_components[None, :, :] + rotation_effect
        
        # Compute log probability: log N(v_ℓ | expected_velocity, σ²I)
        # Shape: [M, n_blobs]
        squared_diff = jnp.sum((all_blob_velocities[None, :, :] - expected_velocities) ** 2, axis=-1)
        log_liks = (-0.5 * squared_diff / sigma_V) - (1.5 * jnp.log(2 * jnp.pi * sigma_V))
        
        # Apply the mask to only include blobs belonging to this hyperblob
        # and weight by pseudocounts (adding log of pseudocounts since we're in log space)
        # Shape: [M, n_blobs]
        masked_log_liks = jnp.where(mask[None, :], log_liks + jnp.log(blob_pseudocounts[None, :]), 0.0)
        
        # Sum log likelihoods across all blobs for each rotation candidate
        # Shape: [M]
        sum_log_liks = jnp.sum(masked_log_liks, axis=1)
        
        # Add the prior log probability for each rotation candidate
        # Shape: [M]
        return sum_log_liks + rot_logprobs
    
    # Compute log probabilities for all hyperblobs and all rotation candidates
    logprobs_per_hk = jax.vmap(compute_rotation_logprobs_for_hyperblob)(
        jnp.arange(n_hyperblobs),
        hyperblob_mask
    )  # [K, M]
    
    # Sample from the computed distributions
    sampled_indices = genjax.categorical.sample(posterior_key, logits=logprobs_per_hk)  # [K]
    updated_rotations = rot_support[sampled_indices]  # [K, d, d]

    return hdgmm_state.replace({'hyperblobs_state': {'hyperblob_rot_vels': updated_rotations}})

# Gibbs #12: Update hyperblob translation velocities 
def gibbs_hyperblob_trans(key, hdgmm_state: HDGMM_State, use_weighted_blobs=False):
    posterior_key, _ = jax.random.split(key)

    hypers = hdgmm_state.hypers
    hyperblobs_state = hdgmm_state.hyperblobs_state
    blob_means = hdgmm_state.blobs_state.blob_means
    blob_vel_means = hdgmm_state.blobs_state.blob_vel_means
    blob_weights = hdgmm_state.blobs_state.blob_weights
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments
    hyperblob_rot_vels = hyperblobs_state.hyperblob_rot_vels

    d = blob_means.shape[-1]

    n_hyperblobs = hypers.n_hyperblobs
    n_blobs = hypers.n_blobs
    trans_support = hypers.discrete_translation.support  # [M, d]
    trans_logprobs = hypers.discrete_translation.logprobs  # [M] - prior logprobs
    sigma_V = hypers.sigma_V

    # Create pseudocounts based on blob weights if use_weighted_blobs is True
    blob_pseudocounts = jnp.where(use_weighted_blobs, 
                                 blob_weights * n_blobs, 
                                 jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))
    
    # Create a mask for each hyperblob indicating which blobs belong to it
    hyperblob_mask = (hyperblob_assignments[None, :] == jnp.arange(n_hyperblobs)[:, None])  # [K, n_blobs]
    
    # Pre-compute the rotation effect term (R_k - I)(μ_ℓ - μ_k) for all blobs
    eye_matrices = jnp.repeat(jnp.eye(d)[None, ...], n_blobs, axis=0)
    
    # Use hyperblob assignments to get the right hyperblob parameters for each blob
    rot_matrices_per_blob = hyperblob_rot_vels[hyperblob_assignments]  # [n_blobs, d, d]
    hyperblob_means_per_blob = hyperblobs_state.hyperblob_means[hyperblob_assignments]  # [n_blobs, d]
    
    # (R_k - I) term for each blob
    rot_difference = rot_matrices_per_blob - eye_matrices  # [n_blobs, d, d]
    
    # (μ_ℓ - μ_k) term for each blob
    position_difference = blob_means - hyperblob_means_per_blob  # [n_blobs, d]
    
    # Pre-compute (R_k - I)(μ_ℓ - μ_k) for each blob
    rotation_effect = jnp.einsum('nij,nj->ni', rot_difference, position_difference)  # [n_blobs, d]
    
    # Function to compute log probabilities for all translation candidates for a single hyperblob
    def compute_translation_logprobs_for_hyperblob(hk, mask):
        # Get all blobs (we'll use the mask later)
        all_blob_velocities = blob_vel_means  # [n_blobs, d]
        all_rotation_effects = rotation_effect  # [n_blobs, d]
        
        # For each candidate, compute expected velocity for each blob
        # expected velocity = t_k + (R_k - I)(μ_ℓ - μ_k)
        # Shape: [M, n_blobs, d]
        expected_velocities = trans_support[:, None, :] + all_rotation_effects[None, :, :]
        
        # Compute log probability: log N(v_ℓ | expected_velocity, σ²I)
        # Shape: [M, n_blobs]
        squared_diff = jnp.sum((all_blob_velocities[None, :, :] - expected_velocities) ** 2, axis=-1)
        log_liks = (-0.5 * squared_diff / sigma_V) - (1.5 * jnp.log(2 * jnp.pi * sigma_V))
        
        # Apply the mask to only include blobs belonging to this hyperblob
        # Shape: [M, n_blobs]
        masked_log_liks = jnp.where(mask[None, :], log_liks + jnp.log(blob_pseudocounts[None, :]), 0.0)
        
        # Sum log likelihoods across all blobs for each translation candidate
        # Shape: [M]
        sum_log_liks = jnp.sum(masked_log_liks, axis=1)
        
        # Add the prior log probability for each translation candidate
        # Shape: [M]
        return sum_log_liks + trans_logprobs
    
    
    # Compute log probabilities for all hyperblobs and all translation candidates
    logprobs_per_hk = jax.vmap(compute_translation_logprobs_for_hyperblob)(
        jnp.arange(n_hyperblobs),
        hyperblob_mask
    )  # [K, M]
    
    # Sample from the computed distributions
    sampled_indices = genjax.categorical.sample(posterior_key, logits=logprobs_per_hk)  # [K]
    updated_translations = trans_support[sampled_indices]  # [K, d]

    return hdgmm_state.replace({'hyperblobs_state': {'hyperblob_trans_vels': updated_translations}})


def gibbs_transitioned_blob_means(key, hdgmm_state, prior_means, prior_covs):
    posterior_key, _ = jax.random.split(key)

    # Data
    datapoint_positions = hdgmm_state.datapoints_state.datapoint_positions  # [N, d]
    blob_assignments = hdgmm_state.datapoints_state.blob_assignments        # [N]
    blob_vel_means = hdgmm_state.blobs_state.blob_vel_means                 # [L, d]
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments   # [L]
    blob_covs = hdgmm_state.blobs_state.blob_covs                           # [L, d, d]

    # Hyperblob info
    mu_H = hdgmm_state.hyperblobs_state.hyperblob_means                     # [K, d]
    trans_vels = hdgmm_state.hyperblobs_state.hyperblob_trans_vels         # [K, d]
    rot_vels = hdgmm_state.hyperblobs_state.hyperblob_rot_vels             # [K, d, d]

    # Hyperparams
    sigmaV = hdgmm_state.hypers.sigma_V
    L = hdgmm_state.hypers.n_blobs
    d = hdgmm_state.blobs_state.blob_means.shape[-1] # [L,3] --> [3]

    # Assign hyperblob means to blobs
    muH_l = prior_means  # [L, d]

    # Prior precision
    hyperblob_cov_inv = jnp.linalg.inv(prior_covs)  # [L, d, d]

    # Data terms (from datapoints)
    N_l = jax.ops.segment_sum(jnp.ones(datapoint_positions.shape[0]), blob_assignments, num_segments=L) # [L]
    S_l = jax.ops.segment_sum(datapoint_positions, blob_assignments, num_segments=L) # [L, d]

    # Likelihood precision (from blob covariances)
    blob_cov_inv = jnp.linalg.inv(blob_covs)  # [L, d, d]

    # Velocity affine likelihood
    A_l = rot_vels[hyperblob_assignments] - jnp.eye(d)  # [L, d, d]
    b_l = trans_vels[hyperblob_assignments] - jnp.einsum("lij,lj->li", A_l, muH_l)  # [L, d]
    v_l = blob_vel_means # [L, d]
    residuals = v_l - b_l  # [L, d]

    lhs_affine = jnp.einsum("lij,lik->ljk", A_l, A_l)    # [L, d, d]
    rhs_affine = jnp.einsum("lij,li->lj", A_l, residuals)  # [L, d]

    # Handle empty blobs by creating a mask
    has_points = N_l > 0
    # Use the mask to zero out the contribution from empty blobs
    N_l_safe = jnp.where(has_points, N_l, 0.0)

    # Combine into posterior precision and weighted mean
    P_post = (
        hyperblob_cov_inv +  # prior from hyperblob covariance [L, d, d]
        blob_cov_inv * N_l_safe[:, None, None] +  # likelihood from datapoints [L, d, d]
        lhs_affine / sigmaV  # velocity model [L, d, d]
    )  # [L, d, d]

    weighted_mean = (
        jnp.einsum("lij,lj->li", hyperblob_cov_inv, muH_l) +  # prior with hyperblob covariance
        jnp.einsum("lij,lj->li", blob_cov_inv, S_l) +         # datapoint positions
        rhs_affine / sigmaV                                   # velocity affine
    )  # [L, d]

    # Solve for posterior mean
    cov_post = jnp.linalg.inv(P_post)  # [L, d, d]
    mean_post = jnp.einsum("lij,lj->li", cov_post, weighted_mean)  # [L, d]

    # For empty blobs, fall back to the prior mean and covariance
    mean_post = jnp.where(has_points[:, None], mean_post, muH_l)
    # For empty blobs, use the prior covariance instead of posterior covariance
    cov_post = jnp.where(has_points[:, None, None], cov_post, prior_covs)
    
    # Sample new blob means
    new_blob_means = genjax.mv_normal.sample(posterior_key, mean_post, cov_post)

    return hdgmm_state.replace({'blobs_state': {'blob_means': new_blob_means}})

@jax.jit
def f_gibbs_sweep(carry, gibbs_sweep_idx):
    key, hdgmm_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs = carry
    # jprint("MCMC Gibbs sweep {}/{}", gibbs_sweep_idx+1, NUM_GIBBS_SWEEPS)
    
    # Datapoint-related updates 
    def datapoint_update_loop(i, state_key_tuple):
        hdgmm_state, key = state_key_tuple
        # Gibbs #3: Update blob assignments
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(
            gibbs_dials["blob_assignments"], 
            lambda k, s: gibbs_blob_assignments(k, s, position_only=False, velocity_only=False, disable_outlier_prob=True), 
            empty_gibbs, 
            gibbs_key, hdgmm_state)
        return (hdgmm_state, key)
    
    
    # Blob-related updates 
    def blob_update_loop(i, state_key_tuple):
        hdgmm_state, key = state_key_tuple
        # Gibbs #1: Update blob weights
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["blob_weights"], gibbs_blob_weights, empty_gibbs, gibbs_key, hdgmm_state)
        
        # Gibbs #4: Update hyperblob assignments
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_assignments"], gibbs_hyperblob_assignments, empty_gibbs, gibbs_key, hdgmm_state)
        
        # Gibbs #6: Update blob covariances
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["blob_covs"], gibbs_blob_covs, empty_gibbs, gibbs_key, hdgmm_state)
        
        # Gibbs #7: Update blob velocity covariances
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["blob_vel_covs"], gibbs_blob_vel_covs, empty_gibbs, gibbs_key, hdgmm_state)
        
        # Gibbs #8: Update blob velocity means
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["blob_vel_means"], gibbs_blob_vel_means, empty_gibbs, gibbs_key, hdgmm_state)
        
        # Gibbs #10: Update blob means
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["blob_means"], gibbs_blob_means, empty_gibbs, gibbs_key, hdgmm_state)
        
        return (hdgmm_state, key)
    
    # Hyperblob-related updates 
    def hyperblob_update_loop(i, state_key_tuple):
        hdgmm_state, key, use_weighted_blobs = state_key_tuple
        # Gibbs #2: Update hyperblob weights
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_weights"], gibbs_hyperblob_weights, empty_gibbs_weighted, gibbs_key, hdgmm_state, use_weighted_blobs)
        
        # Gibbs #5: Update hyperblob covariances
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_covs"], gibbs_hyperblob_covs, empty_gibbs_weighted, gibbs_key, hdgmm_state, use_weighted_blobs)
        
        # Gibbs #9: Update hyperblob means
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_means"], gibbs_hyperblob_means, empty_gibbs_weighted, gibbs_key, hdgmm_state, use_weighted_blobs)

        # Gibbs #11: Update hyperblob rotation velocities
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_rot_vels"], gibbs_hyperblob_rot, empty_gibbs_weighted, gibbs_key, hdgmm_state, use_weighted_blobs)

        # Gibbs #12: Update hyperblob translation velocities
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_trans_vels"], gibbs_hyperblob_trans, empty_gibbs_weighted, gibbs_key, hdgmm_state, use_weighted_blobs)
        
        return (hdgmm_state, key, use_weighted_blobs)
    
    hdgmm_state, key = jax.lax.fori_loop(0, num_gibbs_inner_loops, datapoint_update_loop, (hdgmm_state, key))
    hdgmm_state, key = jax.lax.fori_loop(0, num_gibbs_inner_loops, blob_update_loop, (hdgmm_state, key))
    hdgmm_state, key, _ = jax.lax.fori_loop(0, num_gibbs_inner_loops, hyperblob_update_loop, (hdgmm_state, key, use_weighted_blobs))
    
    return (key, hdgmm_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs), hdgmm_TraceWrapper(force_retval=hdgmm_state)

def hdgmm_full_gibbs(key, init_hdgmm_state, num_gibbs_sweeps, gibbs_dials, use_weighted_blobs=False, num_gibbs_inner_loops=1):
    _, stacked_hdgmm_wtrs = jax.lax.scan(f_gibbs_sweep, (key, init_hdgmm_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs), jnp.arange(num_gibbs_sweeps))
    gibbs_wtrs = HDGMM_Gibbs_TraceWrapper(hdgmm_TraceWrapper(force_retval=init_hdgmm_state), stacked_hdgmm_wtrs)
    return gibbs_wtrs

@jax.jit
def blob_tracking_gibbs(key, hdgmm_state : HDGMM_State):

    prior_means = hdgmm_state.blobs_state.blob_means
    prior_covs = jnp.repeat(hdgmm_state.hypers.Psi_B[None, :, :], hdgmm_state.hypers.n_blobs, axis = 0)

    def update_blob_assignments_position_only(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_assignments(gibbs_key, hdgmm_state, position_only = True, disable_outlier_prob = True)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_weights(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def update_blob_assignments_position_w_outlier(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_assignments(gibbs_key, hdgmm_state, position_only = True, disable_outlier_prob = False)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_weights(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def update_blob_velocities(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_vel_means(gibbs_key, hdgmm_state)
        return key, hdgmm_state


    def update_blob_velocity_covariances(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_vel_covs(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def update_blob_means_with_prior(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_transitioned_blob_means(gibbs_key, hdgmm_state, prior_means, prior_covs)

        return key, hdgmm_state
    
    key, hdgmm_state = jax.lax.fori_loop(0, 1, update_blob_assignments_position_only, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 5, update_blob_means_with_prior, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 1, update_blob_assignments_position_w_outlier, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 5, update_blob_velocities, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 5, update_blob_velocity_covariances, (key, hdgmm_state))
    
    return hdgmm_state

@jax.jit
def f_tracking_sweep(carry, timestep_idx):
    key, hdgmm_state, tracked_points, tracked_motion_vectors = carry
    
    next_blob_means = hdgmm_state.blobs_state.blob_vel_means + hdgmm_state.blobs_state.blob_means
    hdgmm_state = hdgmm_state.replace({'blobs_state': {'blob_means': next_blob_means}})
    
    hdgmm_state = hdgmm_state.replace({
        'datapoints_state': {
            'datapoint_positions': tracked_points[timestep_idx],
            'datapoint_vels': tracked_motion_vectors[timestep_idx]
        }
    })
    
    key, gibbs_key = jax.random.split(key)
    hdgmm_state = blob_tracking_gibbs(gibbs_key, hdgmm_state)
    
    return (key, hdgmm_state, tracked_points, tracked_motion_vectors), hdgmm_TraceWrapper(force_retval=hdgmm_state)

def hdgmm_tracking_gibbs(key, init_hdgmm_state, tracked_points, tracked_motion_vectors, outlier_prob):
    # Set outlier probability to very low value
    init_hdgmm_state = init_hdgmm_state.replace({'hypers': {'outlier_prob': f_(outlier_prob)}})
    
    timestep_indices = jnp.arange(1, len(tracked_points)-1)
    
    _, stacked_hdgmm_wtrs = jax.lax.scan(
        f_tracking_sweep, 
        (key, init_hdgmm_state, tracked_points, tracked_motion_vectors), 
        timestep_indices,
        unroll=1
    )
    
    tracking_wtrs = HDGMM_Gibbs_TraceWrapper(
        hdgmm_TraceWrapper(force_retval=init_hdgmm_state), 
        stacked_hdgmm_wtrs
    )
    
    return tracking_wtrs


@jax.jit
def f_gibbs_sweep_return_last_state(i, carry):
    key, hdgmm_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs = carry
    
    # Datapoint-related updates 
    def datapoint_update_loop(i, state_key_tuple):
        hdgmm_state, key = state_key_tuple
        # Gibbs #3: Update blob assignments
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(
            gibbs_dials["blob_assignments"], 
            lambda k, s: gibbs_blob_assignments(k, s, position_only=False, velocity_only=False, disable_outlier_prob=True), 
            empty_gibbs, 
            gibbs_key, hdgmm_state)
        return (hdgmm_state, key)
    
    
    # Blob-related updates 
    def blob_update_loop(i, state_key_tuple):
        hdgmm_state, key = state_key_tuple
        # Gibbs #1: Update blob weights
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["blob_weights"], gibbs_blob_weights, empty_gibbs, gibbs_key, hdgmm_state)
        
        # Gibbs #4: Update hyperblob assignments
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_assignments"], gibbs_hyperblob_assignments, empty_gibbs, gibbs_key, hdgmm_state)
        
        # Gibbs #6: Update blob covariances
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["blob_covs"], gibbs_blob_covs, empty_gibbs, gibbs_key, hdgmm_state)
        
        # Gibbs #7: Update blob velocity covariances
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["blob_vel_covs"], gibbs_blob_vel_covs, empty_gibbs, gibbs_key, hdgmm_state)
        
        # Gibbs #8: Update blob velocity means
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["blob_vel_means"], gibbs_blob_vel_means, empty_gibbs, gibbs_key, hdgmm_state)
        
        # Gibbs #10: Update blob means
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["blob_means"], gibbs_blob_means, empty_gibbs, gibbs_key, hdgmm_state)
        
        return (hdgmm_state, key)
    
    # Hyperblob-related updates 
    def hyperblob_update_loop(i, state_key_tuple):
        hdgmm_state, key, use_weighted_blobs = state_key_tuple
        # Gibbs #2: Update hyperblob weights
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_weights"], gibbs_hyperblob_weights, empty_gibbs_weighted, gibbs_key, hdgmm_state, use_weighted_blobs)
        
        # Gibbs #5: Update hyperblob covariances
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_covs"], gibbs_hyperblob_covs, empty_gibbs_weighted, gibbs_key, hdgmm_state, use_weighted_blobs)
        
        # Gibbs #9: Update hyperblob means
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_means"], gibbs_hyperblob_means, empty_gibbs_weighted, gibbs_key, hdgmm_state, use_weighted_blobs)

        # Gibbs #11: Update hyperblob rotation velocities
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_rot_vels"], gibbs_hyperblob_rot, empty_gibbs_weighted, gibbs_key, hdgmm_state, use_weighted_blobs)

        # Gibbs #12: Update hyperblob translation velocities
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = jax.lax.cond(gibbs_dials["hyperblob_trans_vels"], gibbs_hyperblob_trans, empty_gibbs_weighted, gibbs_key, hdgmm_state, use_weighted_blobs)
        
        return (hdgmm_state, key, use_weighted_blobs)
    
    hdgmm_state, key = jax.lax.fori_loop(0, num_gibbs_inner_loops, datapoint_update_loop, (hdgmm_state, key))
    hdgmm_state, key = jax.lax.fori_loop(0, num_gibbs_inner_loops, blob_update_loop, (hdgmm_state, key))
    hdgmm_state, key, _ = jax.lax.fori_loop(0, num_gibbs_inner_loops, hyperblob_update_loop, (hdgmm_state, key, use_weighted_blobs))
    
    return (key, hdgmm_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs)

from functools import partial

def hdgmm_full_gibbs_return_last_state(key, init_hdgmm_state, num_gibbs_sweeps, gibbs_dials, use_weighted_blobs, num_gibbs_inner_loops):
    carry = (key, init_hdgmm_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs)
    carry = jax.lax.fori_loop(0, num_gibbs_sweeps, f_gibbs_sweep_return_last_state, carry)
    _, hdgmm_state, _, _, _ = carry
    return hdgmm_state

hdgmm_full_gibbs_return_last_state_vmap = jax.vmap(hdgmm_full_gibbs_return_last_state, in_axes=(0, 0, None, None, None, None))
hdgmm_full_gibbs_return_last_state_vmap_jit = jax.jit(hdgmm_full_gibbs_return_last_state_vmap, static_argnames=['num_gibbs_sweeps', 'num_gibbs_inner_loops'])


def ransac_motion_only(points_data, ransac_thresh=2.0, fill_value=0.0):
    """
    Compute motion vectors using nearest neighbor + RANSAC (no symmetric check).
    
    Args:
        points_data: Array of shape (T, N, 2)
        dist_thresh: Distance threshold for accepting NN matches
        ransac_thresh: RANSAC reprojection error threshold
        fill_value: Value for unmatched or outlier points
    
    Returns:
        motion_vectors: Array of shape (T-1, N, 2)
    """


    T, N, _ = points_data.shape
    motion_vectors = np.full((T - 1, N, 2), fill_value, dtype=np.float32)

    outlier_mask = np.zeros((T - 1, N), dtype=bool)

    for t in range(T - 1):
        pts_t = points_data[t]
        pts_t1 = points_data[t + 1]

        if len(pts_t) < 3:
            continue  # Not enough for RANSAC
            
        # First compute KDTree-based motion vectors for all points
        tree = cKDTree(pts_t1)
        _, nn_indices = tree.query(pts_t, k=1)
        kdtree_motion_vectors = pts_t1[nn_indices] - pts_t

        # Then refine with RANSAC
        _, inliers = cv2.estimateAffinePartial2D(
            pts_t, pts_t1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
        )

        outlier_mask[t] = ~inliers[:,0].astype(bool)

        if inliers is None:
            continue

        inliers = inliers.flatten().astype(bool)
        # print number of inliers
        for src_idx, (dst_pt, is_inlier) in enumerate(zip(pts_t1, inliers)):
            if is_inlier:
                motion_vectors[t, src_idx] = dst_pt - pts_t[src_idx]

    return motion_vectors, outlier_mask

def initialize_model(key, config_num, start_frame, end_frame):
    """Initialize the HDGMM model with data from the specified configuration."""

    number_of_hyperblobs = 5
    number_of_blobs = 500
    # Load data
    npz_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"assets/RDK/config_{config_num}/data.npz")
    data = np.load(npz_path)

    # Extract and scale data
    if start_frame == None or end_frame == None:
        points_data = data["points_data"]       # (T, N, 2)
    else:
        points_data = data["points_data"][start_frame:end_frame+2]       # (T, N, 2)
    motion_vectors, outlier_mask = ransac_motion_only(points_data, ransac_thresh=50, fill_value=0.0)

    num_t_steps = motion_vectors.shape[0]

    # Create hierarchical kmeans model
    kmeans_chm_original = make_hierarchical_kmeans_chm_2d(points_data, number_of_blobs, number_of_hyperblobs, frame_idx=0, motion_vectors=motion_vectors)

    vel_scale = 1e-6
    v_sigma = 0.35

    kmeans_chm = kmeans_chm_original.at['blobs', 'blob_vel_covs'].set(kmeans_chm_original['blobs', 'blob_vel_covs'] * vel_scale) # 1e-16

    # Single sample, no batching
    num_samples = 1

    GIBBS_DIALS = {
        "blob_weights": True,
        "hyperblob_weights": True,
        "blob_assignments": True,
        "hyperblob_assignments": True,
        "hyperblob_covs": True,
        "blob_covs": False,
        "blob_vel_covs": False,
        "blob_vel_means": True,
        "hyperblob_means": True,
        "blob_means": True,
        'hyperblob_rot_vels': True,
        'hyperblob_trans_vels': True
    }

    num_datapoints = kmeans_chm['datapoints', 'datapoint_positions'].shape[0]
    num_blobs = kmeans_chm['blobs', 'hyperblob_assignments'].shape[0]
    num_hyperblobs = kmeans_chm['hyperblobs', 'hyperblob_means'].shape[0]
    empirical_mu_H = jnp.median(kmeans_chm['datapoints', 'datapoint_positions'], axis=0)
    empirical_sigma_H = (400*0.5)**2
    empirical_Psi_B = jnp.median(kmeans_chm['blobs', 'blob_covs'], axis=0)
    empirical_Psi_H = jnp.median(kmeans_chm['hyperblobs', 'hyperblob_covs'], axis=0)
    empirical_Psi_V = jnp.median(kmeans_chm['blobs', 'blob_vel_covs'], axis=0)
    empirical_nu_H = f_(int(jnp.median(kmeans_chm['hyperblobs', 'hyperblob_weights']) * num_blobs))
    empirical_nu_B = empirical_nu_V = f_(int(jnp.median(kmeans_chm['blobs', 'blob_weights']) * num_datapoints))

    # Create hyperparameters for 2D model
    hypers = HDGMM_Hyperparams.create(
        outlier_prob = f_(5e-2),
        outlier_velocity_gamma_shape = f_(5.0),
        outlier_velocity_gamma_rate = f_(1.0),
        alpha = f_(99.994),
        beta = f_(0.009899),
        mu_H = empirical_mu_H,
        sigma_H = empirical_sigma_H,
        nu_H = empirical_nu_H,
        Psi_H = empirical_Psi_H,
        nu_B = empirical_nu_B,
        Psi_B = empirical_Psi_B,
        sigma_V = f_(v_sigma), # 5e-8
        nu_V = empirical_nu_V,
        Psi_V = empirical_Psi_V * vel_scale,
        translation_gaussian_scale = snp(f_(400)),
        translation_max_radius = snp(40),
        translation_num_radii_cells = snp(400),
        translation_theta_step_deg = snp(1),
        rotation_vmf_kappa = snp(f_(75)),
        rotation_angle_max_deg = snp(5),
        rotation_angle_step_deg = snp(0.02),
        n_hyperblobs = num_hyperblobs,
        n_blobs = num_blobs,
        n_datapoints = num_datapoints,
    )
    
    # Initialize single sample
    key, key_init = jax.random.split(key, 2)
    init_tr, _ = model_jimportance(key_init, kmeans_chm, (hypers,))
    init_hdgmm_state = init_tr.get_retval()

    init_hdgmm_state = init_hdgmm_state.replace(
        datapoints_state=init_hdgmm_state.datapoints_state.replace(
            blob_assignments=init_hdgmm_state.datapoints_state.blob_assignments.at[outlier_mask[0]].set(number_of_blobs)
        )
    )

    return key, init_hdgmm_state, points_data, motion_vectors, outlier_mask, num_t_steps, number_of_blobs, GIBBS_DIALS


def run_gibbs_sampling(key, init_hdgmm_state, points_data, motion_vectors, outlier_mask, num_t_steps, number_of_blobs, GIBBS_DIALS, show_viz=False):
    """Run Gibbs sampling on the initialized model."""
    NUM_GIBBS_SWEEPS = 5
    posterior_over_time = []

    for t in range(num_t_steps):
        if t > 0:
            # Update datapoints_state for the next timestep
            init_hdgmm_state = init_hdgmm_state.replace(
                {'datapoints_state': 
                    {'datapoint_positions': points_data[t],
                    'datapoint_vels': motion_vectors[t]}
                }
            )

            init_hdgmm_state = init_hdgmm_state.replace(
                {'blobs_state':
                    {
                        'blob_means': init_hdgmm_state.blobs_state.blob_means + init_hdgmm_state.blobs_state.blob_vel_means
                    }
                }
            )
        
        # Generate random key
        key, key_gibbs = jax.random.split(key, 2)
        
        # Run Gibbs sampling for single state
        init_hdgmm_state = hdgmm_full_gibbs_return_last_state(
            key_gibbs, init_hdgmm_state, NUM_GIBBS_SWEEPS, GIBBS_DIALS, False, 1
        )

        init_hdgmm_state = init_hdgmm_state.replace(
            datapoints_state=init_hdgmm_state.datapoints_state.replace(
                blob_assignments=init_hdgmm_state.datapoints_state.blob_assignments.at[outlier_mask[t]].set(number_of_blobs)
            )
        )
        
        posterior_over_time.append(hdgmm_TraceWrapper(force_retval=init_hdgmm_state))

    posterior_over_time = pytree_stack(posterior_over_time)

    return posterior_over_time, outlier_mask


def get_model_results_via_datapoints(key, posterior_over_time, outlier_mask, start_frame, end_frame, red_point, green_point, probe_timestep):

    # Get the probe index relative to the start frame
    probe_index = probe_timestep - start_frame
    num_frames = end_frame - start_frame
    
    # Initialize arrays to track hyperblob assignments over time
    red_hyperblob_over_time = []
    green_hyperblob_over_time = []
    
    # Helper function to find k nearest non-outlier neighbors and their hyperblob assignments
    def find_neighbors_and_hyperblob(point, positions, indices, blob_assignments, hyperblob_assignments, k=5):
        # Calculate distances from the point to all valid positions
        distances = jnp.sqrt(jnp.sum((positions - point) ** 2, axis=1))
        
        # Get indices of k nearest points
        nearest_indices = jnp.argsort(distances)[:k]
        
        # Get the original indices, distances, and blob assignments
        original_indices = indices[nearest_indices]
        nearest_distances = distances[nearest_indices]
        nearest_blob_assignments = blob_assignments[original_indices]
        
        # Get hyperblob assignments for these blobs
        nearest_hyperblob_assignments = jnp.array([hyperblob_assignments[int(blob)] for blob in nearest_blob_assignments])
        
        # Find the most common hyperblob assignment
        hyperblob_counts = {}
        for assignment in nearest_hyperblob_assignments:
            assignment_int = int(assignment)
            if assignment_int in hyperblob_counts:
                hyperblob_counts[assignment_int] += 1
            else:
                hyperblob_counts[assignment_int] = 1
        
        most_common_hyperblob = max(hyperblob_counts, key=hyperblob_counts.get)
        
        # Return the indices of neighbors that belong to the most common hyperblob
        same_hyperblob_mask = nearest_hyperblob_assignments == most_common_hyperblob
        same_hyperblob_indices = original_indices[same_hyperblob_mask]
        
        return most_common_hyperblob, same_hyperblob_indices, original_indices
    
    # Track points through time starting from probe timestep
    red_current_point = jnp.array(red_point).astype(jnp.float32)
    green_current_point = jnp.array(green_point).astype(jnp.float32)
    
    # Process each frame starting from probe_index
    for t in range(probe_index, num_frames):
        # Get the current frame's posterior and outlier mask
        current_posterior = posterior_over_time[t].retval
        current_outlier_mask = outlier_mask[t]
        
        # Get datapoint positions and blob assignments
        datapoint_positions = current_posterior.datapoints_state.datapoint_positions
        blob_assignments = current_posterior.datapoints_state.blob_assignments
        hyperblob_assignments = current_posterior.blobs_state.hyperblob_assignments
        
        # Convert to jax numpy for manipulation
        positions_jnp = jnp.array(datapoint_positions)
        non_outlier_mask = ~current_outlier_mask
        valid_positions = positions_jnp[non_outlier_mask]
        valid_indices = jnp.where(non_outlier_mask)[0]
        
        # Find hyperblob assignments and neighbors for red and green points
        k = 10  # Number of nearest neighbors
        red_hyperblob, red_same_hyperblob_indices, red_all_indices = find_neighbors_and_hyperblob(
            red_current_point, valid_positions, valid_indices, blob_assignments, hyperblob_assignments, k
        )
        green_hyperblob, green_same_hyperblob_indices, green_all_indices = find_neighbors_and_hyperblob(
            green_current_point, valid_positions, valid_indices, blob_assignments, hyperblob_assignments, k
        )
        
        # Record hyperblob assignments
        red_hyperblob_over_time.append(red_hyperblob)
        green_hyperblob_over_time.append(green_hyperblob)
        
        # If we're not at the last frame, propagate points forward
        if t < num_frames - 1:
            # Get the current frame's velocity information directly
            datapoint_velocities = current_posterior.datapoints_state.datapoint_vels
            
            # Calculate velocities for neighbors in the same hyperblob
            if len(red_same_hyperblob_indices) > 0:
                red_velocities = jnp.array(datapoint_velocities)[red_same_hyperblob_indices]
                red_median_velocity = jnp.median(red_velocities, axis=0)
                red_current_point = red_current_point + red_median_velocity
            
            if len(green_same_hyperblob_indices) > 0:
                green_velocities = jnp.array(datapoint_velocities)[green_same_hyperblob_indices]
                green_median_velocity = jnp.median(green_velocities, axis=0)
                green_current_point = green_current_point + green_median_velocity
    
    # Convert lists to arrays
    red_hyperblob_over_time = jnp.array(red_hyperblob_over_time)
    green_hyperblob_over_time = jnp.array(green_hyperblob_over_time)
    
    # Calculate frames where they belong to the same hyperblob
    same_hyperblob_over_time = red_hyperblob_over_time == green_hyperblob_over_time
    
    # Check if they belong to the same hyperblob in the final frame
    final_frame_same_hyperblob = same_hyperblob_over_time[-1]
    
    return final_frame_same_hyperblob


def model_prediction_on_stimulus(exp_key, config_num, start_frame, end_frame, probe_timestep, red_point, green_point, num_runs=17):
    # Initialize the model once
    initialize_model_vmapped = jax.vmap(
        lambda key: initialize_model(key, config_num=config_num, start_frame=start_frame, end_frame=end_frame)
    )
    exp_key, init_key = jax.random.split(exp_key, 2)
    init_keys = jax.random.split(init_key, num_runs)
    _, init_hdgmm_states, points_data, motion_vectors, outlier_masks, num_t_steps, number_of_blobs, GIBBS_DIALS = initialize_model_vmapped(init_keys)

    points_data = points_data[0]
    motion_vectors = motion_vectors[0]
    outlier_masks = outlier_masks[0]
    num_t_steps = num_t_steps[0]
    number_of_blobs = number_of_blobs[0]

    GIBBS_DIALS = {
        "blob_weights": True,
        "hyperblob_weights": True,
        "blob_assignments": True,
        "hyperblob_assignments": True,
        "hyperblob_covs": True,
        "blob_covs": False,
        "blob_vel_covs": False,
        "blob_vel_means": True,
        "hyperblob_means": True,
        "blob_means": True,
        'hyperblob_rot_vels': True,
        'hyperblob_trans_vels': True
    }

    # Generate multiple keys for sampling
    key, gibbs_keys = jax.random.split(exp_key, 2)
    exp_keys = jax.random.split(gibbs_keys, num_runs)

    # Create a vmapped version of just the Gibbs sampling part
    run_gibbs_sampling_vmapped = jax.vmap(
        lambda key, init_hdgmm_state: run_gibbs_sampling(
            key, init_hdgmm_state, points_data, motion_vectors, outlier_masks, 
            num_t_steps, number_of_blobs, GIBBS_DIALS, show_viz=False
        )
    )
    posterior_over_time_list, outlier_mask_list = run_gibbs_sampling_vmapped(exp_keys, init_hdgmm_states)
    
    # Visualize the model results via datapoints
    final_results = []
    for i in range(num_runs):
        posterior = posterior_over_time_list[i]
        outlier_mask = outlier_mask_list[i]
        final_frame_same_hyperblob = get_model_results_via_datapoints(
            key,
            posterior, 
            outlier_mask, 
            start_frame, 
            end_frame, 
            red_point, 
            green_point, 
            probe_timestep=probe_timestep
        )
        final_results.append(final_frame_same_hyperblob)
    
    # Convert the list to a numpy array and compute the mean
    final_results = jnp.array(final_results)
    
    return final_results