import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

def calculate_mutual_info(z1, z2):
    """
    Calculate mutual information between z1 and z2.
    
    Args:
        z1: First set of embeddings (numpy array).
        z2: Second set of embeddings (numpy array).
        
    Returns:
        Mutual information value.
    """
    n_features = z1.shape[1]
    mi_scores = np.zeros(n_features)
    
    for i in range(n_features):
        mi_scores[i] = mutual_info_regression(
            z1[:, i].reshape(-1, 1), 
            z2[:, i]
        )
    
    return np.mean(mi_scores)

def calculate_pca_explained_variance(embeddings, n_components=10):
    """
    Calculate PCA explained variance for embeddings.
    
    Args:
        embeddings: Embedding vectors (numpy array).
        n_components: Number of PCA components.
        
    Returns:
        Explained variance ratio array.
    """
    # Ensure we don't try more components than samples or features
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    
    return pca.explained_variance_ratio_

def normalize_embeddings(embeddings):
    """
    Normalize embeddings to unit norm.
    
    Args:
        embeddings: Embedding vectors (numpy array).
        
    Returns:
        Normalized embeddings.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)
