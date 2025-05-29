import numpy as np
import random
from matplotlib.image import imread
import matplotlib.pyplot as plt


def initialize(A, K):
    # Initialize centroids using k-means++ strategy
    reshaped_A = A.reshape(-1, A.shape[-1])
    centroids = [reshaped_A[random.randint(0, reshaped_A.shape[0] - 1)]]
    
    for _ in range(1, K):
        distances = np.min(
            np.linalg.norm(reshaped_A[:, None, :] - np.array(centroids)[None, :, :], axis=2),
            axis=1
        )
        probabilities = distances / distances.sum()
        next_centroid = reshaped_A[np.random.choice(reshaped_A.shape[0], p=probabilities)]
        centroids.append(next_centroid)
        
    return centroids


def closest_to_centroids(A, centroids):
    # Vectorized computation of closest centroid for each pixel
    reshaped_A = A.reshape(-1, A.shape[-1])
    distances = np.linalg.norm(reshaped_A[:, None, :] - np.array(centroids)[None, :, :], axis=2)
    closest = np.argmin(distances, axis=1)
    return closest.reshape(A.shape[:2])


def A_selection(A, closest, j):
    # Efficient selection of pixels belonging to cluster j
    mask = closest == j
    return A[mask]


def K_clusters(A, K, max_iter, threshold):
    iter = 0
    centroids = initialize(A, K)
    pre_centroids = None

    while iter < max_iter:
        pre_centroids = centroids.copy()
        
        # Assign each pixel to the closest centroid
        closest = closest_to_centroids(A, centroids)
        
        # Recompute centroids
        for j in range(K):
            A_ik = A_selection(A, closest, j)
            if len(A_ik) > 0:
                centroids[j] = np.mean(A_ik, axis=0)
        
        iter += 1
        
        if iter % 10 == 0:
            print(f"At iteration {iter}")
        
        # Convergence check
        if np.linalg.norm(np.array(centroids) - np.array(pre_centroids)) < threshold:
            print(f'Convergence at iteration {iter}')
            return centroids, closest
    
    return centroids, closest


def compression(A, centroids, closest):
    # Compress image by replacing pixels with their centroid values
    compressed_A = np.zeros_like(A)
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            compressed_A[i][k] = centroids[closest[i][k]]
    return compressed_A


def main():
    # Load image
    A = imread("../data/peppers-small.tiff")
    B = imread("../data/peppers-large.tiff")
    # Apply K-means clustering for compression
    centroids, closest = K_clusters(B, K=16, max_iter=100, threshold=1e-3)
    
    # Compress the image
    compressed_A = compression(B, centroids, closest)
    
    # Display compressed image
    plt.imshow(compressed_A)
    plt.show()


if __name__ == "__main__":
    main()
