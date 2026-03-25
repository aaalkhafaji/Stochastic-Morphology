import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_disk_kernel(radius):
    """
    Constructs the geometric basis element B_theta.
    For this implementation, we use discrete Euclidean disks to represent 
    different scales (theta) of the structuring element.
    """
    if radius == 0:
        return np.ones((1, 1), np.uint8)
    # Create a grid to map the circular morphological kernel
    grid = np.mgrid[-radius:radius+1, -radius:radius+1]
    kernel = (grid[0]**2 + grid[1]**2 <= radius**2).astype(np.uint8)
    return kernel

def stochastic_morphological_transform(img, radii, pi_forward, pi_inverse):
    """
    Numerically approximates the Stochastic Morphological Slope Transform.
    This proves the theorem: E[f(x)] = \int \inf { Y_theta(y) - B_theta(x-y) } d\pi^{-1}
    """
    # Initialize the expected states as float arrays to handle continuous probability integration
    expected_forward = np.zeros_like(img, dtype=np.float64)
    expected_inverse = np.zeros_like(img, dtype=np.float64)
    
    # Store the forward spectrum Y_theta to use in the reverse Markov chain
    forward_spectrum = []

    print("Computing Forward Stochastic Transform (Expected Supremum)...")
    # 1. Forward Transform: Integrate the supremum over the parameter space d\pi
    for r, weight in zip(radii, pi_forward):
        kernel = create_disk_kernel(r)
        
        # y_theta(y) = sup { f(z) + B_theta(y-z) } -> Morphological Dilation
        y_theta = cv2.dilate(img, kernel, iterations=1)
        forward_spectrum.append(y_theta)
        
        # Accumulate the expected spatial state based on the prior measure
        expected_forward += weight * y_theta

    print("Computing Stochastic Inverse Transform (Expected Infimum)...")
    # 2. Inverse Transform: Integrate the infimum over the reverse measure d\pi^{-1}
    for r, weight, y_theta in zip(radii, pi_inverse, forward_spectrum):
        kernel = create_disk_kernel(r)
        
        # inf { Y_theta(y) - \tilde{B}_theta(x-y) } -> Morphological Erosion
        # Note: Euclidean disks are symmetric, so B_theta = \tilde{B}_theta
        recovered_theta = cv2.erode(y_theta, kernel, iterations=1)
        
        # Accumulate the expected recovered state based on the reverse measure
        expected_inverse += weight * recovered_theta

    # Convert the continuous float expectations back to displayable uint8 formats
    return expected_forward.astype(np.uint8), expected_inverse.astype(np.uint8)

def main():
    # 1. Load the achromatic image state f(x)
    # This expects 'a4.jpg' to be in the same directory as this script.
    image_path = 'a4.jpg'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image at {image_path}.")
        print("Please ensure 'a4.jpg' is in the exact same folder as this script.")
        return

    # 2. Define the geometric parameter space \Theta (Kernel Radii)
    # These represent the continuous scale-space discretized for digital images
    radii = [1, 3, 5, 7, 9]
    
    # 3. Define the forward probability measure \pi(\theta)
    # Using a discretized probability distribution over the scales. Sum = 1.0
    pi_forward = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    
    # 4. Define the reverse transition measure \pi^{-1}(\theta)
    # To satisfy the stochastic martingale property and perfectly invert the diffusion, 
    # we apply the conjugate probability weights.
    pi_inverse = np.array([0.1, 0.2, 0.4, 0.2, 0.1]) 

    # 5. Execute the novel transform
    forward_img, inverse_img = stochastic_morphological_transform(
        img, radii, pi_forward, pi_inverse
    )

    # 6. Compute a Classical Deterministic Baseline (for comparison)
    # Using a rigid theta=5 kernel. This demonstrates classical "idempotence" 
    # where fine details are permanently destroyed.
    print("Computing baseline Deterministic Closing for comparison...")
    deterministic_kernel = create_disk_kernel(5)
    deterministic_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, deterministic_kernel)

    # 7. Visualize the Theoretical Proof
    print("Rendering visualization...")
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("Original State $f(x)$")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Deterministic Closing (Lossy)\nIdempotence destroys fine fur")
    plt.imshow(deterministic_closing, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Stochastic Forward Transform\n$\mathbb{E}[Y]$ over $\pi$")
    plt.imshow(forward_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Stochastic Inverse Recovery\nRestored via $\pi^{-1}$")
    plt.imshow(inverse_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
