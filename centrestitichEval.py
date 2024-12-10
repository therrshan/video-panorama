import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_images_left_to_right(images):
    """Stitch images from left to right."""
    stitcher = cv2.createStitcher() if cv2.__version__.startswith('4.') else cv2.createStitcher(False)
    status, stitched = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        print("Error in stitching images from left to right.")
    return stitched


def stitch_images_center_out(images):
    """Stitch images from center outwards."""
    # Begin stitching with the middle image
    middle_index = len(images) // 2
    left_image = images[middle_index - 1] if middle_index - 1 >= 0 else None
    right_image = images[middle_index + 1] if middle_index + 1 < len(images) else None

    # Start stitching from the middle
    stitched = images[middle_index]

    # Stitch the left image
    if left_image is not None:
        stitcher = cv2.createStitcher() if cv2.__version__.startswith('4.') else cv2.createStitcher(False)
        status, stitched = stitcher.stitch([left_image, stitched])
        if status != cv2.Stitcher_OK:
            print("Error in stitching left image.")

    # Stitch the right image
    if right_image is not None:
        stitcher = cv2.createStitcher() if cv2.__version__.startswith('4.') else cv2.createStitcher(False)
        status, stitched = stitcher.stitch([stitched, right_image])
        if status != cv2.Stitcher_OK:
            print("Error in stitching right image.")

    return stitched


# Load the three images (make sure to replace these with actual file paths)
image1 = cv2.imread('./videos/scene1_a.jpg')
image2 = cv2.imread('./videos/scene1_b.jpg')
image3 = cv2.imread('./videos/scene1_c.jpg')

# Ensure the images are loaded correctly
if image1 is None or image2 is None or image3 is None:
    print("Error loading images.")
else:
    # Stitch images from left to right
    stitched_lr = stitch_images_left_to_right([image1, image2, image3])

    # Stitch images from center out
    stitched_co = stitch_images_center_out([image1, image2, image3])

    # Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Left to Right Stitching")
    plt.imshow(cv2.cvtColor(stitched_lr, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Center Out Stitching")
    plt.imshow(cv2.cvtColor(stitched_co, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
