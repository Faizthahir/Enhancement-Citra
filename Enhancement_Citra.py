import cv2
import numpy as np
import matplotlib.pyplot as plt

# LOAD IMAGE
def load_images():
    under = cv2.imread("underexposed.jpeg", cv2.IMREAD_GRAYSCALE)
    over = cv2.imread("overexposed.jpeg", cv2.IMREAD_GRAYSCALE)
    uneven = cv2.imread("uneven.jpeg", cv2.IMREAD_GRAYSCALE)
    return under, over, uneven

# POINT PROCESSING
def negative_transform(img):
    return 255 - img


def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img)
    return np.array(log_img, dtype=np.uint8)


def gamma_transform(img, gamma):
    img_norm = img / 255.0
    gamma_corrected = np.power(img_norm, gamma)
    return np.uint8(gamma_corrected * 255)

# HISTOGRAM BASED METHODS
def contrast_stretch_manual(img, r1=70, r2=140):
    s1 = 0
    s2 = 255

    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            r = img[i, j]

            if r < r1:
                result[i, j] = (s1 / r1) * r

            elif r1 <= r <= r2:
                result[i, j] = ((s2 - s1) / (r2 - r1)) * (r - r1) + s1

            else:
                result[i, j] = ((255 - s2) / (255 - r2)) * (r - r2) + s2

    return result.astype(np.uint8)


def contrast_stretch_auto(img):
    r_min = np.min(img)
    r_max = np.max(img)

    stretched = (img - r_min) * (255 / (r_max - r_min))
    return stretched.astype(np.uint8)


def histogram_equalization(img):
    return cv2.equalizeHist(img)


def clahe_equalization(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)


# METRICS
def contrast_ratio(img):
    return (np.max(img) - np.min(img)) / (np.max(img) + np.min(img) + 1e-5)


def entropy(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    hist_norm = hist_norm[hist_norm>0]
    return -np.sum(hist_norm*np.log2(hist_norm))

# VISUALIZATION
def show_images(images, titles):

    plt.figure(figsize=(12,8))

    for i in range(len(images)):
        plt.subplot(2,4,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_histogram(img, title):

    plt.figure()
    plt.hist(img.ravel(),256,[0,256])
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.show()

# PROCESS PIPELINE
def process_image(img, name):

    results = []
    titles = []

    results.append(img)
    titles.append("Original")

    # Point processing
    neg = negative_transform(img)
    log = log_transform(img)
    gamma1 = gamma_transform(img, 0.5)
    gamma2 = gamma_transform(img, 1.5)
    gamma3 = gamma_transform(img, 2.5)

    # Histogram methods
    cs_manual = contrast_stretch_manual(img)
    cs_auto = contrast_stretch_auto(img)
    hist_eq = histogram_equalization(img)
    clahe = clahe_equalization(img)

    results.extend([neg, log, gamma1, gamma2, gamma3, cs_auto, hist_eq, clahe])

    titles.extend([
        "Negative",
        "Log",
        "Gamma 0.5",
        "Gamma 1.5",
        "Gamma 2.5",
        "Contrast Stretch",
        "Hist Equalization",
        "CLAHE"
    ])

    show_images(results[:8], titles[:8])

    # histogram before-after
    show_histogram(img, f"{name} - Original Histogram")
    show_histogram(hist_eq, f"{name} - After Histogram Equalization")

    # metrics
    print("\n=== Metrics:", name, "===")
    print("Original Contrast:", contrast_ratio(img))
    print("Original Entropy :", entropy(img))

    print("CLAHE Contrast:", contrast_ratio(clahe))
    print("CLAHE Entropy :", entropy(clahe))

# MAIN
def main():

    under, over, uneven = load_images()

    process_image(under, "Underexposed")
    process_image(over, "Overexposed")
    process_image(uneven, "Uneven Illumination")


if __name__ == "__main__":
    main()