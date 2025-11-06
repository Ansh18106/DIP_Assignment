import cv2
import numpy as np
import os
import scipy.ndimage as ndi
import skimage.measure
import matplotlib.pyplot as plt


# Question 1 #

image = cv2.imread('images\coins.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image can not be loaded")
    exit()
else:
    print("Image Loaded Successfully\n")
    print("Image Shape:", image.shape)


def compute_histogram(img):
    histogram = [0] * 256
    for row in img:
        for pixel in row:
            histogram[int(pixel)] += 1
    return histogram


# histogram = compute_histogram(image)
# print(" Histogram computed successfully\n")
# print(histogram)


def plot_histograms(histogram):
    plt.figure(figsize=(10, 5))
    plt.bar(range(256), histogram, color='gray')
    plt.xlabel('Intensity Level')
    plt.ylabel("Frequency")
    plt.title('Histogram of Coins Image')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'coins_histogram.png')
    plt.savefig(save_path)
    plt.show()


# plot_histograms(histogram)


def avg_intensity_of_Histogram(histogram):
    total_pixels = sum(histogram)
    weighted_sum = 0

    for intensity in range(256):
        weighted_sum += intensity * histogram[intensity]

    avg = weighted_sum / total_pixels
    return avg


# avg_intensity_Coins = avg_intensity_of_Histogram(histogram)
# print("Average Intensity Calculated from Hisotgram : ", avg_intensity_Coins)

# avg_direct = np.mean(image)
# print("Avergae Intensity Calculated from Image Directly : ", avg_direct)

# diff = abs(avg_intensity_Coins - avg_direct)
# print(f"Difference between Averages : {diff}")


# if diff < 1e-6:
#     print("Verification successful: Both averages match closely.")
# else:
#     print("Warning: Averages differ significantly.")


# Question 2#


# 2A#
def compute_probs(image):
    hist = compute_histogram(image)
    total_pixels = image.shape[0] * image.shape[1]
    if total_pixels == 0:
        return [0] * 256
    probs = [count / total_pixels for count in hist]
    return probs


def within_class_var(image, t):
    prob = compute_probs(image)
    w_0 = sum(prob[:t+1])
    w_1 = sum(prob[t+1:])

    if w_0 == 0 or w_1 == 0:
        return float('inf')

    mean_0 = sum(i*prob[i] for i in range(t+1)) / w_0
    mean_1 = sum(i*prob[i] for i in range(t+1, 256)) / w_1

    var_0 = sum(((i-mean_0)**2) * prob[i] for i in range(t+1)) / w_0
    var_1 = sum(((i-mean_1)**2) * prob[i] for i in range(t+1, 256)) / w_1

    sigma_w = w_0 * var_0 + w_1 * var_1
    return sigma_w


def otsu_min_within_class_var(image):
    min_var = float('inf')
    optimal_threshold = 0

    for t in range(1, 255):
        sigma_w = within_class_var(image, t)
        if sigma_w < min_var:
            min_var = sigma_w
            optimal_threshold = t

    return optimal_threshold


def binarize_image(image, threshold):
    binary_img = []
    for row in image:
        binary_row = []
        for pixels in row:
            if pixels > threshold:
                binary_row.append(255)
            else:
                binary_row.append(0)
        binary_img.append(binary_row)
    return binary_img


# t = otsu_min_within_class_var(image)
# print("Optimal Threshold by minimizing within-class variance:", t)

# binary_img_original = binarize_image(image, t)

# 2B#


def add_offset(image, offset=20):
    offset_img = np.clip(image + offset, 0, 255)
    return offset_img.astype(np.uint8)


def all_within_class_var(img):
    var = []
    for t in range(256):
        sigma_w = within_class_var(img, t)
        var.append(sigma_w)
    return var


# img_shift = add_offset(image)
# p_shift = compute_probs(img_shift)
# within_class_var_shift_img = all_within_class_var(img_shift)
# optimal_t_shift_img = otsu_min_within_class_var(img_shift)
# print("Optimal Threshold(Shifted Image) by minimizing within-class variance:",
#       optimal_t_shift_img)


def between_class_var(image, t):
    prob = compute_probs(image)
    w_0 = sum(prob[:t+1])
    w_1 = sum(prob[t+1:])

    if w_0 == 0 or w_1 == 0:
        return 0

    mean_0 = sum(i * prob[i] for i in range(t+1)) / w_0
    mean_1 = sum(i * prob[i] for i in range(t+1, 256)) / w_1

    sigma_b = w_0 * w_1 * (mean_0 - mean_1) ** 2
    return sigma_b


def all_between_class_var(img):
    var = []
    for t in range(256):
        sigma_b = between_class_var(img, t)
        var.append(sigma_b)
    return var


def otsu_max_between_class_var(image):
    all_bwc_var = all_between_class_var(image)
    optimal_t = int(np.argmax(all_bwc_var))
    return optimal_t


# optimal_t2_shifted_img = otsu_max_between_class_var(img_shift)
# print("Optimal Threshold(Shifted Image) by maximizing between-class variance:",
#       optimal_t2_shifted_img)

# binary_shifted_img = binarize_image(img_shift, optimal_t2_shifted_img)
# ## plots##

# plt.figure(figsize=(10, 4))
# plt.plot(within_class_var_shift_img,
#          label=" Within class variance(Shifted Image)")
# plt.axvline(optimal_t_shift_img, color='r', linestyle='--',
#             label=f"Optimal t (min wcv) = {optimal_t_shift_img}")
# plt.xlabel("Threshold t")
# plt.ylabel("Variance")
# plt.title("Within Class Variance VS Threshold (Shifted Image)")
# plt.legend()

# script_dir = os.path.dirname(os.path.abspath(__file__))
# save_path = os.path.join(script_dir, '1.png')
# plt.savefig(save_path)
# plt.show()


# between_class_var_shift_img = all_between_class_var(img_shift)
# optimal_t2_shifted_img = int(np.argmax(between_class_var_shift_img))
# plt.figure(figsize=(10, 4))
# plt.plot(between_class_var_shift_img,
#          label="Between-class variance (Shifted Image)")
# plt.axvline(optimal_t2_shifted_img, color='g', linestyle='--',
#             label=f"Optimal t (max bcv) = {optimal_t2_shifted_img}")
# plt.xlabel("Threshold t")
# plt.ylabel("Variance")
# plt.title("Between-class variance vs Threshold (shifted image)")
# plt.legend()
# script_dir = os.path.dirname(os.path.abspath(__file__))
# save_path = os.path.join(script_dir, '2.png')
# plt.savefig(save_path)
# plt.show()

# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(binary_img_original, cmap='gray')
# plt.title(f"Binarize Original Image\n Threshold = {t}")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(binary_shifted_img, cmap='gray')
# plt.title(f"Binarize Shifted Image\n Threshold = {optimal_t2_shifted_img}")
# plt.axis('off')
# script_dir = os.path.dirname(os.path.abspath(__file__))
# save_path = os.path.join(script_dir, '3.png')
# plt.savefig(save_path)
# plt.show()

# Question 3#

image_1 = cv2.imread('images\sudoku.png', cv2.IMREAD_GRAYSCALE)
rows, cols = image_1.shape


def get_overlapping_blocks(image, block_size, overlap=0.2):
    stride = int(block_size*(1-overlap))
    positions = []
    blocks = []
    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            block = image[i:min(i+block_size, rows), j:min(j+block_size, cols)]
            # if block.shape == (block_size, block_size):
            positions.append((i, j))
            blocks.append(block)
    return positions, blocks

# blocks_10 = get_overlapping_blocks(image, 10)
# print(f"Extracted {len(blocks_10)} blocks of size 10x10 with 20% overlap.")


def binarize_block(block):
    probs = compute_probs(block)
    threshold = otsu_min_within_class_var(block)
    binary_block = (block > threshold).astype(np.uint8)
    return binary_block


def construct_images_with_votes(positions, bin_blocks, image_shape, block_size):
    vote_count = np.zeros(image_shape, dtype=int)
    vote_sum = np.zeros(image_shape, dtype=int)
    for (i, j), block in zip(positions, bin_blocks):
        vote_count[i:i+block_size, j:j+block_size] += 1
        vote_sum[i:i+block_size, j:j+block_size] += (block > 0)
    output = (vote_sum > (vote_count/2)).astype(np.uint8) * 255
    return output


def adaptive_binarization(image, block_size, overlap=0.2):
    positions, blocks = get_overlapping_blocks(image, block_size, overlap)
    bin_blocks = [binarize_block(block) for block in blocks]
    output = construct_images_with_votes(
        positions, bin_blocks, image.shape, block_size)
    return output


block_sizes = [50]
results = []

for n in block_sizes:
    print(f"Binarizing with block size {n}x{n}...")
    result = adaptive_binarization(image_1, n)
    results.append((f"{n}x{n}", result))

# thr = otsu_min_within_class_var(image_1)
# global_bin = (image_1 > thr).astype(np.uint8) * 255
# results.append(("Global", global_bin))

plt.figure(figsize=(18, 6))
for i, (title, img) in enumerate(results):
    plt.subplot(1, len(results), i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, '5.png')
plt.savefig(save_path)

plt.show()

# Questions 4#

image = cv2.imread('quote.png', cv2.IMREAD_GRAYSCALE)
thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
binary_img = (thresh == 0).astype(np.uint8)

labels, num = ndi.label(binary_img, structure=np.ones((3, 3)))
sizes = np.bincount(labels.ravel())
sizes[0] = 0
largest_label = sizes.argmax()


regions = skimage.measure.regionprops(labels)
areas = [r.area for r in regions]
min_area = 0.25 * np.max(areas)
char_label = [r.label for r in regions if r.area > min_area]

if largest_label not in char_label:
    largest_label = max(char_label, key=lambda lab: sizes[lab])

original_img = cv2.imread('quote.png')
highlight = original_img.copy()
mask = (labels == largest_label)
highlight[mask, 0] = 0
highlight[mask, 1] = 0
highlight[mask, 2] = 255

plt.imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Largest character highlighted')
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, '6.png')
plt.savefig(save_path)
plt.show()
