def detect_noise(image_path, threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    r = 200  # Radius for the circular region
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 1

    f_transform_shifted *= mask

    f_ishift = np.fft.ifftshift(f_transform_shifted)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    noise = np.abs(image - img_back)
    average_noise = np.mean(noise)
    is_noisy = average_noise > threshold
    return is_noisy, average_noise

for img_path in img_paths:
    start = time.time()
    is_noisy, noise_level = detect_noise(img_path, threshold=10)
    image_name = os.path.basename(img_path)
    stop = time.time()
    if is_noisy:
        print('Time: ', stop - start)
        print(f"The image {image_name} is noisy. Noise level is {noise_level:.2f}")
    else:
        print('Time: ', stop - start)
        print(f"The image {image_name} is not noisy. Noise level is {noise_level:.2f}")

