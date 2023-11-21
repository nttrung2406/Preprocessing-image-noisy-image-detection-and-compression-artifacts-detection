# Preprocessing image noisy image detection and compression artifacts detection without employing DNN

a) Noisy image detection using Fast Fourier Transform

- Key idea is to take the denoise image frm an original image. After that the difference between normal images and their denoise images will be the threshold to make the comparison

b) Compression artifacts detection

Compression artifacts include: 
  
- Blockiness detection: JPEG compression divides an image into 8x8 blocks and applies a discrete cosine transform (DCT) to each block. This can cause blockiness artifacts, which can be detected by analyzing the variance of the image across blocks.

- Blocking effect: JPEG compression quantizes the DCT coefficients, which can cause blocking artifacts. This can be detected by analyzing the distribution of DCT coefficients.

- Ringing artifacts: JPEG compression can introduce ringing artifacts around sharp edges. These can be detected by analyzing the local gradient of the image.
