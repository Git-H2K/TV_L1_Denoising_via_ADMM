clc; clear; 

Ori_Img = rgb2gray(imread('Test your image'));
Noise_Img = imnoise(Ori_Img, 'gaussian', 0, 0.001);
Result = TV_L1_ADMM(Noise_Img, 1, 1e-3, 100);

figure; subplot(121); imshow(Noise_Img); title('Noise Image')
subplot(122); imshow(Result); title('Reconstructed Image')