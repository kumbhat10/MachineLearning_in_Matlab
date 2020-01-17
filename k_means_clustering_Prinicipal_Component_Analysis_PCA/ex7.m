%% Initialization
clear ; close all; clc
%% ================= Part 1: Find Closest Centroids ====================

load('ex7data2.mat');

% Select an initial set of centroids
K = 3; % 3 Centroids
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples using the
idx = findClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');

%% ===================== Part 2: Compute Means =========================
%  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ]\n\n');

%% =================== Part 3: K-Means Clustering ======================

load('ex7data2.mat');
K = 3; % Settings for running K-Means
max_iters = 10;

% initial_centroids = [3 3; 6 2; 8 5];
initial_centroids = kMeansInitCentroids(X,K); %%generate random initialized centroids
% Run K-Means algorithm. The 'true' at the end tells our function to plot the progress of K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');

%% ============= Part 4: K-Means Clustering on Pixels ===============
% It is like converting whole image to 16 bit image 
% as clusters of 16 closest most occuring group of colours 
fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

%  Load an image of a bird
A = double(imread('bird_small.png'));% If imread does not work try instead load ('bird_small.mat');
A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

img_size = size(A);% Size of the image
% Reshape the image into an Nx3 matrix where N = number of pixels.
X = reshape(A, img_size(1) * img_size(2), 3);

% Run K-Means algorithm on this data
% try different values of K and max_iters here
K = 16; 
max_iters = 10;

% When using K-Means, it is important the initialize the centroids randomly. 
%  complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

% %% ================= Part 5: Image Compression ======================

fprintf('\nApplying K-Means to compress an image.\n\n');
% Find closest cluster members
idx = findClosestCentroids(X, centroids);
X_recovered = centroids(idx,:); % Recover the image from the indices (idx) by mapping each pixel (specified by its index in idx) to the centroid value
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);% Reshape the recovered image into proper dimensions

% Display the original image 
subplot(1, 3, 1);
imagesc(A); 
title('Original');

% Display compressed image side by side
subplot(1, 3, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));

subplot(1, 3, 3);
imagesc(round(A))
title('Rounded to 0 or 1 ')
