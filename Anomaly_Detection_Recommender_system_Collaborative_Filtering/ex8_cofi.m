% Anomaly Detection and Collaborative Filtering
%% =============== Part 1: Loading movie ratings dataset ================

fprintf('Loading movie ratings dataset.\n\n');

%  Load data
load ('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i

%  From the matrix, compute statistics like average rating.
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', mean(Y(1, R(1, :))));

% "visualize" the ratings matrix by plotting it with imagesc
imagesc(Y);
ylabel('Movies');
xlabel('Users');

%% ============ Part 2: Collaborative Filtering Cost Function ===========
%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
load ('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
U = 4; M = 5; F = 3;
X = X(1:M, 1:F);
Theta = Theta(1:U, 1:F);
Y = Y(1:M, 1:U);
R = R(1:M, 1:U);

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, U, M, ...
               F, 0);
           
fprintf(['Cost at loaded parameters: %f '...
         '\n(this value should be about 22.22)\n'], J);

%% ============== Part 3: Collaborative Filtering Gradient ==============

fprintf('\nChecking Gradients (without regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction;

%% ========= Part 4: Collaborative Filtering Cost Regularization ========

J = cofiCostFunc([X(:) ; Theta(:)], Y, R, U, M, ...
               F, 1.5);
           
fprintf(['Cost at loaded parameters (lambda = 1.5): %f '...
         '\n(this value should be about 31.34)\n'], J);
     
%% ======= Part 5: Collaborative Filtering Gradient Regularization ======

fprintf('\nChecking Gradients (with regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction(1.5);

%% ============== Part 6: Entering ratings for a new user ===============
%  Before training the collaborative filtering model, first
%  add ratings that correspond to a new user just observed. This
%  part of the code will also allow to put in own ratings for the
%  movies in the dataset!

movieList = loadMovieList();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in  dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", set
my_ratings(1) = 4;

% Or suppose did not enjoy Silence of the Lambs (1991), set
my_ratings(98) = 2;

% selected a few movies liked / did not like and the ratings given are as follows:
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

%% ================== Part 7: Learning Movie Ratings ====================
%  Now, train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users

fprintf('\nTraining collaborative filtering...\n');

%  Load data
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
U = size(Y, 2); %% users
M = size(Y, 1); %% movies
F = 10;         %% features 
 
% Set Initial Parameters (Theta, X)
X     = randn(M, F);   %% MxF
Theta = randn(U, F);   %% UxF

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, U, M, ...
                                F, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X     = reshape(theta(1:M*F)    , M, F);
Theta = reshape(theta(M*F+1:end), U, F);

fprintf('Recommender system learning completed.\n');

%% ================== Part 8: Recommendation for you ====================
%  After training the model, Now make recommendations by computing predictions matrix.

p = X * Theta';   %% MxU
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), movieList{j});
end

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end
