import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

# Load the three datasets with error handling
books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1',
                    on_bad_lines='skip', low_memory=False)
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1',
                    on_bad_lines='skip')
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1')

# Replace zero ratings with mean rating of the book
# First, calculate mean rating for each book (excluding zeros)
book_mean_ratings = ratings[ratings['Book-Rating'] > 0].groupby('ISBN')['Book-Rating'].mean()

def replace_zero_with_mean(row):
    if row['Book-Rating'] == 0:              # If this rating is zero
        isbn = row['ISBN']                    # Get the book's ISBN
        if isbn in book_mean_ratings.index:   # If we calculated a mean for this book
            return book_mean_ratings[isbn]    # Grab the mean value from the group by object
        else:
            return 0                          # Keep as 0 (book has no real ratings)
    return row['Book-Rating']                 # Not zero? Keep original rating

# uses function and replaces 0 rating rows with means, if the row has a mean already, then it is skipped
ratings['Book-Rating'] = ratings.apply(replace_zero_with_mean, axis=1)

# Filter books that have been rated at least 200 times
book_counts = ratings.groupby('ISBN').size()
books_200plus = book_counts[book_counts >= 200].index

# Filter users that have rated at least 5 books
user_counts = ratings.groupby('User-ID').size()
users_5plus = user_counts[user_counts >= 5].index

filtered_ratings = ratings[(ratings['ISBN'].isin(books_200plus)) &
                           (ratings['User-ID'].isin(users_5plus))]

print(f'Filtered Ratings: {filtered_ratings.shape}') #compare the sizes of each df
print(f'Ratings: {ratings.shape}')

# User-book ratings matrix
ratings_matrix = filtered_ratings.pivot_table(
    index='User-ID',
    columns='ISBN',
    values='Book-Rating',
    fill_value=0
)

#dimensions of the matrix
print(f"\nRatings matrix shape: {ratings_matrix.shape}")
print(f"Rows (users): {ratings_matrix.shape[0]}")
print(f"Columns (books): {ratings_matrix.shape[1]}")

most_rated_books = filtered_ratings.groupby('ISBN').size().sort_values(ascending=False)
print(most_rated_books)
#Top 3 most rated books
#1) 0971880107, Wild Animus, 1686 ratings
#2) 0316666343, The Lovely Bones, 981 ratings
#3) 0385504209, The Da Vinci Code, 722 ratings

most_rating_users = filtered_ratings.groupby('User-ID').size().sort_values(ascending=False)
print(most_rating_users.head(3))
#Top 3 users with the most ratings
#1) 11673, 193 ratings
#2) 35859, 121 ratings
#3) 16796, 119 ratings

# ---- Clustering ----

k_values = [2, 4, 8, 16, 32, 64, 128]
inertias = []

for k in k_values:
    #print(f"Running k-means with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42) #use random_state to reproduce results
    kmeans.fit(ratings_matrix)
    inertias.append(kmeans.inertia_)
    #print(f"  Inertia: {kmeans.inertia_:.2f}")

# Plot the inertia scores (Elbow plot)
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method: Inertia vs Number of Clusters', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.savefig('plots/elbow_plot.png', dpi=300, bbox_inches='tight') # save to png
plt.show()

# (b) Choose optimal k
print("ANALYSIS: Choosing the optimal k")
for k, inertia in zip(k_values, inertias):
    print(f"  k={k:3d}: {inertia:,.2f}")

# Calculate rate of change to help identify elbow
rate_of_change = []
for i in range(1, len(inertias)):
    rate = (inertias[i - 1] - inertias[i]) / inertias[i - 1] * 100
    rate_of_change.append(rate)

print("\nPercentage decrease in inertia:")
for i, k in enumerate(k_values[1:]):
    print(f"  From k={k_values[i]} to k={k}: {rate_of_change[i]:.2f}%")

# From the data and the graph, it seems that k=32 or k=64 seem like a good choice

#The curve shows a fairly smooth, gradual decline without a sharp, obvious elbow.
# However: k=2 to k=16: The curve is steeper - you can see the line dropping more sharply k=32 to k=64:
# The curve starts to flatten out more noticeably k=64 to k=128:
# The curve continues to flatten - becomes almost linear The "elbow" appears to be around k=32
# to k=64 The graph suggests that: Before k=32: Still getting substantial improvements
# (steeper slope) Around k=32-64: The curve is starting to level off (the transition zone)
# After k=64: Diminishing returns are more evident (flatter)

# (c) Cluster with chosen k and find top-rated books per cluster
chosen_k = 32
print(f"Chosen k-value: {chosen_k}")

print(f"\nClustering users into {chosen_k} clusters...")
kmeans_final = KMeans(n_clusters=chosen_k, random_state=42)
cluster_labels = kmeans_final.fit_predict(ratings_matrix)

# Add cluster labels to the ratings matrix
ratings_matrix['Cluster'] = cluster_labels

print(f"\nCluster sizes:")
cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
for cluster_id, size in cluster_sizes.items():
    print(f"  Cluster {cluster_id}: {size} users")

print("TOP 3 HIGHEST-RATED BOOKS PER CLUSTER")

for cluster_id in range(chosen_k):
    print(f"\n{'=' * 60}")
    print(f"CLUSTER {cluster_id} ({cluster_sizes[cluster_id]} users)")
    print(f"{'=' * 60}")

    # Get users in this cluster
    cluster_users = ratings_matrix[ratings_matrix['Cluster'] == cluster_id]

    # Remove the Cluster column for calculations
    cluster_ratings = cluster_users.drop('Cluster', axis=1)

    # calculate mean rating for each book (excluding zeros)
    # Replace zeros with NaN for mean calculation
    cluster_ratings_no_zeros = cluster_ratings.replace(0, np.nan)
    mean_ratings = cluster_ratings_no_zeros.mean(axis=0)

    # Get top 3 books
    top_3_isbns = mean_ratings.nlargest(3)

    for rank, (isbn, avg_rating) in enumerate(top_3_isbns.items(), 1):

        book_info = books[books['ISBN'] == isbn]
        if not book_info.empty:
            title = book_info['Book-Title'].values[0]
            author = book_info['Book-Author'].values[0] if 'Book-Author' in book_info.columns else 'Unknown'
        else:
            title = 'Unknown Title'
            author = 'Unknown Author'

        #count how many users in cluster rated this book
        num_ratings = (cluster_ratings[isbn] > 0).sum()

        print(f"\n{rank}. {title}")
        print(f"   Author: {author}")
        print(f"   Average Rating: {avg_rating:.2f}")
        print(f"   Rated by: {num_ratings} users in this cluster")

# Save results
ratings_matrix.to_csv('clustered_users.csv')

# ----- Principal Component Analysis -----

# 3(a) Transpose the matrix and mean center
print("3(a) TRANSPOSE AND MEAN CENTER THE DATA")

# Remove 'Cluster' column if it exists from Q2
if 'Cluster' in ratings_matrix.columns:
    ratings_matrix = ratings_matrix.drop('Cluster', axis=1)

print(f"\nOriginal matrix shape: {ratings_matrix.shape}")
print(f"  Rows (users): {ratings_matrix.shape[0]}")
print(f"  Columns (books): {ratings_matrix.shape[1]}")

# Transpose: rows = books, columns = users
ratings_transposed = ratings_matrix.T

print(f"\nTransposed matrix shape: {ratings_transposed.shape}")
print(f"  Rows (books): {ratings_transposed.shape[0]}")
print(f"  Columns (users): {ratings_transposed.shape[1]}")

# Mean center the data (subtract mean of each column)
# Each column represents a user, so we center each user's ratings
ratings_centered = ratings_transposed - ratings_transposed.mean(axis=0)

print(f"\nMean-centered data shape: {ratings_centered.shape}")
print("Data has been mean-centered (each user's ratings centered around 0)")

# Verify centering
print(f"\nVerification - Column means after centering (should be ~0):")
print(f"  Max absolute mean: {abs(ratings_centered.mean(axis=0)).max():.10f}")

# 3(b) Apply PCA with k=2 components
print("3(b) APPLY PCA WITH k=2 COMPONENTS")

# Apply PCA
pca = PCA(n_components=2)
books_pca = pca.fit_transform(ratings_centered)

print(f"Original dimensions: {ratings_centered.shape[1]} (users)")
print(f"Reduced dimensions: {books_pca.shape[1]} (principal components)")
print(f"\nEach book is now represented by 2 values instead of {ratings_centered.shape[1]} values")

# Explained variance
print(f"\nExplained variance by each component:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)") # round by 2 places
print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
print(f"  Total: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

# Create DataFrame with PCA results
books_pca_df = pd.DataFrame(
    books_pca,
    columns=['PC1', 'PC2'],
    index=ratings_transposed.index
)

# (c) Plot results colored by genre/nationality
print("3(c) PLOT PCA RESULTS - COLORED BY USER NATIONALITY")

# Extract country from Location format: "city, state, country"
users['Country'] = users['Location'].str.split(',').str[-1].str.strip()

print(f"Found {users['Country'].nunique()} unique countries")

# Merge ratings with user country info
print("\nMerging ratings with user nationality data...")
ratings_with_country = filtered_ratings.merge(
    users[['User-ID', 'Country']],
    on='User-ID',
    how='left'
)

# For each book, find the most common country of users who rated it
def get_dominant_country(isbn):
    book_ratings = ratings_with_country[ratings_with_country['ISBN'] == isbn]
    if len(book_ratings) > 0:
        country_counts = book_ratings['Country'].value_counts()
        if not country_counts.empty:
            return country_counts.index[0] # returns highest counted country
    return 'Unknown'

# Merge PCA results with book info
#reset index to avoid error
books_pca_df_reset = books_pca_df.reset_index()
books_pca_df_reset.rename(columns={'index': 'ISBN'}, inplace=True)

books_with_pca = books_pca_df_reset.merge(
    books[['ISBN', 'Book-Title', 'Book-Author', 'Publisher']],
    on='ISBN',
    how='left'
)

#get dominant country for each book
books_with_pca['Dominant_Country'] = books_with_pca['ISBN'].apply(get_dominant_country)

print(f"\nCountry distribution of books:")
print(books_with_pca['Dominant_Country'].value_counts().head(15))

# Get top 10 most common countries for cleaner visualization
top_countries = books_with_pca['Dominant_Country'].value_counts().head(10).index
books_with_pca['Category_Plot'] = books_with_pca['Dominant_Country'].apply(
    lambda x: x if x in top_countries else 'Other'
)

#color map
unique_categories = books_with_pca['Category_Plot'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories))) # 11 unique colors
color_map = dict(zip(unique_categories, colors))

plt.figure(figsize=(14, 10))

for category in unique_categories:
    mask = books_with_pca['Category_Plot'] == category
    plt.scatter(books_with_pca[mask]['PC1'],
               books_with_pca[mask]['PC2'],
               c=[color_map[category]],
               label=category,
               alpha=0.6,
               s=50)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)', fontsize=12)
plt.title('PCA of Books (k=2) - Colored by Dominant User Nationality', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/pca_books_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# USA is the most dominant Country that has rated all of these books, so this PCA analysis does not tell us much.

# (d) Determine intrinsic dimensionality
print("(d) INTRINSIC DIMENSIONALITY")

# Apply PCA with all components to see variance explained
pca_full = PCA()
pca_full.fit(ratings_centered)

# Calculate cumulative variance explained
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components for 80% and 40% variance
n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1 # plus 1 because of zero indexing
n_components_40 = np.argmax(cumulative_variance >= 0.40) + 1

print(f"Total possible components: {len(pca_full.explained_variance_ratio_)}")
print(f"\nComponents needed to explain 40% of variance: {n_components_40}")
print(f"  Actual variance explained: {cumulative_variance[n_components_40-1]*100:.2f}%")
print(f"\nComponents needed to explain 80% of variance: {n_components_80}")
print(f"  Actual variance explained: {cumulative_variance[n_components_80-1]*100:.2f}%")

# Plot cumulative variance explained
plt.figure(figsize=(12, 6))
plt.plot(range(1, min(101, len(cumulative_variance)+1)),
         cumulative_variance[:100],
         'bo-', linewidth=2, markersize=4)
plt.axhline(y=0.40, color='orange', linestyle='--', linewidth=2, label='40% variance')
plt.axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='80% variance')
plt.axvline(x=n_components_40, color='orange', linestyle=':', alpha=0.5)
plt.axvline(x=n_components_80, color='red', linestyle=':', alpha=0.5)
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Cumulative Variance Explained', fontsize=12)
plt.title('PCA: Cumulative Variance Explained', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/pca_variance_explained.png', dpi=300, bbox_inches='tight')
plt.show()


# Comparison with k=2
print("COMPARISON WITH k=2")
print(f"\nWith k=2 components (used in part c):")
print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
print(f"\nTo explain 40% of variance, we need: {n_components_40} components")
print(f"To explain 80% of variance, we need: {n_components_80} components")

# Save results
#books_with_pca.to_csv('books_pca_results.csv', index=False)
#print("\nPCA results saved to 'books_pca_results.csv'")

# ------ Singular Value Decomposition (Question 4)

# Remove 'Cluster' column if it exists from Q2
if 'Cluster' in ratings_matrix.columns:
    ratings_matrix_clean = ratings_matrix.drop('Cluster', axis=1)
else:
    ratings_matrix_clean = ratings_matrix.copy()

print("SVD ANALYSIS - SINGULAR VALUE DECOMPOSITION")
print(f"\nOriginal matrix shape: {ratings_matrix_clean.shape}")
print(f"  Rows (users): {ratings_matrix_clean.shape[0]}")
print(f"  Columns (books): {ratings_matrix_clean.shape[1]}")
print("\nNote: Using original matrix (rows=users, columns=books, NO mean-centering)")

# (a) Apply SVD with k=128 components
print(f"\n4(a) APPLY SVD WITH k=128 COMPONENTS")

svd_128 = TruncatedSVD(n_components=128, random_state=42)
svd_128.fit(ratings_matrix_clean)

# Get singular values
singular_values = svd_128.singular_values_

print(f"\nSVD with k=128 complete!")
print(f"Number of singular values: {len(singular_values)}")

# Plot singular values
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(singular_values)+1), singular_values, 'bo-', linewidth=2, markersize=4)
plt.xlabel('Component Index', fontsize=12)
plt.ylabel('Singular Value', fontsize=12)
plt.title('Singular Values (k=128)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/svd_singular_values.png', dpi=300, bbox_inches='tight')
plt.show()

# Show first 10 singular values
print("\nFirst 10 singular values:")
for i, sv in enumerate(singular_values[:10], 1):
    print(f"  {i}: {sv:.2f}")

# (b) Compare with k-means results from Q2
print(f"\n4(b) EXPLAINED VARIANCE RATIO FOR DIFFERENT k VALUES")

k_values = [2, 4, 8, 16, 32, 64, 128]
svd_results = {}

print("\nCalculating explained variance for each k...")

for k in k_values:
    if k <= min(ratings_matrix_clean.shape):
        svd_k = TruncatedSVD(n_components=k, random_state=42)
        svd_k.fit(ratings_matrix_clean)
        total_variance = svd_k.explained_variance_ratio_.sum()
        svd_results[k] = total_variance
        print(f"  k={k:3d}: {total_variance:.4f} ({total_variance*100:.2f}%)")

#plot explained variance vs k
plt.figure(figsize=(10, 6))
k_list = list(svd_results.keys())
variance_list = list(svd_results.values())

plt.plot(k_list, variance_list, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Components (k)', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.title('SVD: Explained Variance vs Number of Components', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(k_list)
plt.tight_layout()
plt.savefig('plots/svd_explained_variance.png', dpi=300, bbox_inches='tight')
plt.show()

# (c) Apply SVD with k=2 and transform the data
print(f"\n4(c) APPLY SVD WITH k=2 AND TRANSFORM DATA")

svd_2 = TruncatedSVD(n_components=2, random_state=42)
users_svd = svd_2.fit_transform(ratings_matrix_clean)

print(f"\nSVD transformation complete!")
print(f"Original dimensions: {ratings_matrix_clean.shape[1]} (books)")
print(f"Reduced dimensions: {users_svd.shape[1]} (components)")
print(f"\nEach user is now represented by 2 values instead of {ratings_matrix_clean.shape[1]} values")

# Explained variance for k=2
print(f"  Component 1: {svd_2.explained_variance_ratio_[0]:.4f} ({svd_2.explained_variance_ratio_[0]*100:.2f}%)")
print(f"  Component 2: {svd_2.explained_variance_ratio_[1]:.4f} ({svd_2.explained_variance_ratio_[1]*100:.2f}%)")
print(f"  Total: {svd_2.explained_variance_ratio_.sum():.4f} ({svd_2.explained_variance_ratio_.sum()*100:.2f}%)")

#DataFrame with SVD results
users_svd_df = pd.DataFrame(
    users_svd,
    columns=['SVD1', 'SVD2'],
    index=ratings_matrix_clean.index
)

# (d) Plot results colored by cluster membership from Q2
print(f"\n4(d) PLOT SVD RESULTS COLORED BY K-MEANS CLUSTERS")

#Load cluster assignments from Q2
clustered_data = pd.read_csv('clustered_users.csv', index_col=0)

#Merge SVD results with cluster labels
users_svd_df['Cluster'] = clustered_data['Cluster']

print(f"\nLoaded cluster assignments from Q2")
print(f"Number of clusters: {users_svd_df['Cluster'].nunique()}")

#color map
unique_clusters = sorted(users_svd_df['Cluster'].unique())
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
color_map = dict(zip(unique_clusters, colors))

plt.figure(figsize=(14, 10))

for cluster in unique_clusters:
    mask = users_svd_df['Cluster'] == cluster
    plt.scatter(users_svd_df[mask]['SVD1'],
               users_svd_df[mask]['SVD2'],
               c=[color_map[cluster]],
               label=f'Cluster {cluster}',
               alpha=0.6,
               s=30)

plt.xlabel(f'SVD Component 1 ({svd_2.explained_variance_ratio_[0]*100:.2f}% variance)', fontsize=12)
plt.ylabel(f'SVD Component 2 ({svd_2.explained_variance_ratio_[1]*100:.2f}% variance)', fontsize=12)
plt.title('SVD Projection (k=2) - Colored by K-Means Cluster', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/svd_clusters_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Save SVD results
users_svd_df.to_csv('users_svd_results.csv')