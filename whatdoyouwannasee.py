# learner for implementing neighbor searches
from sklearn.metrics.pairwise import cosine_similarity
import ast  # abstract syntax trees
from sklearn.neighbors import NearestNeighbors
# convertion of raw documents to a matrix of TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd  # data manipulation
import pickle  # serializing and de-serializing a Python object structure


# read the datasets
movies = pd.read_csv('Dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('Dataset/tmdb_5000_credits.csv')


# merge the two datasets on the 'title' column and show the before and after shapes
movies = movies.merge(credits, on='title')


# decide on the features that are important for the project, remove the rest
movies = movies[['movie_id', 'vote_average', 'title',
                 'overview', 'genres', 'keywords', 'cast', 'crew']]


# detect missing values
movies.isnull().sum()


# drop missing values
movies.dropna(inplace=True)


# identify duplicate rows
movies.duplicated().sum()


# convert the string to list
def convert(text):
    list = []
    for i in ast.literal_eval(text):
        list.append(i['name'])
    return list


# apply the function to required features
features = ['genres', 'keywords', 'cast']
for feature in features:
    movies[feature] = movies[feature].apply(convert)


# get the director's name from a feature
def get_director(text):
    list = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            list.append(i['name'])
    return list


# apply the function to 'crew' and create a new column called 'director'
movies['director'] = movies['crew'].apply(get_director)

# remove 'crew'
movies = movies.drop('crew', axis=1)


# convert 'overview' to list form
movies['overview'] = movies['overview'].apply(lambda x: x.split())


# merge all the columns inside of a new column called 'tags'
movies['tags'] = movies['overview'] + movies['genres'] + \
    movies['keywords'] + movies['cast'] + movies['director']

# create a new dataset containing only id, title and tags
newDF = movies.drop(columns=['overview', 'genres',
                    'keywords', 'cast', 'director'])

# combine all the tags into a single string
newDF['tags'] = newDF['tags'].apply(lambda x: " ".join(x))

# convert 'tags' to a matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newDF['tags'])


# Create a new instance
new_knn = NearestNeighbors(n_neighbors=5, metric='cosine')
new_knn.fit(X)

# Save the knn model
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(new_knn, file)

# Load the knn model
loaded_knn = pickle.load(open('knn_model.pkl', 'rb'))

all_keywords = [""]

query = vectorizer.transform([' '.join(all_keywords)])
distances, indices = loaded_knn.kneighbors(query)
top_movies_knn = newDF.iloc[indices[0]][['title', 'vote_average']]
top_movies_knn = top_movies_knn.sort_values(by='vote_average', ascending=False)
print(top_movies_knn['title'])


# Convert 'all_keywords' to a TF-IDF vector
query_vector = vectorizer.transform([' '.join(all_keywords)])

# Calculate cosine similarity between the query and all movie tags
cosine_similarities = cosine_similarity(query_vector, X)

# Get the indices of top movies based on similarity
top_movie_indices = cosine_similarities.argsort()[0][::-1][:5]

# Retrieve top movies from the original dataframe
top_movies = newDF.iloc[top_movie_indices][['title', 'vote_average']]

print(top_movies['title'])

ground_truth = top_movies_knn['title']

# Extract the titles of recommended movies
recommended_movies = top_movies['title'].tolist()

# Calculate True Positives (intersection of recommended and ground truth)
true_positives = len(set(recommended_movies) & set(ground_truth))

# Calculate False Positives (recommended movies not in ground truth)
false_positives = len(set(recommended_movies) - set(ground_truth))

# Calculate Precision
precision = true_positives / (true_positives + false_positives)

print(f'Precision: {precision}')
