import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv("SoldPropertiesData.csv")

# Randomly select 4000 samples
random_samples = data.sample(n=4000, random_state=42)

# Extract coordinates for selected samples
coordinates = random_samples[['latitude', 'longitude']].values

# Calculate the 100 nearest neighbors for each point
n_neighbors = 100
neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(coordinates)
distances, indices = neighbors.kneighbors(coordinates)

# Calculate averages for the selected samples' nearest neighbors
averages = []
for i, sample in enumerate(random_samples.iterrows()):
    neighbor_indices = indices[i, 1:]  # Skip the first index, which is the sample itself
    neighbor_data = data.iloc[neighbor_indices]
    avg_rooms = neighbor_data['rooms'].mean()
    avg_floor = neighbor_data['floor'].mean()
    avg_price = neighbor_data['price'].mean()
    averages.append({'avg_rooms': avg_rooms, 'avg_floor': avg_floor, 'avg_price': avg_price})

# Add average values to the DataFrame
random_samples = random_samples.assign(**pd.DataFrame(averages))

# Display the DataFrame with added averages
print(random_samples)
