
<p align="center" width="100%">
    <img width="25%" src="https://github.com/hadjuse/Spotify-recommendation-KNN/blob/main/images/logo.png">
</p>


# Spotify-classification-song-KNN
Here is a project that I made which contains graphics and analysis of a dataset that I found about songs and the caracteristic of them.
The aim of the project is to make prediction of what modes (major or minor) belong a sampled songs using 2 caracteristics like danceability and rythm.

## Summaries:
  1. **[Data cleaning](#Data-cleaning)**
  2. **[Data selection](#Data-selection)**
  3. **[Interpretation and plot](#ploty)**
  4. **training model**
  5. **Visualisation of the results**
  
 # <a name = "Data-cleaning"></a>Data cleaning
 First of all, we import all necessary libraries:
  ```python
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
 ```
 Nothing to do here because the data is organized and clean. So we use this code below:
 ```python
 data = pd.read_csv("hit_songs/Hit Songs/spotify_hits_dataset_complete.csv", sep='\t', parse_dates=True)
print(data.shape)
data.head()
 ```
 # <a name = "Data-selection"></a>Data selection
The interesting datas are : Song_name, artist name, and the different information about the song, such as the key of the song or bpm.
Code:
```python
song_information = data.iloc[0:25, 12:23]
```
# <a name = "ploty"></a>Interpretation of the datas
Lets plot some information like the popularity of the first 25 songs:
```python
artist_information["popularity"].plot.bar()
```

