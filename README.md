
<p align="center" width="100%">
    <img width="25%" src="https://github.com/hadjuse/Spotify-recommendation-KNN/blob/main/images/logo.png">
</p>


# Spotify-classification-song-KNN
Here is a project that I made which contains graphics and analysis of a dataset that I found about songs and the caracteristic of them.
The aim of the project is to make prediction of what modes (major or minor) belong a sampled songs using 2 caracteristics like danceability and rythm.

## Summaries:
  1. **[Data cleaning](#Data-cleaning)**
  2. **[Data selection](#Data-selection)**
  3. **[Interpretation, Classification and plot](#ploty)**
        - [Interpretation](#i)
        - [Classification Data](#c)
  4. **[training model](#train)**
  5. **[Visualisation of the results](#result)**
  
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
![](images/popularity.png)

This one is about the the technical's details on the songs
```python
song_information[["acousticness", "liveness", "valence"]].plot.bar(figsize=(10,7))
```
![](images/alv.png)

# <a name="i"><a/>Interpretation
We can give an interpretation about these two graphics. 
- The first graphic shows the popularity of each song that we pick up.
- The second graphic shows differents bar which represent (based on 3 caracteristic) the efficiency of each song.


Let's explain these 3 caracteristics: 
1. **acousticness**:

    - Informs the probability of a song to be acoustic or not.


2. **liveness**:

    - Detects the presence of an audience in a song.       
    The higher the liveness value, the higher the        
    probability of a song being performed live.
    
    
3. **valence**:

    - Describes the positiveness within a song.          
      High valence values represent happier songs,        
      whereas low values characterize the opposite.

Finally we can assert that the most popular song on this subdata, which is *côte ouest* **n°17**, is not a liveness song but he's quite **acoustic** and has an average **valence**.
And the less popular songs on this selection has a less probability of acousticsness.
# <a name="c">Classification of the songs
This section is to classify the different song according to their:

1. **danceability**:
    - Combines tempo, rhythm and other elements         
      to inform if a song is suitable for dancing.

2. **mode**:
    - If the song is in the minor key or major key.
3. **energy**:
    - Represents the intensity and activity of a song by     
      combining information such as dynamic range, perceived   
    loudness, timbre, onset rate, and general entropy.
