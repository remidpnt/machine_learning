# How to use Glove to create a lookup matrix from text files
<br/>
First, we need to get the Glove file containing each vector corresponding to a token, from the page https://nlp.stanford.edu/projects/glove/, unzipe in and rename it glove.txt


```console
wget $your_glove_file_servor_location  #$your_glove_file_servor_location = http://nlp.stanford.edu/data/glove.840B.300d.zip in my case
unzip $your_file_name
mv $your_file_name glove.txt
```
In a terminal, to separate our worlds to our corresponding vectors, type:

```console
awk '{print $1;}' glove.txt > motAwk.txt
awk '{for(i=2;i<=NF;++i) printf "%s " ,$i ; print("")}' glove.txt > vectAwk.txt #sometimes without ; print(""), depends of the server
split -l 550000 vectAwk.txt ### need to split it to load it in memory
```
then use "python vectors.py" to turn files to .npy file, and then we load using mmap option to save RAM, we concatenate and finally we add 2 lines at he end (average). Save as vectors.npy


We will then load out text files we want to preprocess. Here, we will use stanford Imdb movie review from http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz

```console
wget http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz
```
