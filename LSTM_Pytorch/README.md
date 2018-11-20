# NLP model: Unified language model fine tunning

In this folder, you will find the python code to preprocess data and pretrain a 3 layers AWD-LSTM.
Make sure to have a look to my related blogpost: https://remidpnt.github.io/NLP_transfer.html

* Tokenization.ipynb: Creating a string to integer dictionary to tokenize Wikitext103 data, IMDB data, and transfering the pretrained 
embedding matrices to use on IMDB dataset (each corpus has a different string to integer dictionary).
* python_scripts/AWD_LSTM_model.py My model, described in my blogpost. 
* python_script/pretraining_wikitext103.py Pretraining our model using slangted triangular learning rate 
* python_script/pretraining_imdb.py Transfer learnign part, using different learning-rates per layes, plus gradualy unfreezing
