# preposition_sense_disambiguation

## Data
Download the data from http://www.statmt.org/europarl/.    
Use a tokenizer, make the letters lower case, create a single corpus of 2 languages and clean all empty rows.
I used the cdec word aligner for this (as proposed in the original paper) https://github.com/redpony/cdec but this repo might not be the original source.    
The repo contains a small bug so you can't compile the code for any questions you can send me an email to tobias.marzell@gmail.com.

## The models
I used 3 different approaches for the unsupervised part.   
- A bidirectional LSTM trained from scratch
- A transformers pretrained model that I used for transfer learning
- A pretrained bert model that I used for transfer learning

Check out the included scripts, as they are straight forward.


To make a prediction you can use the script transformerspredict.py.

## Whats still do be done

Implement the output into a supervised model.    
Try different models.
