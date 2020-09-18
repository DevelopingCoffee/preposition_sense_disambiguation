from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
import numpy as np
import csv



class BertTagger:
    def __init__(self):
        self.definitions = self.read_definitions("definitions.tsv")

        model_name = 'bert-base-uncased'
        config = BertConfig.from_pretrained(model_name)
        config.output_hidden_states = False
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
        self.model = BertForSequenceClassification.from_pretrained("model_save")

    def read_definitions(self, path):
        with open('definitions.tsv') as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            defs = dict(reader)
        return defs

    def tag(self, pretokenized_sentence):

        data = self.tokenizer(
            text=[pretokenized_sentence],
            add_special_tokens=True,
            max_length=100,
            truncation=True,
            padding=True, 
            return_tensors='pt',
            return_token_type_ids = False,
            verbose = True)

        logits = self.model(data['input_ids'], token_type_ids=None, attention_mask=data['attention_mask'])[0].detach().numpy()
        prediction = self.definitions[str(np.argmax(logits))]
        print("sentence : {}\nprediction : {}\n\n".format(pretokenized_sentence, prediction))
        return prediction 


tagger = BertTagger()
# IN
#tagger.tag("I am <head>in</head> big trouble")
#tagger.tag("I am <head>in</head> a big airplane")
#tagger.tag("I am <head>in</head> New York")
#tagger.tag("<head>In</head> 2020 Donald Trump will be re-elected")
#tagger.tag("I always see you <head>in</head> my dreams.")
#tagger.tag("I am speaking <head>in</head> portuguese.")
#tagger.tag("Donald Trump appears <head>in</head> a weird manner.")

# BY
tagger.tag("<head>By</head> 2021 I will have a bachelor degree.")
tagger.tag("<head>By</head> the way, Joe Biden is actually better than is Opponent.")
tagger.tag("The Crisis was handled <head>by</head> the best president ever - Donald Trump.")
tagger.tag("He is leading the polls <head>by</head> far.")

# WITH
#tagger.tag("He is swimming <head>with</head> his hands.")
#tagger.tag("He is <head>with</head> his parents.")
#tagger.tag("She blinked <head>with</head>  confusion.") # Manner
#tagger.tag("He combines professionalism <head>with</head>  humor.") # Accompanier
#tagger.tag("He washed a small red teacup <head>with</head>  water.")  # Means

# ON
#tagger.tag("The shop is <head>on</head> the left.")
#tagger.tag("My friend is <head>on</head> the way to Moscow.")
#tagger.tag("When she was a little girl people saw unrealistic cowboy films <head>on</head> television")
#tagger.tag("I am <head>on</head> my way to New York.")
#tagger.tag("I am <head>on</head> massive pressure.")
