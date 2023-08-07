import os.path
import json
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
accuracy = evaluate.load("accuracy")

# References - Resources
# Hugging Face BERT Tutorial - https://huggingface.co/docs/transformers/tasks/sequence_classification

def preprocess_comments(examples):

    return tokenizer(examples['comment'], truncation = True)

def coompute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    # DATA PREPROCESSING

    wd_path = os.path.abspath(os.path.dirname(__file__))

    f = open(wd_path+'/interval_problems.json')

    data = json.load(f)

    f.close()

    # get unique values target

    label_set = set([i['problem_type'] for i in data])

    print(label_set)

    #tokenized_comments = map(preprocess_comments, data)

    #print(tokenized_comments)

    #data_collator = DataCollatorWithPadding(tokenizer=tokenizer)






