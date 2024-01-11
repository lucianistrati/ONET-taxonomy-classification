# MEMO / Documentation

## Introduction

ONET classification - classifying jobs into an ONET taxonomy based on their job 
descriptions and job titles

## Datasets analysis

- Train:
  - Number of datapoints: 17927
  - Text length (min, max, mean, median): 6 31785 3991.17 3423

- Test:
  - Number of datapoints: 19394
  - Text length (min, max, mean, median): (2 603 33.48 28)

## Evaluation and Metrics

### Used
- F1 score
- Accuracy
- Precision
- Recall

### Observations:
- in computing precision, recall and f1 score, and average weighted was used
- N/top_K is used throughout the code and in the documentation  as the 
  number of the first ONET labels which will be assigned for each job description


## Experiments and results

Preprocessing methods:
- Extract from the job descriptions only the words that appear in the ONET taxonomy 
vocabulary and then embed that 
- Leaving out the job descriptions and the job titles as they as they are and then 
  using only the job description versus only the job title

Feature extraction methods:
- Embedding using sentence transformers with the suggested option: 
  sentence-transformers/all-MiniLM-L6-v2;
- Also embedding with this option of sentence transformers: 
  'sentence-transformers/all-mpnet-base-v2'. "all-mpnet-base-v2" was used because it
  has a higher max sequence length, a higher number of dimensions and also it performs
  better than "all-MiniLM-L6-v2" on both of the benchmarks they used for comparison
  at SBERT.NET.

Classifying methods:
- Logistic Regression as a baseline to choose between 6 types of encodings of the data;
- XGBoost Classifier, SVC and Fully connected neural network classifier all compared 
  only what Logistic Regression picked as most successful;
- Using a Vector DB in order to just match each job description / job titles to its 
  most similar label;
- Using a GPT 3.5 model with 2 different prompt designs in order to predict the top N 
  ONET classes for each based on:
  - what the model knows about the ONET classes from its own training data;
  - using a RAG-like mechanism where the model was told to select from the top 10 ONET 
    labels that are the most similar semantically to that job description / title;


### Important limitation to consider

- "all-MiniLM-L6-v2" has a limitation of 256 tokens for the max sequence length
- "all-mpnet-base-v2" has also a limitation of 384 tokens for the max sequence length
- another reason why "all-mpnet-base-v2" was used was also this one of max sequence 
  length as it was a common sense assumption that with seeing an extra of 128 tokens 
  from the job descriptions would help the model (since all job titles were less than 
  256 tokens and there it was not an issue)
- of course this is not perfect either because the job descriptions are much longer 
  than that, and ultimately the best way to do it would be to either apply a 
  summarization of the job description with say Open AI in order to shorten and make 
  the information about the job more robust and then on top of that to apply the 
  sentence transformer embedding
- another way to do it would be a mean embedding of all the chunks of 256/384 tokens 
  over the job description (like in a sliding window, but without overlap)


## Results

### Comparison of 6 methods with Logistic Regression

- Files extension: _BODY
  - Accuracy: 0.33570834200124816
  - Precision: 0.46952535649800314
  - Recall: 0.33570834200124816
  - F1: 0.3226329093236435

- Files extension: _BODY_mpnet
  - Accuracy: 0.38423132931142084
  - Precision: 0.4967612124515013
  - Recall: 0.38423132931142084
  - F1: 0.3678004774288258

- Files extension: _BODY_oov
  - Accuracy: 0.22794882463074684
  - Precision: 0.3560756029833411
  - Recall: 0.22794882463074684
  - F1: 0.22064958308691923

- Files extension: _TITLE_RAW
  - Accuracy: 0.5256396921156646
  - Precision: 0.649622214404331
  - Recall: 0.5256396921156646
  - F1: 0.5243902015487711

- Files extension: _TITLE_RAW_mpnet
  - Accuracy: 0.49875182026211773
  - Precision: 0.6349772230084788
  - Recall: 0.49875182026211773
  - F1: 0.49732690421172454

- Files extension: _TITLE_RAW_oov
  - Accuracy: 0.2175993343041398
  - Precision: 0.3565104090126388
  - Recall: 0.2175993343041398
  - F1: 0.22862063404987978
  
So based on all the metrics (accuracy, precision, recall and F1 score) the 
  classification using the title encoded 'sentence-transformers/all-MiniLM-L6-v2' 
  sentence transformer with disabled out of vocabulary filter and used on the job 
  titles instead of job descriptions worked the best. 


### Next steps
Next possible improvements tested were an XGBOOST, an SVC and a Neural Network.

- XGB:
- Files extension: _TITLE_RAW
- Accuracy: 0.51752652381943
- Precision: 0.5225985569151481
- Recall: 0.51752652381943
- F1: 0.5058228699132713

XGBoost was comparable to Logistic Regression.

- SVC:
- Files extension: _TITLE_RAW
- Accuracy: 0.6129082587892657
- Precision: 0.6501723055080944
- Recall: 0.6129082587892657
- F1: 0.6165664593236386

We see that with an SVC the accuracy and F1 improved by roughly 10%.
But when comparing it to the Fully connect Neural network we were able to increase 
the test accuracy to 64.26%, a significant improvement of almost 3% compare to 
the best classical ML model which got to 61%. The F1 score, recall and precision
were also higher with the NN.

We will now compare that against using a pinecone vector database to get the most 
similar ONET label for each job title. It was better than 
random guess, but very weak performance nonetheless. The overall accuracy was around 
1-2% on a couple of hundreds of examples, so it was abandoned.

Interestingly enough when using just a GPT 3.5 without any information about the 
current ONET taxonomy (so only what it knew about ONET from its own training) the 
accuracy was 18% on also a couple of hundreds of samples (due to time constraints 
there was no way to be able to compare that for the entire test set).

Then, an improvement to this approach could have been to see if using the Vector DB 
and retrieving the top 10 most similar options could be of help in order to "guide" 
the GPT 3.5 towards the right answer, but the accuracy was only around 3-4%.

## Further developments and improvements

These are some possible improvements for the future
- different methods of doing the embeddings, such as Ada Embeddings API from Open AI;
- different feature extraction by extracting some keywords which could be relevant 
  to the problem of classification and embed only those;
- gaining more data from other sources;
- do a train-test split based not only stratified based on the labels, but also 
  stratified on the length of the job descriptions as the current split has job 
  descriptions of very different lengths;
- gaining some more insights and domain knowledge regarding the structure and 
  potentially the hierarchy of this ONET taxonomy. One way to do that would be to 
  check that the first N labels occupations have all the same minor group, 
  major group, broad occupation or SOC level as that should show more robustness
  (since it makes sense for the first N occupations to be very similar between one 
  another);
- changes to the architecture of the neural network;
- hyperparameter finetuning for the SVC;

### Ideas of metrics for the future
- Correlation between job descriptions / titles text similarity and the predicted 
  labels text similarity (i.e. for 2 datapoints how similar are their 
  descriptions/titles on one side and also how similar are the predictions);
- Mean semantical similarity between the first predicted label and the other N - 1 
predicted labels (since the labels should be quite related between one another), or 
  pairwise against all the possible pairs of the first N labels;
- Mean semantical similarity between the actual label and the other N predicted labels;

## Conclusion

In conclusion, the results obtained for this classification problem are satisfactory 
given the time constraints. There is definitely a lot of room for improvement with 
allocating more time to this problem.
