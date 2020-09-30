# Bias Invariant RNA-Seq Annotation

EDIT ABSTRACT

Recent technological advances have resulted in an unprecedented increase in publicly available biomedical data, yet the reuse of the data is often precluded by experimental bias and a lack of annotation depth and consistency. Here we investigate RNA-seq metadata prediction based on gene expression values. We present a deep-learning-based domain adaptation algorithm for the automatic annotation of RNA-seq metadata. We show how this algorithm outperforms existing approaches as well as traditional deep learning methods in the prediction of tissue, sample source, and patient sex information across several large data repositories. By using a model architecture similar to Siamese Networks the algorithm is able to learn biases from data sets with few samples, a situation that closely resembles the current state of data repositories. Our domain adaptation approach achieves metadata annotation accuracies up to 27.4% better than previously published methods. Further we show experimental results indicating the superiority of pooling small and diverse RNA-seq studies versus large homogenous data sets for model training. Lastly, we provide a list of more than 12,000 novel tissue and sex label annotations for 9,283 unique SRA samples.  

LINK TO PREPRINT

# Study Overview

![Overview](image/main_overview_study.png)


# Domain Adaptation Model

![DA Model](image/main_overview_da_model.png)

