Thank you for the comment. Please refer to our response below:

> [...]I would encourage the authors to make an effort so that these additional discussions and clarification pointers make it to the main manuscript? [...]

We already incorporated all the review comments from the 1st round that space permits. We will collect more feedback from reviewers and modify
the paper accordingly. Please refer to Appendix A.7.3 for the comparison with the prototype-based methods.

> [...] I am overlooking something, it is not clear to me how the connection is made with respect to these concepts. Are these concepts additional annotations in the dataset? [...]

We handle both the cases when we have and don't have the explicit annotations for the concepts in the data. For datasets
like CUB-200 and Awa2, we have the concept annotation in the data in the form of attributes. For skin dataset, HAM 10k
we acquire the concept annotations from derm7pt dataset [1, 2]. For chestXrays, we use the Stanford RadGraph pipeline to
obtain the anatomical and observation concepts from the radiology reports [3]. We mention all our assumptions in the
method section 3 (Notation). Furthermore, we describe all the datasets in details in the Appendix section A.1 (Datasets)
.

References:

1. Lucieri et al. On interpretability of deep learning based skin lesion classi- fiers using concept activation vectors.
   In 2020 international joint conference on neural networks
   (IJCNN), pp. 1–10. IEEE, 2020.
2. Daneshjou et al. Disparities in dermatology ai: Assessments using diverse clinical images. arXiv preprint arXiv:
   2111.08006, 2021
3. Yu et al. Anatomy-guided weakly-supervised abnormality localization in chest x-rays. arXiv preprint arXiv:2206.12704,
   2022

