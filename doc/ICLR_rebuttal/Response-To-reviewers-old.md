# R1
We thank the reviewer for the constructive and thoughtful comments. We hope that the following discussion addresses your
questions.
* About Novelty:
  Our method is the first to blur the distinction between **post hoc** explanation and **interpretable** by design-based
  approaches by **carving an interpretable model out of the blackbox**. To the best of our knowledge, there is no such
  approach in the literature.
    - Our model can take advantage of the flexibility of the blackbox DL while classical interpretable methods are
      limited to their architecture.
    - There is always a subset of samples for which the template of the interpretable architecture is not optimal. Our
      approach can identify those via the last selector ($$\pi^k$$). Our method can be viewed as a framework, allowing
      any symbolic backbone for the interpretable method, making our approach applicable to a wide range of data types
      and applications (image, language, tabular data, video).
* "[...] Proto-Tree (Nayuta et al., CVPR21) which have addressed the weaknesses of the original ProtoPNet model. Here I
  would expect the positioning to be made with respect to more recent prototype-based methods. [...]" We agree with the
  reviewer that the prototype method is one of the papers in that family and indeed the ProtoTree addresses some of
  those issues. We will add that citation to the paper. Our method offers a more flexible interpretable method that
  improves accuracy (Table below for CUB-200 dataset).

| Method                                   | Top-1 Accuracy |
|------------------------------------------|---------------|
| ProtoPNet, Chen et al. [Neurips, 2018]   | 79.2 %        |
| ProtoTree h=9, Nayuta et al.[CVPR, 2021] | 82.2 %        |
| MoIE (ours, ResNet Backbone)             | 86.2 %        |
| MoIE (ours, VIT Backbone)                | 90.7 %        |

There are also the following key differences:

- Our method allows leveraging a blackbox and distilling it to any symbolic method (including ProtoTree), while a
  Prototype-based approach should be trained from scratch. Training from scratch can be a difficult optimization task,
  depending on the template or architecture of the interpretable method. - The samples routed to the last residuals can
  be viewed as a subset of data for which the template of the interpretable method is not appropriate. Neither Prototype
  nor ProtoTree offers such flexibility. - Using prototype approaches to fix undesirable properties such as shortcuts is
  not straightforward. We have shown that our method can easily be used for such applications. - Our method is also
  tested on a more diverse dataset.
* "[...] Is there a principle manner to decide on the number of experts to use for the proposed method? [...]"
  We follow two principles to stop the recursive process.
    1) Each expert should have enough data to be trained reliably (coverage $$\zeta^k$$). If insufficient samples fall
       into the expert, we stop the process.
    2) If the latest residual ($$r^k$$) is under-performing, it is not a reliable black box to distill. We stop the
       procedure to avoid degrading the overall accuracy. We add a section about stopping criteria in the appendix or in
       the algorithm block in **red** color.
* "[...] In this regard, it would have strengthen the manuscript if the proposed method was tested under the sanity
  checks proposed by [Adebayo et al., NeurIPS'18]. [...]" Adebayo et al. is about sanity check for the posthoc
  explanation method and, more specifically saliency-based approach. Our method is close to the category of
  interpretable methods. Unlike the traditional interpretable method trained from scratch, we use the flexibility of a
  blackbox DL to design an interpretable method on-the-fly (during training) progressively.
* "[...] Quantitative Comparison At the moment there is no quantitative comparison on the outputs of the proposed
  methods with respect to those from state of the art methods [...]"
  This comment is related to the previous one. Our method is **not** a post hoc explanation method but rather an
  interpretable approach. Many quantitative experiments in the paper are consistent with the evaluation of the
  interpretable methods:
    - **Does our method compromise accuracy?** Sec 4.2.1 and Figure 2 show that our method outperforms the interpretable
      baseline (Concept bottleneck model Koh et el. 2020)
      and does not compromise the performance of the respective blackbox.
    - **Is there any sample that does not fit the template of the interpretable method?** There are always samples that
      need to fit the interpretable template. Our last selector ($$g^k$$) can identify those. Please refer to section A
      7.7 and Table 6 in the supplementary material for details.
    - **Is this a helpful method for application?** This is the primary concern about any XAI approach. We showed that
      our method could fix the shortcut bias. Please refer to section 4.3 for details.
* "[...] On the semantic association of the detected concepts In Section 4.2.2, local explanation produced by the MoIE
  are discussed. Several of these are described based on concepts with a clearly associated semantic meaning , e.g. "
  IrregularStreaks", "IrregularDG" and Blue Whithish Veil". Hoe[...]" On the semantic association of the detected
  concepts. We need clarification on this comment. The last sentence is missing.

# R2
Despite the brevity and lack of specificity of the review, we try to address comments to the best of our knowledge:
- It seems the main purpose of our manuscript and those of Bau et al and Mu et al are not well understood. Bau et al and
  Mu et al aim at **posthoc explanation** of the black-box. Both papers stop at the explanation and do not result in a
  new **interpretable network**. Please note that post hoc explanation and interpretable by design are two different
  categories of eXplainable AI (XAI).
- This paper is about streamlining the development of a new interpretable model by taking advantage of the flexibility
  of a black-box DL. In that sense, neither Bau et al nor Mu et al are SOTA for interpretable methods.
- In the current literature of XAI, posthoc explanation and interpretable methods are two distinct design choices that
  need to be decided at the beginning. We aim to blur that line.
- About novelty: please see the comment above (in the discussion with Reviewer1) where we explain the novelty of the
  paper. We hope to engage with the reviewer to understand his/her/their concern and go beyond a superficial
  understanding of our paper.

# R3
* "[...] a short explanation of the first order logic is missing [...]" Added to the manuscript in red.
* "[...] a short explanation of ``neuro-symbolic interpretable model'' is missing [...]" Added to the manuscript in red.
* "[...] no code [...]" As mentioned in supplementary A.4, we will release the code upon the decision.
* "[...] complicated approach, training required [...]" The proposed approach is **not** a posthoc explanation but
  rather a new interpretable network that is designed on-the-fly (ie during the recursive process). Therefore, a
  training is needed. Each step of the procedure *carves out* an interpretable model (ie FOL expert) to explain a subset
  of data. Please note that a single expert (eg FOL) is suboptimal for the entire data; this is why the selector (
  $$g^k$$) identifies a subset of samples for the expert $$k$$. This procedure is repeated until the mixture of experts
  are sufficiently powerful and in par with the initial blackbox. We appreciate if the reviewer provided further
  feedback regarding the complexity of our approach. We are using SGD and the arch classical vision CNN or ViT. All
  hyperparameters are specified in the supplementary.
* "[...] no comparison with similar concept based explanation methods (like CAV)[...]"
  CAV is a **posthoc explanation-based** method aiming to quantify the association of a concept to the final prediction
  of a blackbox. Our aim is completely different. Our goal is to desing an new **interpretable network** that uses the
  concepts and the interaction between them (via FOL) as bottleneck.
* "[...] no quantitative analysis of the produced explanations [...]"
  Our method is **not** a posthoc explanation method but rather interpretable approach. There are many quantitive
  experiments in the paper that is consistent with the evaluation of the interpretable methods:
    - **Does our method compromise accuracy?** Sec 4.2.1 and Figure 2 show that our method outperforms the interpretable
      baseline (Concept bottleneck model Koh et el. 2020)
      and does not compromise the performance of the respective blackbox.
    - **Is there any sample that does not fit the template of the interpretable method?** There are always samples that
      need to fit the interpretable template. Our last selector ($$g^k$$) can identify those. Please refer to section A
      7.7 and Table 6 in the supplementary material for details.
    - **Is this a helpful method for application?** This is the primary concern about any XAI approach. We showed that
      our method could fix the shortcut bias. Please refer to section 4.3 for details.
* "[...] that the selector routes a sample with a probability... Is this routing really probabilistic [...]"
  Yes, it is probabilistic in the sense that the **loss function** is the expectation of the latent variable that
  selects samples. By taking expectation, the random variable variable is no longer in the loss. $$ \ell( x_i ; \theta )
  = \mathbb{E}_{\phi} \mathbb{1}(x_i)  \ell( x_i, g^k ) + (1 - \mathbb{1}(x_i)) \ell(x_i, r^k),  
  $$ where $$\mathbb{1}(x)$$ is the indicator function selecting sample $$x_i$$ for the expert k ($$g^k$$). We did not
  sample from this expectation to route samples to different experts as it was not our goal
* "[...]  using the original black box classifier in the first iteration, if I understand correctly. Why is the
  performance in Figure 2 then not equal to the black box [...]"
  The figure 2 compares the final performance of the carved interpretable network (ie Mixture of experts) with the
  blackbox. The goal of the figure is to show that the new interpretable network does not compromise accuracy of the
  blackbox. We are distilling from the blackbox to a new network, so they are not the same.
* "[...] why do you not show explanations for the same samples in figure 4? [...]"
* We change the figure and use same samples for the baseline and the experts. We also added another qualitative plot for
  Awa2 in figure 4.
* "[...] Pseudo code is given but no github implementations, so reproducibility may be cumbersome.[...]" As mentioned in
  supplementary A.4, we will release the code upon the decision.

# R4

* "[...] Section 2 on related work seems a bit short, especially the paragraph on concept- based interpretable models
  could be extended. What methods are used by others and how does the paper at hand differ from those? In line with
  that, the contributions of the paper in relation to previous work could be highlighted more. [...]" Added in red.
  Please check.
* "[...] Similarly, there are no comparisons done with existing methods (except against the ‘baseline’). If there is a
  specific reason for that, it was not clear. If not, I would advocate comparing to current state-of-the-art methods for
  symbolic models for image data. [...]" We compare our model with the blackbox (upper bound) and the interpretable by
  design baseline (lower bound). Our interpretable by design approach is based on the SOTA concept bottleneck (Koh et
  al., ICML 2020). Regarding other symbolic models, during the time of the submission there was hardly any symbolic
  model for image classification. Recently Greybox-XAI (Bennetot et al. 2022, available online after we submitted)
  addresses this gap. However, their approach is based on segmentation, whereas we rely on concept attribute
  annotations.
* "[...] Section 5 (discussion and conclusion) would benefit from a discussion of the strengths and limitations of the
  presented work. Future research trajectories could be discussed more detailed. [...]" Added in red. Please check.
* "[...] The readers would benefit from a brief explanation about neuro-symbolic interpretable models in Section
  3.1. [...]" Added in red. Please check.
* "[...] The optimization problem (2) looks highly non-linear and non-convex. The authors have not mentioned difficulty
  of the problem, neither have they discussed the cases of obtaining multiple or only local solutions. They only propose
  one method to solve the problem in the appendix. However, this part is crucial as the method relies on the solution of
  this problem.[...]" Thank you for the suggestion. We agree with the reviewer that the optimization landscape is
  non-convex and providing guarantee for the global optimal is difficult. However, as it is the case for general DL, the
  local optima are providing good generalization (Asymmetric Valleys, He et al. [Neurips, 2019]; Sharp Minima, Dinh et
  al. [ICML, 2017]). Just to note that models being interpretable by design, are trained from scratch. So the
  optimization procedure for these class of models are more complex. However, distillation from the black box is easier
  for our model to optimize.
* "[...] What is a good value for K to stop the algorithm? Is there a rule-of-thumb when to stop? [...]"
  We follow two principles to stop the recursive process.
    1) Each expert should have enough data to be trained reliably (coverage $$\zeta^k$$). If insufficient samples fall
       into the expert, we stop the process.
    2) If the latest residual ($$r^k$$) is under-performing, it is not a reliable black box to distill. We stop the
       procedure to avoid degrading the overall accuracy. We add a section about stopping criteria in the appendix or in
       the algorithm block in **red** color.
* "[...] What is the effect of validation threshold (fixed to 0.7 in the paper)? [...]". We use this validation
  threshold to select only those concepts which the intermediate representation \phi of the blackbox predicts
  accurately. If the blackbox achieves high performance, it’s representation must learn the different semantic concepts
  accurately as well. We want to leverage only the highly predictive concepts for explaining the prediction of the
  blackbox.
* "[...] How would the method extend to non-image datasets? [...]"
  Yes, definitely. Given an example of how the symbolic section (FOL) change be modified to anything else like program
  and the same idea can be applied for NLP.
* "[...] What is a concept extractor in Section 4.1?[...]"
  The baseline (interpretable by design), have two parts - the concept extractor(\phi) and the downstream classifier (g)
  , **trained from sequentially from scratch**. The concept extractor(\phi) maps the input images to the high level
  intermediate concepts (c). The downstream classifier (g) aims to classify the class labels from the concepts,
  extracted by the concept extractor ($$\phi$$).
* "[...] In the beginning of Section 3, all \phi functions should be \Phi. [...]" We have fixed this issue.
* "[...] When referring to a figure, the authors sometimes use "figure #" instead of "Figure #" [...]" We have fixed
  this issue.
* "[...] In equation (3), $$c_j$$ terms should be boldface[...]" We have fixed this issue.