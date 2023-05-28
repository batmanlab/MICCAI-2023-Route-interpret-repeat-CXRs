# General Response

We thank all reviewers for their feedback. We address each reviewer separately. In the following, please find our
general comment.

## About Novelty:

- Our method is the first to blur the distinction between **post hoc** explanation and **interpretable** by design-based
  approaches by **carving an interpretable model out of the BlackBox**. The method shows how to maintain
  interpretability without compromising accuracy by gradually distilling from a BlackBox and building a new hybrid
  symbolic-Blackbox until convergence. To the best of our knowledge, there is no such approach in the literature.
- We showed that by taking advantage of a BlackBox (distilling), we achieved better performance than training from
  scratch (classical interpretable method).
- There is always a subset of samples for which the template of the interpretable architecture is not optimal. Our
  approach can identify those via the last selector ($\pi^k$).
- Our method can be viewed as a framework, allowing any symbolic backbone for the interpretable method, making our
  approach applicable to a wide range of data types and applications (image, language, tabular data, video). We showed
  results using First Order Logic (FOL), but many other choices are possible.

# R1

We thank the reviewer for the constructive and thoughtful comments. We hope that the following discussion addresses your
questions.

* Please find the comment about the novelty above.

> [...] Proto-Tree (Nayuta et al., CVPR21) which have addressed the weaknesses of the original ProtoPNet model. Here I would expect the positioning to be made with respect to more recent prototype-based methods. [...]

We agree with the reviewer that the prototype method is one of the papers in that family and indeed the ProtoTree
addresses some of those issues. We will add that citation to the paper. Our method offers a more flexible interpretable
method that improves accuracy (Table below for CUB-200 dataset).

| Method                                   | Top-1 Accuracy |
|------------------------------------------|---------------|
| ProtoPNet, Chen et al. [Neurips, 2018]   | 79.2 %        |
| ProtoTree h=9, Nayuta et al.[CVPR, 2021] | 82.2 %        |
| MoIE (ours, ResNet Backbone)             | 86.2 %        |
| MoIE (ours, VIT Backbone)                | 90.7 %        |

There are also the following key differences:

- Our method allows leveraging a blackbox and distilling it to any symbolic method (including ProtoTree), while a
  Prototype-based approach should be trained from scratch. Training from scratch can be a difficult optimization task,
  depending on the template or architecture of the interpretable method.
- The samples routed to the last residuals can be viewed as a subset of data for which the template of the interpretable
  method is not appropriate. Neither Prototype nor ProtoTree offers such flexibility.
- Using prototype approaches to fix undesirable properties such as shortcuts is not straightforward. We have shown that
  our method can easily be used for such applications.
- Our method is also tested on a more diverse dataset.

> Is there a principle manner to decide on the number of experts to use for the proposed method? [...]

We follow two principles to stop the recursive process. 1) Each expert should have enough data to be trained reliably (
coverage $\zeta^k$). If insufficient samples are assigned to an expert, we stop the process. 2) If the latest residual (
$r^k$) is underperforming, it is not a reliable black box to distill from. We stop the procedure to avoid degrading the
overall accuracy. We add a section about stopping criteria in the appendix or in the algorithm block in **red** color.

> [...] In this regard, it would have strengthen the manuscript if the proposed method was tested under the sanity checks proposed by [Adebayo et al., NeurIPS'18]. [...]

Adebayo et al. is about sanity check for the posthoc explanation method and, more specifically saliency-based approach.
Our method is close to the category of interpretable methods. Unlike the traditional interpretable method trained from
scratch, we use the flexibility of a blackbox DL to design an interpretable method on-the-fly (during training)
progressively.

> [...] Quantitative Comparison At the moment there is no quantitative comparison on the outputs of the proposed methods with respect to those from state of the art methods [...]

This comment is related to the previous one. Our method is **not** a post hoc explanation method but rather an
interpretable approach. Many quantitative experiments in the paper are consistent with the evaluation of the
interpretable methods:

- **Does our method compromise accuracy?** Sec 4.2.1 and Figure 2 show that our method outperforms the interpretable
  baseline (Concept bottleneck model Koh et el. 2020) and does not compromise the performance of the respective
  blackbox.

- **Is there any sample that does not fit the template of the interpretable method?** There are always samples that need
  to fit the interpretable template. Our last selector ($$g^k$$) can identify those. Note that for these samples the
  performance of the blackbox should be low. We compare the performance of the black box for these samples with the last
  residual ($$r^k$$). Please refer to section A 7.7 and Table 6 in the supplementary material for details.

- **Is this a helpful method for application?** This is the primary concern about any XAI approach. We showed that our
  method could fix the shortcut bias. Please refer to section 4.3 for details.

> [...] On the semantic association of the detected concepts In Section 4.2.2, local explanation produced by the MoIE are discussed. Several of these are described based on concepts with a clearly associated semantic meaning , e.g.

It seems that the last sentence is missing. We would be thankful if you update your comment.

# R2

Despite the brevity and lack of specificity of the review, we try to address comments to the best of our knowledge. We
hope to engage with the reviewer to understand his/her/their concern and go beyond a superficial understanding of our
paper.

* It seems the main purpose of our manuscript and those of Bau et al and Mu et al are not well understood. Bau et al and
  Mu et al aim at **posthoc explanation** of the black-box. Both papers stop at the explanation and do not result in a
  new **interpretable network**. Please note that post hoc explanation and interpretable by design are two different
  categories of eXplainable AI (XAI).
* This paper is about streamlining the development of a new interpretable model by taking advantage of the flexibility
  of a black-box DL. In that sense, neither Bau et al nor Mu et al are SOTA for interpretable methods.
* In the current literature of XAI, posthoc explanation and interpretable methods are two distinct design choices that
  need to be decided at the beginning. We aim to blur that line.
* About novelty: please see the general comment about novelty.

# R3

We thank the reviewer for the feedback and constructive comments. We hope the following reply address the points raised
by the reviewer. We would be more than happy to address any other outstanding issue.


> [...] a short explanation of the first order logic is missing [...]

We will add a short description of FOL in the paper. Due to the space limitation, a longer version will be added to the
Supplementary Material. In short, FOL is a logical function that accepts predicates (concept presence/absent) as input
and returns a True/False output that is a logical expression of the predicates. The logical expression, which is a set
of AND, OR, Negative, and parenthesis, can be written in the so-called Disjunctive Normal Form (DNF). Disjunctive Normal
Form (DNF) is a FOL logical formula composed of a disjunction (OR) of conjunctions (AND). It is also referred to as the
“sum of products.”


> [...] a short explanation of ``neuro-symbolic interpretable model'' is missing [...]

We added a short description to the manuscript in red. In short, Neuro-symbolic AI is an area of study that encompasses
deep neural networks with symbolic approaches to computing and AI to complement the strengths and weaknesses of each,
resulting in a robust AI capable of reasoning and cognitive modeling.


> [...] no code [...]

As mentioned in appendix A.4, we will release the code upon the decision. We listed all the hyperparameters in details
Appendix A.6.

> [...] complicated approach, training required [...]

The proposed approach is **not** a posthoc explanation but rather a new interpretable network that is designed
on-the-fly (ie during the recursive process). Therefore, training is needed. Each step of the procedure *carves out* an
interpretable model (ie FOL expert) to explain a subset of data. Please note that a single expert (eg FOL) is suboptimal
for the entire data; this is why the selector ($g^k$) identifies a subset of samples for the expert $k$. This procedure
is repeated until the mixture of experts is sufficiently powerful and on par with the initial BlackBox. We would
appreciate it if the reviewer provided further feedback regarding their comment about the complexity. We are using SGD
and the arch classical vision CNN or ViT. All hyperparameters are specified in the supplementary.

> [...] no comparison with similar concept based explanation methods (like CAV)[...]

CAV is a **posthoc explanation-based** method aiming to quantify the association of a concept to the final prediction of
a blackbox. Our aim is completely different. Our goal is to design a new **interpretable network** that uses the
concepts and the interaction between them (via FOL) as bottlenecks. We update

> . [...] no quantitative analysis of the produced explanations [...]

Our method is **not** a post hoc explanation method but rather an interpretable approach. There are many quantitive
experiments in the paper. The metric we used for evaluation are consistent with the evaluation of the interpretable
methods:

- **Does our method compromise accuracy?** Sec 4.2.1 and Figure 2 show that our method outperforms the interpretable
  baseline (Concept bottleneck model Koh et el. 2020) and does not compromise the performance of the respective
  blackbox.

- **Is there any sample that does not fit the template of the interpretable method?** There are always samples that need
  to fit the interpretable template. Our last selector ($g^k$) can identify those. Please refer to section A 7.7 and
  Table 6 in the supplementary material for details.

- **Is this a helpful method for application?** This is the primary concern about any XAI approach. We showed that our
  method could fix the shortcut bias. Please refer to section 4.3 for details.

> [...] that the selector routes a sample with a probability... Is this routing really probabilistic [...]

Yes, it is probabilistic in the sense that the **loss function** is the expectation of the latent variable that selects
samples. By taking expectation, the random variable is no longer in the loss. $$ \ell( x_i ; \theta ) = \mathbb{E}_
{\phi} \left[ \mathbb{1}(x_i)  \ell( x_i, g^k ) + (1 - \mathbb{1}(x_i)) \ell(x_i, r^k) \right], $$ where $\mathbb{1}(x)$
is the indicator function selecting sample $x_i$ for the expert $k$ ($g^k$). We did not sample from this expectation to
route samples to different experts as it was not the paper's goal.

> [...]  using the original black box classifier in the first iteration, if I understand correctly. Why is the performance in Figure 2 then not equal to the black box [...]

The figure 2 compares the final performance of the carved interpretable network (ie Mixture of experts) with the
BlackBox. The goal of the figure is to show that the new interpretable network does not compromise the accuracy of the
BlackBox. We are distilling from the BlackBox to a new network, so they are not the same.

> [...] why do you not show explanations for the same samples in figure 4? [...]

We changed the figure and used the same samples for the baseline and the experts. We also added another qualitative plot
for Awa2 in figure 4.

> [...] Pseudo code is given but no github implementations, so reproducibility may be cumbersome.[...]

As mentioned in supplementary A.4, we will release the code upon the decision. We listed all the hyperparameters in
details Appendix A.6.

# R4

We thank the reviewer for the constructive feedback. We hope our reply answer the comments. We would be happy to address
any outstanding issues.

> [...] Section 2 on related work seems a bit short, especially the paragraph on concept-based interpretable models could be extended. What methods are used by others and how does the paper at hand differ from those? In line with that, the contributions of the paper in relation to previous work could be highlighted more. [...]"

We have updated the section accordingly.

> [...] Similarly, there are no comparisons done with existing methods (except against the ‘baseline’). If there is a specific reason for that, it was not clear. If not, I would advocate comparing to current state-of-the-art methods for symbolic models for image data. [...]"

We compare our model with the blackbox (upper bound) and the interpretable by design baseline (lower bound). Our
interpretable by design approach is based on the SOTA concept bottleneck (Koh et al., ICML 2020). Regarding other
symbolic models, during the time of the submission there was hardly any symbolic model for image classification.
Recently Greybox-XAI (Bennetot et al. 2022, available online after we submitted)
addresses this gap. However, their approach is based on segmentation, whereas we rely on concept attribute annotations.

> [...] Section 5 (discussion and conclusion) would benefit from a discussion of the strengths and limitations of the presented work. Future research trajectories could be discussed more detailed. [...]

Added in red. Please check.

> [...] The readers would benefit from a brief explanation about neuro-symbolic interpretable models in Section 3.1. [...]

Added in red. Please check.

> [...] The optimization problem (2) looks highly non-linear and non-convex. The authors have not mentioned difficulty of the problem, neither have they discussed the cases of obtaining multiple or only local solutions. They only propose one method to solve the problem in the appendix. However, this part is crucial as the method relies on the solution of this problem.[...]"

Thank you for the suggestion. We agree with the reviewer that the optimization landscape is non-convex, and providing a
guarantee for the global optimum is difficult. However, as is the case for general DL, the local optima provide good
generalization (Asymmetric Valleys, He et al. [Neurips, 2019]; Sharp Minima, Dinh et al. [ICML, 2017]). Please note that
models being interpretable by design, are trained from scratch. So the optimization procedure for this class of models
is more complex. However, distillation from the black box is easier for our model to optimize.

> What is a good value for K to stop the algorithm? Is there a rule of thumb when to stop? [...]

We follow two principles to stop the recursive process. 1) Each expert should have enough data to be trained reliably (
coverage $\zeta^k$). If insufficient samples fall into the expert, we stop the process. 2) If the latest residual (
$r^k$) is under-performing, it is not a reliable black box to distill. We stop the procedure to avoid degrading the
overall accuracy. We add a section about stopping criteria in the appendix or in the algorithm block in **red** color.

> [...] What is the effect of validation threshold (fixed to 0.7 in the paper)? [...].

We use this validation threshold to select only those concepts which the intermediate representation \phi of the
blackbox predicts accurately. If the blackbox achieves high performance, its representation must learn the different
semantic concepts accurately as well. We want to leverage only the highly predictive concepts for explaining the
prediction of the BlackBox.

> [...] How would the method extend to non-image datasets? [...]

Yes, definitely. Given an example of how the symbolic section (FOL) change be modified to anything else like a program
and the same idea can be applied for NLP or categorical dataset.

> [...] What is a concept extractor in Section 4.1?[...]

The baseline (interpretable by design), have two parts - the concept extractor(\phi) and the downstream classifier (g)
, **trained from sequentially from scratch**. The concept extractor(\phi) maps the input images to the high level
intermediate concepts (c). The downstream classifier (g) aims to classify the class labels from the concepts, extracted
by the concept extractor ($\phi$).

> [...] In the beginning of Section 3, all \phi functions should be \Phi. [...]

We have fixed this issue.

> [...] When referring to a figure, the authors sometimes use "figure #" instead of "Figure #" [...]"

We have fixed this issue.

> [...] In equation (3), $c_j$ terms should be boldface[...]

We have fixed this issue.

 


 
