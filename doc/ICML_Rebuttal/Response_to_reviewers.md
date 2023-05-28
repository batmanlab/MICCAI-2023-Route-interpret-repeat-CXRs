# Reviewer 3eyg

> From the description of the method I did not understand how the concepts C enter the picture, and how the concept extractor t is being trained.. you state that the model does not require concept-based annotations, but you use the ELE approach by Barbiero et al. (the acronym is ELE, non ELL, if I'm not mistaken) which requires interpretable concepts as inputs. Is it trained end-to-end with the concept extractors? looking at eq. (3), it seems concepts are provided as inputs.. can you clarify?

We are sorry about the confusion. So, we train $t$ with the image embeddings ($\Phi$) as an input to predict the
Concepts $C$. We get the concept annotation from the dataset itself (for CUB, Awa2) or from some other datasets (for
HAM10000, we use Derm7pt dataset as [PosthocCBM (PCBM)
(Yuksekgonul et al., 2022)](https://openreview.net/pdf?id=nA5AZ8CEyow)).


> While the high-level idea is clear, the meaning of defining the residual as the difference between f^{k-1}(x_j) and g^k(c_j), and training f^{k}(x_j) to fit this difference is unclear to me.

`meaning of residual:` We hypothesize that the experts focuses on different subsets of data unlike the Blackbox that
fits the entire dataset. So at each iteration, after fitting the expert, we define the residual to focus on whatever
remains.

`training f^{k}(x_j):`
We train $f^k$ to make it specialize on the samples not covered by the corresponding expert $g^k$ approximating the
residual $r^k = f^{k-1}- g^k$. We use $f^k$ as a Blackbox for the next iteration.


> In table 2, I believe boldfacing the results of the blackbox and the proposed method is misleading.. boldface usually means best results (possibly statistically significantly better)

Good point, we will change it in the paper.

PS: ICML does not allow us to edit the paper during rebuttal. Upon acceptance, we will include the points in the main
paper.

# Reviewer JWPv

> making the process by which samples are selected at each iteration clearer. I am sure I could find it with more time, but remain unsure: is there a backprop mechanism of some sort; is it greedy? What are the computational costs (again, I may have missed this)?

The selector associates the explainable samples to the corresponding expert. Each selector, interpretable expert and the
corresponding residual is learned using backprop with Eq. 3 and Eq. 4 respectively.

Regarding computation cost, the selectors and the experts consumes low dimensional concepts as input and are shallow
networks. For the residuals, we only update the last layer (fully connected one). So the computation cost for them is
cheap. Also we plotted them in Fig.11 in the Appendix.

> perhaps rework the abstract to include the hypothesis above, and the MoIE name.

Good point. Here is the new abstract. We will update the abstract upon decision:

*ML model design either starts with an interpretable model or a Blackbox and explains it post hoc. Blackbox models are
flexible but difficult to explain, while interpretable models are inherently explainable. Yet, interpretable models
require extensive ML knowledge and tend to be less flexible, potentially underperforming than their Blackbox
equivalents. This paper aims to blur the distinction between a post hoc explanation of a Blackbox and constructing
interpretable models. **We hypothesize that a Blackbox model encodes several interpretable models, each applicable to
different portions of data**. Beginning with a Blackbox, we iteratively \emph{carve out} a mixture of interpretable
experts (**MoIE**) and a \emph{residual network}. The interpretable models identify a subset of samples and explain them
using First Order Logic (FOL), providing basic reasoning on concepts from the Blackbox. We route the remaining samples
through a flexible residual. We repeat the method on the residual network until all the interpretable models explain the
desired proportion of data. Our extensive experiments show that our \emph{route, interpret, and repeat} approach
(1) identifies a richer diverse set of instance-specific concepts with high concept completeness via interpretable
models by specializing in various subsets of data without compromising in performance,
(2) identifies the relatively harder'' samples to explain via residuals,
(3) outperforms the interpretable by-design models by significant margins during test-time interventions,
(4) can be used to fix the shortcut learned by the original Blackbox.*

> tighten exposition more generally (e.g. cut "Potentially, FOL can be used...", "and many more", reduce repetitions of R-I-R methodology) and use the saved space to hone intuitions.

Good point. We update the paper.

> I had not seen 'FOL' used to refer to a set of sentences rather than to the logic itself. If that is common practice, then this is fine. If not, I would prefer to distinguish between the logic itself and sentences formed using it.

Your observation is correct. We use the sentence to clarify the FOL while discussing the results.

> the term "sufficient statistic" (p.6) didn't seem right to me: it refers to the proportion of variance explained?

Your observation is correct, it refers to the proportion of variance explained. However, this term is used in
completeness score paper [Chih-Kuan Yeh et al.,Neurips, 2020](https://arxiv.org/pdf/1910.07969.pdf) and we adopt them to
show utility of the identified concepts.


> minor typos: e.g. on p.6 "MoIE-identified" versus "MoIE identifies".

We fix the typos.

> in Table 2's caption, I was confused by mention of both AUROC and accuracy when the table reports only a single figure.

Thanks for pointing this out. We update the caption accordingly. Tab. 2 reports AUROC for medical imaging datasets (
HAM10000, ISIC, and Effusion of MIMIC-CXR) and accuracy for vision datasets (Awa2 and CUB-200) only. As medical imaging
datasets contain class imbalance, we report AUROC instead of accuracy.


> residuals are not designed to fix the mistakes made by the experts" means that there is nothing like a backprop mechanism to update an expert model once it has been selected?

Here we discuss the difference between the design of the residuals
of [Posthoc CBM (Yuksekgonul et al., ICLR 2022)](https://openreview.net/pdf?id=nA5AZ8CEyow) and our model. In PCBM, they
fit a interpretable model ($g$) to explain the Blackbox ($f^0$). Next, they fit the residual on the entire dataset not
to compromise the performance of the Blackbox. In the experiments, they found that
their `residual component in PCBM-h intervenes only when the prediction for PCBM is wrong, and fixes mistakes (Page 5, section: Comparing to CBM. PCBM paper)`
. However, in our design, the residual is used to carve out the Blackbox for the next iteration, specializing on those
samples not covered by the respective interpretable model. Like PCBM, we also fit the entire data to the residual.
However, our selectors impose a soft attention scores (Eq.4) to ensure the carved Blackbox of the next iteration from
the residual, focus on the subset of samples rejected by the corresponding selector. So fundamentally, the design of the
residuals are different.

#### Questions

> could you give an example of 'interactions between concepts' (p.1)?

Sure.

BayBreastedWarbler $\leftrightarrow$ (back\_pattern\_striped $\land$ upperparts\_color\_black $\land$ wing\_color\_white
$\land$ $\neg$ nape\_color\_grey $\land$ $\neg$ throat\_color\_black )
$\boldsymbol{\lor}$
(belly\_pattern\_solid $\land$ forehead\_color\_black $\land$ shape\_perchinglike $\land$ upperparts\_color\_black
$\land$ wing\_color\_white $\land$ $\neg$nape\_color\_grey $\land$ $\neg$primary\_color\_black $\land$
$\neg$primary\_color\_white $\land$ $\neg$throat\_color\_black $\land$ $\neg$under\_tail\_color\_black)

Here BayBreastedWarbler is the predicted bird species and back_pattern_striped, upperparts_color_black, .. etc are the
concepts for CUB-200 dataset.

> the MoIE has the flavour of a decision tree, but with a model at each bifurcation point, and probabilistic bifurcation points. If this is roughly right, it seems a bit odd that the are not in . Is there an intuition for this?

One can view the $\pi_k$ as posterior probability in so-called stick-breaking process [1] where in each iteration, the
probabilistic bifurcation decides to break a stick of length 1 . If the stick is broken $k$ times, one can create $k+1$
dimensional membership values with the same procedure as done in non-parametric bayesian methods. To connect to your
analogy of decision tree: the decision is to break the stick or not. You can find more detail in [1]
[1] https://en.wikipedia.org/wiki/Dirichlet_process#The_stick-breaking_process

> when multiple experts use the same concept, e.g. back_pattern_striped, would there be any advantage to promoting that concept to a higher level in the 'tree'?

We have not explored this. Currently, all experts have access to the same level concepts but each can make their own FOL
function. Perhaps, in the future work we can put some budget constrain on the used concepts.

# Reviewer ruR9

## Weakness

> The overall hypothesis is a little bit too strong (personal thought). The design of MoIE process seems intuitive without too much insights on why.

Our hypothesis is a set of interpretable models and residual covering the explainable and unexplainable components of a
Blackbox respectively. As the interpretable models are simple models, they are not sufficient for all the data. So we
fit multiple interpretable models to focus on various subsets of data.

> The number of considered baselines is small, even though many representative ones has been mentioned in the Related Work.

For concept based models, CEM and CBM (sequential) performs the best. While the initial CBM (Koh et al., 2020) has both
sequential and end-to-end variants, the end-to-end CEM performs better than end-to-end CBM. So we include end-to-end CEM
and sequential CBM as a interpretable baseline. Along with this, there is antehoc (Sarkar et al.) network. However, the
core idea of that network is to use the decoder with CBM for unsupervised concept discovery. For posthoc concept based
model, only PCBM is SOTA.

## Question

> I'm not try to challenge the basic idea of this paper, but still I want ask, why black-box model can be assumed with encoding several interpretable models. Even though possible, I'm afraid the value k could be super large for a simple neural nets.

The interpretable-by-design approach leverage on the presence or absence of concepts to predict the labels. This makes
the optimization difficult and the model not expressive. The higher layers of the Blackbox encode the concepts and a
Blackbox is easy to optimize and flexible. So we start with the Blackbox to carve out the interpretable models. We stop
the method if all the interpretable models cover substantial amount of data or the residua of the previous iteration
under performs than a desired threshold.

> It would be super helpful if the authors can provide deeper insights on such additivity from model structure perspective.

> Is it possible to compare with more commonly-used interpretation methods?

As discussed in the experiments, we compare with CBM (Koh et al., 2020) and CEM (Zarlenga et al., 2022) as
interpretable-by-design concept based baselines. We also compare with PCBM (Yuksekgonul et al., 2022) as post-hoc
concept based baseline. Also we modified the standard logistic classifier of CBM and PCBM to use the classifier with
same setting as our expert to compare the FOLs with our method.

> MoIE has multiple steps and components. So, can we have some ablation study results on that? I think it would be helpful for readers to understand the essential improvement.

Though our model is iterative, still the optimization does have too many parameters. Though we perform two ablations for
CUB-200 and effusion Resnet and VIT:

1. We train all the experts and final residual together.
2. We train the concept detector ($t$), all the experts and final residual together Here are the results:

| Method                                                            | Top-1 Accuracy (CUB-200, Resnet101) | Top-1 Accuracy (CUB-200, VIT) | AUROC (Effusion, MIMIC-CXR) |
|-------------------------------------------------------------------|-------------------------------------|-------------------------------|-----------------------------|
| Our model (MoIE+R)(reported in the paper)                         | 84 %                                | 90.1 %                        | 0.86                        |
| All experts and final residual (end to end)                       | 80.2 %                              | 90.1 %                        | 0.79                        |
| All concept detector ($t$)experts and final residual (end to end) | 75.8 %                              | 77.5 %                        | 0.72                        |

## Limitations:

> Lack theoretical support and guarantee. Need to be cautious for high-stake applications.

Though our method is empirical, we evaluate it to classify *Effusion* from a real-life large radiology dataset MIMIC-CXR
using weakly annotated concepts as concept annotation is highly expensive in medical imaging. Tab.2 in our paper shows
that for MIMIC-CXR, our method does not compromise the performance of a Blackbox. Also due to space constraint, we were
not able to include the explanations in the paper, we are doing it here.

`Expert1:
Effusion <=> left_pleural & ~right_pleural & pleural_unspec & retrocardiac & ~right_costophrenic & costophrenic_unspec & cardiophrenic_sulcus & ~wire & fluid & ~distention`

`Expert2:
Effusion <=> left_pleural & right_pleural & pleural_unspec & ~left_diaphragm & ~costophrenic_unspec & ~cardiophrenic_sulcus & fluid & pressure & ~aspiration`

`Expert3:
Effusion <=> left_pleural & right_pleural & right_lower_lung & ~cardiophrenic_sulcus & ~engorgement & drainage & ~pressure & ~redistribution & ~aspiration`

Also, as a future work, we are planning to tackle the imbalance in the dataset of such large dataset like MIMIC-CXR.

> The authors do not evaluate from the efficiency perspective. Honestly, I think MoIE may not be that efficient compared with many post-hoc methods we commonly use.

Efficiency can be estimated with in terms of model efficiency and data efficiency. For model efficiency, our method does
not fall into the category of post hoc explanation methods like saliency maps. We agree that saliency maps are
computationally inexpensive as they don't require retraining, but they can not be intervened. Specifically, they will
not to rectify the mistakes of a Blackbox, thus algorithmic recourse is not possible. Our method is based on presence
and absence of concepts, so as shown in the test time intervention experiment, we can intervene and fix a wrong
prediction.

For data efficiency, we are planning to utilise our method for efficient transfer learning to a new domain as finetuning
a Blackbox can be expensive. We plan to train our model on MIMIC_CXR and transfer to Stanford-CXR with using 5% and 10%
of training samples of Stanford-CXR to classify `Effusion`. Here are very preliminary results:

| Method                     | AUROC with 5% training data | Flops (T) with 5% training data | AUROC with 10% training data | Flops (T) with 10% training data |
|----------------------------|-----------------------------|---------------------------------|------------------------------|----------------------------------|
| MoIE                       | 0.89                        | 0.03                            | 0.91                         | 0.04                             |
| MoIE + R                   | 0.88                        | 1.07                            | 0.90                         | 1.47                             |
| BB (finetuned on Stanford) | 0.83                        | 1665.4                          | 0.90                         | 2258.5                           |

> The interpretability of each expert also varies a lot regarding to its complexity and correctness, which can be problematic for many real-world applications

We respectfully disagree the reviewer. The issue with all the concept based interpretable models is that they provide a
generic explanation for all the samples, failing to identify the diverse sample specific concepts. For example to
classify a pneumothorax, either left lower lobe or right lower lobe or both can be important for different samples. The
SOTA interpretable models mostly identify the these 3 to be the most important factors for all the samples. As our model
focuses on various subsets of data, they can indentify samples for which either left lower lobe or right lower lobe or
both important. Also, the later iterations deal with the `harder` samples. So based on the sample difficulty the
complexity of explanations get increased.

| Method                      | CUB-200 (ResNet101) | Awa2 (ResNet101) | Effusion         |
|-----------------------------|------------------|------------------|------------------|
| Antehoc w sup [1]           | 0.71 $\pm$ 0.25  | 0.85 $\pm$ 0.17  | 0.75 $\pm$ 0.10  |
| Antehoc w/o sup [1]         | 0.64 $\pm$ 0.25  | 0.81 $\pm$ 0.17  | 0.70 $\pm$ 0.10  |
| Hard w/o AR [2]             | 0.78 $\pm$ 0.3   | 0.83 $\pm$ 0.3   | 0.71 $\pm$ 0.14  |
| Hard w AR  [2]              | 0.81 $\pm$ 0.3   | 0.86 $\pm$ 0.2   | 0.73 $\pm$ 0.14  |
| MoIE (Ours) (Cov)           | 0.86 $\pm$ 0.01 (0.9) | 0.87 $\pm$ 0.02 (0.91) | 0.87 $\pm$ 0.00 (0.98) |
| MoIE + Residual (Ours) | 0.84 $\pm$ 0.01  | 0.86 $\pm$ 0.02  | 0.86 $\pm$ 0.00  |

[1] Sarkar et al., CVPR, 2021 A Framework for Learning Ante-hoc Explainable Models via Concepts

[2] Havasi et. al., Neurips 2022 Addressing Leakage in Concept Bottleneck Models

