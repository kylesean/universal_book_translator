# Chapter 3: Working Memory and Attention

## 3.1 Architecture of Working Memory

Working Memory (WM) refers to a cognitive system capable of temporarily holding and manipulating information in mind for complex tasks such as reasoning, comprehension, and learning. Unlike classical short-term memory (STM), which merely acts as a passive storage buffer, working memory actively coordinates attentional resources through multiple domain-specific subsystems.

Alan Baddeley and Graham Hitch proposed a multicomponent model of working memory that remains the prevailing framework in contemporary cognitive science. The model originally posited three distinct components: the Central Executive, the Phonological Loop, and the Visuospatial Sketchpad. Decades later, Baddeley introduced a fourth component, the Episodic Buffer, to explain how information across different modalities is bound together into coherent multidimensional representations.

<aside class="sidebar-box" data-flow="sidebar_aside">
### Sidebar 3.1: Clinical Neuropsychology of Working Memory Deficits

Patient H.M., whose bilateral medial temporal lobes were surgically resected in 1953 to alleviate intractable epilepsy, exhibited profound anterograde amnesia. Crucially, however, his working memory capacity remained remarkably intact. When provided with a three-digit sequence, H.M. could rehearse and maintain it indefinitely through the phonological loop, provided his attention was not diverted by external distraction.
</aside>

The Central Executive functions as the supervisory attentional controller, directing focal attention, inhibiting prepotent responses, and coordinating dual-task processing. It does not possess intrinsic storage capacity; rather, it regulates the allocation of processing bandwidth across the slave systems.

The Phonological Loop comprises two subcomponents: the phonological store ("inner ear"), which holds acoustic and speech-based memory traces for roughly two seconds, and the articulatory rehearsal process ("inner voice"), which refreshes decaying traces via subvocal articulation. Empirical phenomena supporting this architecture include the phonological similarity effect, the word length effect, and articulatory suppression.

The Visuospatial Sketchpad is responsible for the temporary maintenance and manipulation of visual features and spatial coordinates. Behavioral experiments involving concurrent visual tracking demonstrate that spatial interference impairs visual imagery tasks without affecting phonological retention.

| Subsystem | Primary Modality | Anatomical Correlates | Typical Assessment Paradigm |
| :--- | :--- | :--- | :--- |
| Central Executive | Modality-independent | Dorsolateral Prefrontal Cortex (DLPFC) | N-back Task, Stroop Task |
| Phonological Loop | Acoustic / Verbal | Left Inferior Parietal & Broca's Area | Digit Span, Nonword Repetition |
| Visuospatial Sketchpad | Visual / Spatial | Right Occipito-Parietal Network | Corsi Block-Tapping Task |
| Episodic Buffer | Multimodal Integration | Anterior Cingulate & Hippocampus | Prose Recall, Complex Span |

Mathematical models of signal detection theory (SDT) quantify perceptual sensitivity $d'$ independently of response bias $c$:

$$ d' = Z(\text{Hit Rate}) - Z(\text{False Alarm Rate}) $$

In empirical attention tasks, reaction times and error rates are frequently simulated using cognitive modeling architectures:

```python
def compute_signal_detection_sensitivity(hit_rate: float, fa_rate: float) -> float:
    """Compute parametric sensitivity index d-prime from empirical rates."""
    import scipy.stats as stats

    z_hit = stats.norm.ppf(hit_rate)
    z_fa = stats.norm.ppf(fa_rate)
    return float(z_hit - z_fa)
```

Contemporary neuroimaging studies employing functional magnetic resonance imaging (fMRI) have confirmed that distinct prefrontal cortical networks subserve these dissociable memory mechanisms[^1].

[^1]: Baddeley, A. (2000). The episodic buffer: A new component of working memory? Trends in Cognitive Sciences, 4(11), 417-423.
