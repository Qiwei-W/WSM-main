# WSM-main

🚧 **The code is currently being organized and will be released soon. Stay tuned!**

Transformers have demonstrated strong global modeling capabilities in vision tasks, but their lack of inherent inductive
biases often limits their ability to capture fine-grained and stable local structures. To address this limitation, we propose a
lightweight and flexible Wavelet Scattering Module (WSM)
that can be seamlessly integrated into Transformer-based vision architectures to enhance local feature modeling. Specifically, we first design a novel method to extract compact and
interpretable frequency-domain features, allowing the input
image to be partitioned into adjustable local regions, and extracting mean, standard deviation, and significant energy for
each region. Furthermore, we design an efficient multi-modal
feature fusion strategy that combines the extracted scattering
features with visual features without introducing additional
trainable parameters.
