                   NeuroFit: Ultra-Low-Power Edge ECG Neuromorphic Diagnostics
NeuroFit is an end-to-end edge-computing pipeline that converts continuous analog ECG telemetry into event-driven digital spike trains, running real-time cardiac diagnostic inference through a Spiking Neural Network (SNN).

By transitioning away from computationally heavy, frame-based Deep Learning paradigms, NeuroFit leverages event-driven biological sparsity to minimize redundant synaptic operations, paving the way for ultra-low-power, always-on cardiac monitoring on neuromorphic hardware constraints.


                                    Key Features
Asynchronous Event Extraction: Implements a localized Delta-Modulation Encoder that translates continuous raw analog $mV$ signals into distinct positive ($+V$) and negative ($-V$) discrete spiking events based on dynamic amplitude variance thresholds.

Biologically Plausible Inference: Utilizes a 2-layer Spiking Neural Network (SNN) implemented with Leaky Integrate-and-Fire (LIF) neurons and fast-sigmoid surrogate gradient backward propagation.

Hardware-Aware Sparsity Tracking: Real-time telemetry monitoring displaying precise mathematical data sparsity and calculated theoretical power reduction percentages.

Interactive Diagnostic Telemetry: A Streamlit visual panel presenting raw continuous signals side-by-side with event-driven spike matrices (heatmaps) across multiple clinical patient profiles.
