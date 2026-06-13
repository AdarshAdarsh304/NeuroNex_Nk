import numpy as np
import torch

def generate_mock_ecg(patient_type="normal", num_steps=300):
    """
    Simulates standardized synthetic ECG waves for localized processing stability.
    """
    t = np.linspace(0, 6 * np.pi, num_steps)
    
    base_wave = 0.1 * np.sin(t) + 0.4 * np.sin(2 * t)
    
    
    r_peaks = np.zeros(num_steps)
    r_peaks[::50] = 1.2  
    raw_signal = base_wave + r_peaks
    
    if patient_type == "arrhythmia":
        
        noise = np.random.normal(0, 0.22, num_steps)
        raw_signal += noise
        raw_signal[25:135] += 0.75  
        
    
    raw_signal = (raw_signal - np.min(raw_signal)) / (np.max(raw_signal) - np.min(raw_signal))
    return raw_signal

def delta_modulation_encoder(signal, threshold=0.15):
    """
    Converts a continuous voltage amplitude wave into separate 
    positive-increase and negative-drop binary spike channels.
    """
    num_steps = len(signal)
    spike_train = np.zeros((num_steps, 2))
    
    prev_val = signal[0]
    for t in range(1, num_steps):
        diff = signal[t] - prev_val
        if diff >= threshold:
            spike_train[t, 0] = 1.0  
            prev_val = signal[t]
        elif diff <= -threshold:
            spike_train[t, 1] = 1.0  
            prev_val = signal[t]
            
    
    return torch.tensor(spike_train, dtype=torch.float32).unsqueeze(1)

if __name__ == "__main__":
    test_sig = generate_mock_ecg("normal")
    spikes = delta_modulation_encoder(test_sig)
    print("Verification Pass -> Extracted Spike Matrix Shape Tensor:", spikes.shape)