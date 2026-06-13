import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class NeuroFitSNN(nn.Module):
    def __init__(self, num_inputs=2, num_hidden=32, num_outputs=2, beta=0.85):
        super().__init__()
        
        spike_grad = surrogate.fast_sigmoid()
        
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, spike_data):
        time_steps = spike_data.size(0)
        batch_size = spike_data.size(1)
        
        
        
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=spike_data.device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=spike_data.device)
        
        spike2_recording = []
        mem2_recording = []
        
        for step in range(time_steps):
            current_input = spike_data[step]
            
            
            cur1 = self.fc1(current_input)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spike2_recording.append(spk2)
            mem2_recording.append(mem2)
            
        return torch.stack(spike2_recording), torch.stack(mem2_recording)

def run_inference(model, spike_train):
    
    
    with torch.no_grad():
        spk_rec, _ = model(spike_train)
        
        spike_counts = spk_rec.sum(dim=0).squeeze(0)
        prediction = torch.argmax(spike_counts).item()
        
        total_elements = spike_train.numel()
        active_spikes = torch.count_nonzero(spike_train).item()
        sparsity = 1.0 - (active_spikes / total_elements)
    return prediction, sparsity

def get_pretrained_mock():
    """
    Generates a functional starter weight state for real-time app visualization.
    """
    model = NeuroFitSNN()
    with torch.no_grad():
        
        model.fc2.weight[1] += 0.60  
    return model