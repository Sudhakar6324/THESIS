import torch
from torch import nn
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)


    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

## Has no bottleneck layer. Only two layers for residula block
class ResidualSineLayer(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)
        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1
        self.init_weights()


    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0,
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0,
                                           np.sqrt(6 / self.features) / self.omega_0)

    ## Version 2
    def forward(self, input):
        sine_1 = self.omega_0 * self.linear_1(self.weight_1*input)
        sine_1 = torch.sin(sine_1)
        sine_2 = self.omega_0 * self.linear_2(sine_1)
        sine_2 = torch.sin(sine_2)
        return self.weight_2*(input+sine_2)

## Standard model without any bottleneck layers
class MyResidualSirenNet(nn.Module ):
    def __init__(self, num_layers, 
                neurons_per_layer,
                num_input_dim,
                num_output_dim):
        super(MyResidualSirenNet, self).__init__()

        self.Omega_0=30
        self.n_layers = num_layers + 1
        self.input_dim = num_input_dim
        self.output_dim = num_output_dim
        self.neurons_per_layer = neurons_per_layer
        self.layers = [self.input_dim] + [self.neurons_per_layer] * num_layers + [self.output_dim]

        self.net_layers = nn.ModuleList()
        for idx in np.arange(self.n_layers):
            layer_in = self.layers[idx]
            layer_out = self.layers[idx+1]

            ## if not the final layer
            if idx != self.n_layers-1:
                ## if first layer
                if idx==0:
                    self.net_layers.append(SineLayer(layer_in,layer_out,bias=True,is_first=True))
                ## if an intermdeiate layer
                else:
                    self.net_layers.append(ResidualSineLayer(layer_in,bias=True,ave_first=idx>1,ave_second=idx==(self.n_layers-2)))
            ## if final layer
            else:
                final_linear = nn.Linear(layer_in,layer_out)
                ## initialize weights for the final layer
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / (layer_in)) / self.Omega_0, np.sqrt(6 / (layer_in)) / self.Omega_0)
                self.net_layers.append(final_linear)

    def forward(self,x):
        for _, net_layer in enumerate(self.net_layers):
            x = net_layer(x)
        return x        

