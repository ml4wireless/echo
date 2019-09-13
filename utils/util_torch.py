import torch
from torch.autograd import Function

class StraightThroughArgMaxLayer(Function):
    def forward(self, input):
        values, indices = torch.max(input, 1)
        eye = torch.eye(input.size(-1)) 
        one_hot = eye[indices]
        self.one_hot = one_hot
        return one_hot
    def backward(self, grad_output):
        input = self.one_hot
        grad_output[input==0]=0
        return grad_output

class SoftKArgMaxLayer(Function):
    def __init__(self):
        self.k = 10
    #takes in logits
    def forward(self, input):
        self.logits = input
        values, indices = torch.max(input, 1)
        eye = torch.eye(input.size(-1)) 
        one_hot = eye[indices]
        self.one_hot = one_hot
        return one_hot
    def backward(self, grad_output):
        self.k = 1.02*self.k
        logits = self.logits                             
        with torch.enable_grad():                               
            softmax_k = nn.functional.softmax(logits*self.k, dim=1)                                      
        return torch.autograd.grad(softmax_k, logits, grad_output)