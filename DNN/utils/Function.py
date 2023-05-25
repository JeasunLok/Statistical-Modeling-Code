import torch
import torch.nn as nn

# The net is first trained using a cross entropy loss function
class score(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, theta, c, output):
        loss = torch.mean(output * torch.log(1/(1 + torch.exp(-torch.sum((theta[0] * c + theta[1]), axis = 1)))) + (1 - output) * torch.log(1 - 1/(1 + torch.exp(-torch.sum((theta[0] * c + theta[1]), axis = 1)))))
        return -loss
    
#The G function(decided by the problems)
def G(theta, c):
    "the G function"
    return 1/(1 + torch.exp(-torch.sum((theta * c), axis = 1)))

def Lambda(theta, c):
    "the second derivative of loss"
    g = G(theta, c)
    out = [g[i] * (1-g[i])* (torch.mm(c[i].reshape(1,-1).T, c[i].reshape(1,-1)) + torch.cov(c.T)) for i in range(len(g))]
    return out

def H(theta, c):
    "the H function"
    c = torch.as_tensor(c, dtype=torch.float32)
    c_mean = torch.mean(c, axis = 0)
    c_mean = torch.ones(theta.shape[0], c_mean.shape[0])* c_mean
    g = G(theta, c_mean)
    return torch.mul(theta.T, g * (1 - g)).T

def partial_H(theta, c):
    "the first derivation"
    c = torch.as_tensor(c, dtype=torch.float32)
    c_mean = torch.mean(c, axis = 0)
    c_mean = torch.ones(theta.shape[0], c_mean.shape[0])* c_mean
    g = G(theta, c_mean)
    out = torch.mul((theta * c).T, g) - 3 * torch.mul((theta * c).T, g**2) + g - (g**2) + 2 * torch.mul((theta * c).T, g**3)
    out = out.T
    return [torch.eye(out.shape[1]) * out[i] for i in range(out.shape[0])]

def partial_loss(theta, c, y):
    "the second derivative of loss"
    g = G(theta, c)
    return torch.mul(c.T, y-g).T

def psi(theta, c, y):
    h = H(theta, c)
    ph= partial_H(theta, c)

    pl = partial_loss(theta, c, y)
    l = Lambda(theta, c)
    return [h[i] - torch.mv(torch.mm(ph[i], l[i].pinverse()), pl[i]) for i in range(h.shape[0])]

def evaluate(model, input_test, output_test, c_test):
    loss = score()
    _ = []
    __ = []
    model.eval()
    for i in range(input_test.shape[0]):
        theta = model(input_test[i])
        _.append(loss(theta, c_test[i], output_test[i]))
    return torch.mean(torch.tensor(_))
