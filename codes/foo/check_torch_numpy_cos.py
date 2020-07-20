import torch
import numpy as np

x = torch.FloatTensor(torch.rand([10, 5]))
print('x', x)
y = torch.FloatTensor(torch.rand([1, 5]))
print('y', y)
 
similarity = torch.cosine_similarity(x, y, dim=1)
print(similarity.shape)
similarity = torch.stack((similarity, similarity), dim=1)
a = similarity.numpy()

def cos_distane(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    # dist = 1. - similiarity
    return similiarity

b = cos_distane(x.numpy(), y.numpy())
print(a-b)