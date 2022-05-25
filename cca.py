import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

def svd_reduction(M):
    M-=np.mean(M, axis=1, keepdims=True)
    U,s,V=np.linalg.svd(M, full_matrices=False)
    sv_sum = np.sum(s)
    i = 0
    quality_ratio = 0
    while i < len(s) and quality_ratio < 0.99:
        i += 1
        quality_ratio = np.sum(s[:i])/sv_sum

    M_reduced = np.dot(U[:, :i], s[:i]*np.eye(i))
    print(M_reduced.shape)

    return M_reduced

def cca(M1,M2): #shape is datapoints x features
    n_comps = min(M1.shape[1], M2.shape[1])
    cca = CCA(n_components=n_comps)
    cca.fit(M1, M2)
    M1_scores, M2_scores = cca.transform(M1, M2)
    print(M1_scores.shape, M2_scores.shape)
    print(np.corrcoef(M1_scores[:, 0], M2_scores[:, 0]))
    corrs = np.array([np.corrcoef(M1_scores[:, i], M2_scores[:, i])[0, 1] for i in range(n_comps)])
    final_score = sum(corrs)/n_comps
    return final_score, corrs

def svcca(M1, M2):
    M1=svd_reduction(M1)
    M2=svd_reduction(M2)
    return cca(M1, M2)

a = np.random.randn(2000,50)
b = np.random.randn(2000,100)

print(cca(a, b))
#print(cca(svd_reduction(a), svd_reduction(b)))
