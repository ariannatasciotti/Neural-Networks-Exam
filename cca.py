import numpy as np
from sklearn.cross_decomposition import CCA

def svd(M):
    M-=np.mean(M, axis=1, keepdims=True)
    U,s,V=np.linalg.svd(M, full_matrices=False)
    sv_sum = np.sum(s)
    i = 0
    quality_ratio = 0
    while i < len(s) and quality_ratio < 0.99:
        i += 1
        quality_ratio = np.sum(s[:i])/sv_sum

    M_reduced = np.dot(s[:i]*np.eye(i), V[:i])

    return M_reduced.T

def cca(M1,M2):
    n_comps = min(M1.shape[1], M2.shape[1])
    cca = CCA(n_components=n_comps, tol=1e-5)
    cca.fit(M1, M2)
    M1_scores, M2_scores = cca.transform(M1, M2)
    corrs = [np.corrcoef(M1_scores[:, i], M2_scores[:, i])[0, 1] for i in range(n_comps)]  

    final_score = sum(corrs)/n_comps
    return final_score


a = np.random.rand(500,100)
b = np.random.rand(500,100)

ar = svd(a)
br = svd(b)
print(ar.shape,br.shape,cca(ar,br))


# non sono sicura di aver capito cosa torna "fit" e "transform" di sklearn.cca, la linea 23 dovrebbe tornare i fantomatici
# rho di cui si parla nel paper e per il final score ho provato a fare sta media 