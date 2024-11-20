import pickle
import numpy as np
fp=open("result.pkl","rb")
obj=pickle.load(fp)
vec=[e[0] for e in obj]
#print(len(obj))
out=np.concatenate(vec,1)
print(out.shape)

#print(obj[0][0].shape)

X=out[0]
idx=np.arange(X.shape[0])
np.random.shuffle(idx)

import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

idx=idx[:100000]

umap = umap.UMAP(n_components=2, random_state=0)
X_umap = umap.fit_transform(X[idx])
plt.figure(figsize=(32, 32))
plt.scatter(X_umap[:, 0], X_umap[:, 1], cmap='jet', alpha=0.5, s=3 )
plt.title("UMAP")
plt.savefig("umap.png")

pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X[idx])
plt.figure(figsize=(32, 32))
plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='jet', alpha=0.5, s=3 )
plt.title("PCA")
plt.savefig("pca.png")

