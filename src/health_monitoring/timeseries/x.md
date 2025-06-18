| Method                            | Needs training data?                         | Notes                                                                             |
| --------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| **Z-score** / **IQR**             | ❌ No                                         | Compare each feature to the distribution (mean/std or quartiles) across samples   |
| **Mahalanobis Distance**          | ❌ No (if covariance can be estimated)        | Assumes multivariate Gaussian; needs enough samples to estimate mean + covariance |
| **PCA for anomaly detection**     | ❌ No (uses data to find dominant components) | Projects data and measures reconstruction error; unsupervised                     |
| **Isolation Forest**              | ❌ No                                         | Builds random trees to isolate points; doesn’t need labels                        |
| **Local Outlier Factor (LOF)**    | ❌ No                                         | Compares local density to neighbors; fully unsupervised                           |
| **One-Class SVM**                 | ✅ Yes (needs training on "normal" data)      | Requires a known clean dataset; not usable without it                             |
| **Autoencoders**                  | ✅ Yes                                        | Needs training phase on normal data to learn reconstruction                       |
| **Gaussian Mixture Models (GMM)** | ❌ No                                         | Can be fitted unsupervised; anomalies are low-likelihood points                   |
| **DBSCAN**                        | ❌ No                                         | Clustering algorithm; labels small/noise clusters as anomalies                    |


| Method type               | Training required? | Use case                                   |
| ------------------------- | ------------------ | ------------------------------------------ |
| **Statistical**           | ❌ No               | Great for small data, interpretable        |
| **Distance-based**        | ❌ No               | LOF, Mahalanobis, PCA                      |
| **Tree-based**            | ❌ No               | Isolation Forest is excellent unsupervised |
| **Neural models**         | ✅ Yes              | Autoencoders need training                 |
| **Model-based (1-class)** | ✅ Yes              | One-Class SVM needs clean training data    |
