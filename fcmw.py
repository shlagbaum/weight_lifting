from typing import Optional
import numpy as np
from pydantic import BaseModel, Extra, Field, validate_arguments
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


class _ArrayMeta(type):

    def __getitem__(self, t):

        return type("ArrayLike", (ArrayLike,), {"__dtype__": t})

class ArrayLike(np.ndarray, metaclass=_ArrayMeta):

    @classmethod
    def __get_validators__(cls):

        yield cls.validate_type



    @classmethod
    def validate_type(cls, val):

        if isinstance(val, np.ndarray):

            return val

        raise ValueError(f"{val} is not an instance of numpy.ndarray")

class FCMWD(BaseModel):
    n_clusters: int = Field(8, ge=1, le=100)
    max_iter: int = Field(300, ge=0, le=10000)
    m: float = Field(2.0, ge=1.0)
    error: float = Field(1e-5, ge=1e-9)
    random_state: Optional[int] = None
    trained: bool = Field(False, const=True)

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True

    @validate_arguments
    def fit(self, K: ArrayLike, C: ArrayLike = None, uinit: ArrayLike = None, sample_weights: ArrayLike = None) -> ArrayLike:
        """Train the fuzzy-c-means model..
        Parameters
        ----------
        K : array-like, shape = [n_samples, n_samples]
            Training instances to cluster.
        """
        n_samples = K.shape[0]
        self.sample_weights = np.ones(n_samples) if sample_weights is None else np.reshape(sample_weights, n_samples)
        if C is None:
            self.C = np.ones(n_samples)
        else:
            self.C = np.reshape(np.array(C), (n_samples))
        if self.n_clusters < 2:
            self.u = self.C ** (1.0 / self.m)
            d = self._kernel_dist(K);
            self.u = min(d) / d
        else:
            if uinit is None:
                self.rng = np.random.default_rng(self.random_state)
                self.u = self.rng.uniform(size=(n_samples, self.n_clusters))
            else: 
                self.u = uinit
            self.u = self.u / np.tile(
                (self.u.sum(axis=1)/self.C)[np.newaxis].T, self.n_clusters
            )
            w = np.sqrt(self.sample_weights)
            for _ in range(self.max_iter):
                u_old = self.u.copy()
                self.u = self.soft_predict(K, self.C)
                # Stopping rule
                dlt = self.u - u_old
                for j in range(dlt.shape[1]): 
                    dlt[:,j] = dlt[:,j]*w
                if np.linalg.norm(dlt) < self.error:
                    break
        self.trained = True
        return self.u

    def soft_predict(self, K: ArrayLike, C: ArrayLike = None) -> ArrayLike:
        """Soft predict of FCM
        Parameters
        ----------
        K : array, shape = [n_samples, n_samples]
            New data to predict.
        Returns
        -------
        array, shape = [n_samples, n_clusters]
            Fuzzy partition array, returned as an array with n_samples rows
            and n_clusters columns.
        """
        if C is None:
            C = np.ones(K.shape[0])
    
        temp = self._kernel_dist(K)
        if self.m <= 1.0:
            idx = temp.argmin(axis = 0)
            temp[:,:] = 0.0
            temp[:,idx] = 1.0
        else:
            temp = temp ** float(2 / (self.m - 1))
        temp = 1.0 / temp.T
        temp = temp / temp.sum(axis=0)
        return C.reshape((temp.shape[-1],1)).repeat(temp.shape[0], axis=1) * (temp.T)

    def _kernel_dist(self, K: ArrayLike):
        """Compute distances in the feature space"""
        W = ((self.u ** self.m) / (self.u ** self.m).sum(axis=0)).reshape((K.shape[0], self.n_clusters))
        Kw = (K @ W).reshape((K.shape[0], self.n_clusters))
        sqnorm = np.einsum("ii->i", W.T @ Kw)
        return np.sqrt(np.einsum("ii->i", K).repeat(self.n_clusters, axis = 0).reshape((K.shape[0], self.n_clusters)) + sqnorm - 2*Kw)

    @validate_arguments
    def predict(self, K: ArrayLike, C: ArrayLike = None, mode = 0):
        """Predict meberships (0), the closest cluster each sample in X belongs to(1), distances to cluster centers(2).
        Parameters
        ----------
        K : array, shape = [n, n_samples]
            New data to predict.
        Returns
        -------
        labels : array, shape = [n_samples,]
            Index of the cluster each sample belongs to.
        """
        if self.is_trained():
            if mode < 2:
                out = self.soft_predict(K, C)
                if mode > 0:
                    out = out.argmax(axis=-1)
            else:
                out = self._kernel_dist(K)
            return out
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )


    def is_trained(self) -> bool:
        if self.trained:
            return True
        return False

    @property
    def partition_coefficient(self) -> float:
        """Partition coefficient
        (Equation 12a of https://doi.org/10.1016/0098-3004(84)90020-7)
        Returns
        -------
        float
            partition coefficient of clustering model
        """
        if self.is_trained():
            return np.mean(self.u**2)
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    @property
    def partition_entropy_coefficient(self):
        if self.is_trained():
            return -np.mean(self.u * np.log2(self.u))
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )
        
class FCMWD1(FCMWD):
    def _kernel_dist(self, K: ArrayLike):
        """Compute distances to cluster centers in the feature space"""
        K = np.reshape(K, (len(K), 1))
        W = ((self.u ** self.m) / (self.u ** self.m).sum(axis=0)).reshape((K.shape[0], self.n_clusters))
        Kw = K.T @ W
        Kw = K @ Kw
        sqnorm = np.einsum("ii->i", W.T @ Kw)
        return np.sqrt((K * K).repeat(self.n_clusters, axis = 0).reshape((K.shape[0], self.n_clusters)) + sqnorm - 2*Kw)
        

class FCMW(FCMWD):
    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True

    @validate_arguments
    def fit(self, X: ArrayLike, C: ArrayLike = None, uinit: ArrayLike = None, centers_init: ArrayLike = None) -> ArrayLike:
        """Train the fuzzy-c-means model..
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training instances to cluster.
        """
        n_samples = X.shape[0]
        if C is None:
            self.C = np.ones(n_samples)
        else:
            self.C = np.reshape(np.array(C), (n_samples))
        if self.n_clusters < 2:
           self._centers =  FCMW._next_centers(X, self.C, 1.0)
           temp = FCMW._dist(X, self._centers)
           self.u = min(temp) / temp
        else:
            if uinit is None:
                self.rng = np.random.default_rng(self.random_state)
                self.u = self.rng.uniform(size=(n_samples, self.n_clusters))
            else: 
                self.u = uinit
            self.u = self.u / np.tile(
                (self.u.sum(axis=1)/self.C)[np.newaxis].T, self.n_clusters
            )
            if not centers_init is None:
                self._centers = centers_init
                self.u = self.soft_predict(X, self.C)
            for _ in range(self.max_iter):
                u_old = self.u.copy()
                self._centers = FCMW._next_centers(X, self.u, self.m)
                self.u = self.soft_predict(X, self.C)
                # Stopping rule
                if np.linalg.norm(self.u - u_old) < self.error:
                    break
        self.trained = True
        return self.u

    def soft_predict(self, X: ArrayLike, C: ArrayLike = None) -> ArrayLike:
        """Soft predict of FCM
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        array, shape = [n_samples, n_clusters]
            Fuzzy partition array, returned as an array with n_samples rows
            and n_clusters columns.
        """
        if C is None:
            C = self.C
    
        temp = FCMW._dist(X, self._centers)
        if self.m <= 1.0:
            idx = temp.argmin(axis = 0)
            temp[:,:] = 0.0
            temp[:,idx] = 1.0
        else:
            temp = temp ** float(2 / (self.m - 1))
        temp = 1.0 / temp.T
        temp = temp / temp.sum(axis=0)
        return C.reshape((temp.shape[-1],1)).repeat(temp.shape[0], axis=1) * (temp.T)

    @validate_arguments
    def predict(self, X: ArrayLike, C: ArrayLike = None, mode = 0):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape = [n_samples,]
            Index of the cluster each sample belongs to.
        """
        if self.is_trained():
            X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
            if mode < 2:
                out = self.soft_predict(X, C)
                if mode > 0:
                    out = out.argmax(axis=-1)
            else:
                out = FCMW._dist(X, self._centers)
            return out
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    def is_trained(self) -> bool:
        if self.trained:
            return True
        return False

    @staticmethod
    def _dist(A: ArrayLike, B: ArrayLike):
        """Compute the euclidean distance two matrices"""
        return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2))

    @staticmethod
    def _next_centers(X, u, m):
        """Update cluster centers"""
        um = u**m
        return (X.T @ um / np.sum(um, axis=0)).T

    @property
    def centers(self):
        if self.is_trained():
            return self._centers
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    @property
    def partition_coefficient(self) -> float:
        """Partition coefficient
        (Equation 12a of https://doi.org/10.1016/0098-3004(84)90020-7)
        Returns
        -------
        float
            partition coefficient of clustering model
        """
        if self.is_trained():
            return np.mean(self.u**2)
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    @property
    def partition_entropy_coefficient(self):
        if self.is_trained():
            return -np.mean(self.u * np.log2(self.u))
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

def dbscann(X, num_clusters = 2, eps_values = None, min_samples_values = None, **kwargs):
    # Define the range of values for eps and min_samples
    if eps_values is None:
        eps_values = np.linspace(0.1, 1, 10)
    if min_samples_values is None:
        min_samples_values = np.arange(2, 10, 1)

    # Store the best silhouette score and the corresponding eps and min_samples values
    best_silhouette_score = -1
    best_eps = None
    best_min_samples = None
    best_labels = None

    # Loop over all possible combinations of eps and min_samples
    for eps in eps_values:
        for min_samples in min_samples_values:
            
            # Fit the DBSCAN model
            model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
            model.fit(X)
            
            # Check if the number of clusters is equal to the desired number of clusters
            if len(np.unique(model.labels_)) != num_clusters:
                continue
            
            # Compute the silhouette score for the current model
            silhouette_score_ = silhouette_score(X, model.labels_)
            
            # If the silhouette score is better than the best silhouette score so far, update the best silhouette score
            if silhouette_score_ > best_silhouette_score:
                best_silhouette_score = silhouette_score_
                best_eps = eps
                best_min_samples = min_samples
                best_labels = model.labels_
                
    return {'labels': best_labels, 'eps': best_eps, 'min_samples': best_min_samples, 'silhouette_score': best_silhouette_score}


