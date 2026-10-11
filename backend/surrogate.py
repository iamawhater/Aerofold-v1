"""Exact Gaussian process surrogates implemented in double-precision PyTorch."""

from __future__ import annotations

import math

import numpy as np
import torch


class ExactGP:
    """RBF GP with learned lengthscale, signal variance and observation noise.

    Returned standard deviation is latent (epistemic), excluding assay noise.
    Input and outcome normalization use training observations only.
    """

    def fit(self, x: np.ndarray, y: np.ndarray, steps: int = 65) -> ExactGP:
        self.x_mean = x.mean(0)
        self.x_scale = np.maximum(x.std(0), 0.05)
        self.y_mean = float(y.mean())
        self.y_scale = max(float(y.std()), 0.1)
        self.x = torch.tensor((x - self.x_mean) / self.x_scale, dtype=torch.float64)
        target = torch.tensor((y - self.y_mean) / self.y_scale, dtype=torch.float64)
        self.params = torch.nn.Parameter(torch.tensor([0.0, 0.0, -3.0], dtype=torch.float64))
        optimizer = torch.optim.Adam([self.params], lr=0.06)
        identity = torch.eye(len(x), dtype=torch.float64)
        for _ in range(steps):
            optimizer.zero_grad()
            covariance = self.kernel(self.x, self.x) + (self.params[2].exp() + 1e-6) * identity
            chol = torch.linalg.cholesky(covariance)
            alpha = torch.cholesky_solve(target[:, None], chol)
            loss = (
                0.5 * (target[:, None] * alpha).sum()
                + chol.diag().log().sum()
                + len(x) / 2 * math.log(2 * math.pi)
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.params.clamp_(-7, 5)
        with torch.no_grad():
            self.chol = torch.linalg.cholesky(
                self.kernel(self.x, self.x) + (self.params[2].exp() + 1e-6) * identity
            )
            self.alpha = torch.cholesky_solve(target[:, None], self.chol)
        return self

    def kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        squared = torch.cdist(a, b).square() / a.shape[1]
        return self.params[1].exp() * torch.exp(-0.5 * squared / self.params[0].exp().square())

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            inputs = torch.tensor((x - self.x_mean) / self.x_scale, dtype=torch.float64)
            cross = self.kernel(self.x, inputs)
            mean = (cross.T @ self.alpha).squeeze(1)
            solve = torch.linalg.solve_triangular(self.chol, cross, upper=False)
            variance = (self.params[1].exp() - solve.square().sum(0)).clamp_min(1e-10)
            return mean.numpy() * self.y_scale + self.y_mean, variance.sqrt().numpy() * self.y_scale


class CorrelatedGP:
    """Two-output intrinsic coregionalization GP for paired assay observations.

    A shared linear plus RBF sequence kernel captures additive mutations and
    residual interactions. The learned task correlation couples binding and
    expression; independent observation noise remains task-specific. Returned
    covariance is latent uncertainty in the original outcome units.
    """

    def fit(self, x: np.ndarray, y: np.ndarray, steps: int = 55) -> CorrelatedGP:
        if x.ndim != 2 or y.shape != (len(x), 2) or len(x) < 3:
            raise ValueError("Expected paired two-objective observations at every training sequence")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Training features and targets must be finite")
        self.x_mean = x.mean(0)
        centered = x - self.x_mean
        self.x_scale = max(float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))), 1e-3)
        self.y_mean = y.mean(0)
        self.y_scale = np.maximum(y.std(0), np.array([0.1, 0.1]))
        self.x = torch.as_tensor(centered / self.x_scale, dtype=torch.float64)
        target = torch.as_tensor((y - self.y_mean) / self.y_scale, dtype=torch.float64).reshape(-1, 1)
        # log(linear signal), log(RBF signal), log(lengthscale),
        # raw task correlation, log(binding noise), log(expression noise).
        self.params = torch.nn.Parameter(torch.tensor([0.0, -1.0, 0.0, 0.0, -2.3, -2.3], dtype=torch.float64))
        optimizer = torch.optim.Adam([self.params], lr=0.045)
        identity = torch.eye(2 * len(x), dtype=torch.float64)
        for _ in range(steps):
            optimizer.zero_grad()
            covariance = self._training_covariance() + 1e-5 * identity
            chol = torch.linalg.cholesky(covariance)
            alpha = torch.cholesky_solve(target, chol)
            loss = 0.5 * (target * alpha).sum() + chol.diag().log().sum()
            loss = loss + len(x) * math.log(2 * math.pi)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.params[:3].clamp_(-6, 5)
                self.params[3].clamp_(-2.5, 2.5)
                self.params[4:].clamp_(-7, 2)
        with torch.no_grad():
            self.chol = torch.linalg.cholesky(self._training_covariance() + 1e-5 * identity)
            self.alpha = torch.cholesky_solve(target, self.chol)
        return self

    def _task_covariance(self) -> torch.Tensor:
        correlation = 0.95 * torch.tanh(self.params[3])
        one = torch.ones((), dtype=torch.float64)
        return torch.stack((torch.stack((one, correlation)), torch.stack((correlation, one))))

    def _kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        linear = (a @ b.T) * self.params[0].exp()
        lengthscale = self.params[2].exp()
        rbf = self.params[1].exp() * torch.exp(-torch.cdist(a, b).square() / (2 * lengthscale**2))
        return linear + rbf

    def _training_covariance(self) -> torch.Tensor:
        covariance = torch.kron(self._kernel(self.x, self.x), self._task_covariance())
        noise = self.params[4:].exp().repeat(len(self.x))
        return covariance + torch.diag(noise)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mean, marginal standard deviations and 2x2 latent covariances."""
        with torch.no_grad():
            inputs = torch.as_tensor((x - self.x_mean) / self.x_scale, dtype=torch.float64)
            task_covariance = self._task_covariance()
            cross = torch.kron(self._kernel(self.x, inputs), task_covariance)
            mean = (cross.T @ self.alpha).reshape(-1, 2)
            solve = torch.linalg.solve_triangular(self.chol, cross, upper=False)
            solve = solve.reshape(2 * len(self.x), len(inputs), 2)
            reduction = torch.einsum("nmi,nmj->mij", solve, solve)
            prior_variance = self.params[0].exp() * inputs.square().sum(1) + self.params[1].exp()
            covariance = prior_variance[:, None, None] * task_covariance[None] - reduction
            covariance = 0.5 * (covariance + covariance.transpose(1, 2))
            covariance = covariance * torch.as_tensor(self.y_scale[None, :, None] * self.y_scale[None, None, :])
            diagonal = covariance.diagonal(dim1=1, dim2=2).clamp_min(1e-10)
            scaled_mean = mean.numpy() * self.y_scale + self.y_mean
            return scaled_mean, diagonal.sqrt().numpy(), covariance.numpy()
