"""RepSteer 0.2.4 estimators with explicitly weighted Safety adaptation."""

from __future__ import annotations
import torch
from repsteer.learners.actadd import paired_difference_mean
from repsteer.learners.pca import principal_components, _orient_by_group_difference
from repsteer.learners.lat import lat_components


def weights(labels):
    y = torch.as_tensor(labels)
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError("Both gold classes required")
    return torch.where(
        y == 0, len(y) / (2 * (y == 0).sum()), len(y) / (2 * (y == 1).sum())
    ).float()


def mean(x, w):
    return (x * w[:, None]).sum(0) / w.sum()


def unit(x):
    n = x.norm()
    if not torch.isfinite(n) or n <= 1e-12:
        raise ValueError("Degenerate learned direction")
    return x / n


def projection_sd(pos, neg, w, direction):
    x = torch.cat([pos, neg]) @ direction
    ww = torch.cat([w, w])
    mu = (x * ww).sum() / ww.sum()
    return (((x - mu).square() * ww).sum() / ww.sum()).sqrt()


def probe(x, y, w, l2=1e-4):
    """Same weighted objective as RepSteer Newton solver; scalable L-BFGS solve.

    Avoid a dense 4097×4097 Newton solve for every residual probe. Record gradient
    and objective, and fail on poor convergence rather than silently accepting it.
    """
    x = x.detach().double().cpu()
    y = y.double().cpu()
    w = w.double().cpu()
    w = w / w.sum()
    # Centering only reparameterizes the unpenalized intercept. It leaves the
    # original weighted objective and coefficient regularization unchanged.
    center = (x * w[:, None]).sum(0)
    original = x
    x = x - center
    beta = torch.zeros(x.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [beta],
        max_iter=5000,
        history_size=50,
        tolerance_grad=1e-10,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        z = x @ beta[:-1] + beta[-1]
        loss = ((torch.nn.functional.softplus(z) - y * z) * w).sum() + 0.5 * l2 * beta[
            :-1
        ].square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    objective = float(closure().detach())
    bias = beta[-1].detach() - center @ beta[:-1].detach()
    error = (torch.sigmoid(original @ beta[:-1].detach() + bias) - y) * w
    gradient = float(
        torch.cat([original.T @ error + l2 * beta[:-1].detach(), error.sum().view(1)])
        .abs()
        .max()
    )
    if gradient > 2e-5 or not torch.isfinite(beta).all():
        raise ValueError(f"Probe did not converge: {gradient}")
    return (
        beta[:-1].detach().float(),
        bias.float(),
        {
            "objective": objective,
            "max_gradient": gradient,
            "solver": "torch_LBFGS_centered_intercept",
            "l2": l2,
            "max_iter": 5000,
            "iterations": opt.state[beta]["n_iter"],
        },
    )


def fit_residual(method, pos, neg, labels, seed=42):
    w = weights(labels)
    metadata = {"repsteer": "0.2.4", "gold_class_weights": "equal_total", "seed": seed}
    if method == "caa":
        vector = paired_difference_mean(pos, neg, weights=w, normalize=None)
        unit(vector)
        metadata["estimator"] = "paired_difference_mean(normalize=None)"
    elif method == "pca":
        basis, _, variance = principal_components(
            torch.cat([pos, neg]), weights=torch.cat([w, w])
        )
        d = _orient_by_group_difference(
            basis, pos, neg, positive_weights=w, negative_weights=w
        )[0]
        vector = d * projection_sd(pos, neg, w, d)
        metadata.update(estimator="pooled_centered_PCA", variance=variance.tolist())
    elif method == "lat":
        basis, _, variance = lat_components(pos, neg, weights=w, seed=seed)
        d = basis[0]
        vector = d * projection_sd(pos, neg, w, d)
        metadata.update(
            estimator="signed_paired_difference_PCA",
            variance=variance.tolist(),
            orientation="weighted_pair_vote",
        )
    elif method == "linear_probe":
        coef, bias, fit = probe(
            torch.cat([pos, neg]),
            torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]),
            torch.cat([w, w]),
        )
        d = unit(coef)
        vector = d * projection_sd(pos, neg, w, d)
        metadata.update(
            estimator="weighted_logistic",
            solver=fit,
            probe_bias=float(bias),
            probe_weight=coef,
        )
    else:
        raise ValueError(method)
    unit(vector)
    return {
        "vector": vector,
        "metadata": metadata,
        "delta_norm_alpha1": float(vector.norm()),
    }


def fit_iti_layer(
    pos, neg, labels, validation_pos, validation_neg, validation_labels, layer
):
    w = weights(labels)
    vw = weights(validation_labels)
    ranking = []
    for head in range(pos.shape[1]):
        x = torch.cat([pos[:, head], neg[:, head]])
        y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        coef, bias, fit = probe(x, y, torch.cat([w, w]))
        vx = torch.cat([validation_pos[:, head], validation_neg[:, head]])
        vy = torch.cat(
            [torch.ones(len(validation_pos)), torch.zeros(len(validation_neg))]
        )
        ww = torch.cat([vw, vw])
        correct = ((vx @ coef + bias >= 0) == vy).float()
        score = float((correct * ww).sum() / ww.sum())
        shift = mean(pos[:, head] - neg[:, head], w)
        if shift.norm() <= 1e-12:
            continue
        d = unit(shift)
        sigma = projection_sd(pos[:, head], neg[:, head], w, d)
        if not torch.isfinite(sigma) or sigma <= 0:
            continue
        ranking.append(
            {
                "layer": layer,
                "head": head,
                "score": score,
                "direction": d,
                "sigma": float(sigma),
                "probe_fit": fit,
            }
        )
    return ranking
