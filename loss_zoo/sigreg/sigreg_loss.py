"""
Overview:
    SIGReg (Sketched Isotropic Gaussian Regularization) Loss implementation.

    This module provides a standalone implementation of the core regularization loss from LeJEPA,
    which tests whether learned embeddings follow a standard normal distribution using the
    empirical characteristic function and numerical integration.

    Key components:
        - SIGReg: Main loss module for embedding regularization

    Reference:
        LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics
        https://arxiv.org/abs/2511.08544
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Overview:
        Sketched Isotropic Gaussian Regularization (SIGReg) loss function.

        This loss measures how far a batch of embeddings deviates from a standard normal distribution
        by comparing the empirical characteristic function (CF) with the theoretical CF of N(0,I).
        The comparison is done via numerical integration (Simpson's rule) weighted by a Gaussian window.

        The core idea is to:
        1. Project embeddings to lower dimensions via random sketching (reduces computation)
        2. Compute empirical CF at multiple frequency points (quadrature nodes)
        3. Compare with theoretical CF of standard normal: exp(-t²/2)
        4. Integrate the squared error to get a scalar statistic

    Arguments:
        - knots (:obj:`int`): Number of quadrature nodes for numerical integration.
            More knots provide better approximation but increase computation. Default: 17.
            Recommended range: [11, 25].

    Shapes:
        - Input (:obj:`torch.Tensor`): :math:`(B, D)`, where B is batch size, D is embedding dimension
        - Output (:obj:`torch.Tensor`): :math:`()`, scalar loss value

    Notes:
        - The random projection matrix is regenerated at each forward pass for better estimation
        - The sketching dimension (256) is a good balance between accuracy and efficiency
        - The integration range [0, 3] covers the most informative part of the CF
        - Uses symmetric property of CF: integrate on [0, t_max] and double, instead of [-t_max, t_max]
        - The loss is scale-invariant due to L2 normalization of projection matrix

    Examples::
        >>> sigreg = SIGReg(knots=17)
        >>> embeddings = torch.randn(256, 128)  # batch_size=256, embedding_dim=128
        >>> loss = sigreg(embeddings)
        >>> print(f"SIGReg loss: {loss.item():.4f}")
    """

    def __init__(self, knots: int = 17):
        super().__init__()

        # Setup quadrature nodes uniformly in [0, 3]
        # NOTE: Range [0, 3] is chosen to cover the most informative frequencies
        t = torch.linspace(0, 3, knots, dtype=torch.float32)

        # Compute quadrature weights using Simpson's rule
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Boundary correction for Simpson's rule

        # Gaussian window: exp(-t²/2)
        # NOTE: This is the theoretical CF of standard normal distribution N(0,1)
        window = torch.exp(-t.square() / 2.0)

        # Register as buffers so they automatically move with model.to(device)
        self.register_buffer("t", t)  # Quadrature nodes
        self.register_buffer("phi", window)  # Theoretical CF values
        self.register_buffer("weights", weights * window)  # Combined weights for integration

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Compute the SIGReg loss for a batch of embeddings.

            The computation flow:
            1. Generate random projection matrix A: [D, 256]
            2. Project embeddings: z = proj @ A  -> [B, 256]
            3. Compute empirical CF at each quadrature node t
            4. Compare with theoretical CF and compute squared error
            5. Integrate using numerical quadrature to get final statistic

        Arguments:
            - proj (:obj:`torch.Tensor`): Input embeddings of shape :math:`(B, D)`,
                where B is batch size and D is embedding dimension.

        Returns:
            - loss (:obj:`torch.Tensor`): Scalar loss value measuring deviation from standard normal.
                Higher values indicate embeddings are farther from N(0,I).

        Shapes:
            - proj (:obj:`torch.Tensor`): :math:`(B, D)`, batch of embeddings
            - loss (:obj:`torch.Tensor`): :math:`()`, scalar

        Notes:
            - Random matrix A is regenerated each call (Monte Carlo estimation)
            - Column normalization of A ensures scale invariance
            - Sketching dimension 256 is empirically sufficient for most cases
            - The batch size B appears in the final statistic as a scaling factor

        Examples::
            >>> sigreg = SIGReg()
            >>> embeddings = torch.randn(64, 128)
            >>> loss = sigreg(embeddings)
            >>> loss.backward()  # Can be used for gradient-based optimization
        """
        device = proj.device
        embedding_dim = proj.size(-1)
        batch_size = proj.size(0)

        # Generate random sketching matrix: [embedding_dim, 256]
        # NOTE: 256 is the sketching dimension, balancing accuracy and efficiency
        A = torch.randn(embedding_dim, 256, device=device)

        # L2 normalize each column to make the projection scale-invariant
        A = A.div_(A.norm(p=2, dim=0))  # Each column has unit norm

        # Project embeddings to 256 dimensions: [batch_size, 256]
        z = proj @ A

        # Compute z * t for all quadrature nodes
        # Shape: [batch_size, 256, num_knots]
        x_t = z.unsqueeze(-1) * self.t

        # Empirical characteristic function: E[exp(i*z*t)] = E[cos(z*t)] + i*E[sin(z*t)]
        # For real-valued statistic, we use |E[exp(i*z*t)] - phi_0(t)|^2
        #   = |E[cos(z*t)] - phi(t)|^2 + |E[sin(z*t)]|^2
        # where phi(t) = exp(-t²/2) is the theoretical CF of N(0,1)

        cos_term = x_t.cos().mean(0)  # E[cos(z*t)], shape: [256, num_knots]
        sin_term = x_t.sin().mean(0)  # E[sin(z*t)], shape: [256, num_knots]

        # Squared error in characteristic function space
        # Shape: [256, num_knots]
        err = (cos_term - self.phi).square() + sin_term.square()

        # Numerical integration using precomputed weights
        # Multiply by batch_size as per the statistical test definition
        # Shape: [256]
        statistic = (err @ self.weights) * batch_size

        # Return mean across all 256 random projections
        return statistic.mean()


if __name__ == "__main__":
    """
    Overview:
        Simple demonstration of SIGReg loss usage.

        This example shows:
        1. How to create the loss module
        2. How to compute loss on random embeddings
        3. How to use it in a gradient-based optimization loop
    """
    print("=" * 70)
    print("SIGReg Loss - Simple Demo")
    print("=" * 70)

    # Configuration
    batch_size = 256
    embedding_dim = 128

    # Create SIGReg loss module
    sigreg = SIGReg(knots=17)
    print(f"\n✓ Created SIGReg loss with {sigreg.t.shape[0]} quadrature nodes")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, embedding_dim)
    )
    print(f"✓ Created simple model: 784 -> 256 -> {embedding_dim}")

    # Create dummy input data
    x = torch.randn(batch_size, 784)
    print(f"✓ Generated input data: {x.shape}")

    # Forward pass
    embeddings = model(x)
    print(f"✓ Forward pass complete, embeddings: {embeddings.shape}")

    # Compute SIGReg loss
    loss = sigreg(embeddings)
    print(f"\n📊 SIGReg loss: {loss.item():.6f}")

    # Demonstrate gradient flow
    loss.backward()
    print(f"✓ Gradients computed successfully")

    # Check gradient magnitude
    grad_norm = model[0].weight.grad.norm().item()
    print(f"✓ Model gradient norm: {grad_norm:.6f}")

    # Demonstrate with normalized embeddings (should give lower loss)
    print(f"\n{'-' * 70}")
    print("Testing with normalized embeddings (closer to N(0,I)):")
    print(f"{'-' * 70}")

    embeddings_normalized = torch.randn(batch_size, embedding_dim)
    loss_normalized = sigreg(embeddings_normalized)
    print(f"📊 SIGReg loss (normalized): {loss_normalized.item():.6f}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nUsage in your code:")
    print("  from sigreg_loss import SIGReg")
    print("  sigreg = SIGReg()")
    print("  loss = sigreg(embeddings)  # embeddings: [batch_size, embedding_dim]")
    print("=" * 70)
