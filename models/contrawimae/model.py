"""
ContraWiMAE model implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .wimae import WiMAE
from .modules import ContrastiveHead, apply_channel_augmentations


class ContraWiMAE(WiMAE):
    """
    Contrastive Wireless Masked Autoencoder (ContraWiMAE) model.

    This model extends WiMAE with contrastive learning capabilities,
    enabling the model to learn representations that are invariant
    to certain augmentations.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int],
        encoder_dim: int = 64,
        encoder_layers: int = 12,
        encoder_nhead: int = 8,
        decoder_layers: int = 8,
        decoder_nhead: int = 8,
        mask_ratio: float = 0.6,
        contrastive_dim: int = 64,
        temperature: float = 0.1,
        snr_min: float = 0.0,
        snr_max: float = 30.0,
        max_len: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the ContraWiMAE model.

        Args:
            patch_size: Size of patches (height, width)
            encoder_dim: Dimension of encoder
            encoder_layers: Number of encoder layers
            encoder_nhead: Number of encoder attention heads
            decoder_layers: Number of decoder layers
            decoder_nhead: Number of decoder attention heads
            mask_ratio: Ratio of patches to mask during training
            contrastive_dim: Dimension of contrastive projection
            temperature: Temperature parameter for contrastive loss
            snr_min: Minimum SNR for augmentations
            snr_max: Maximum SNR for augmentations
            device: Device to place the model on
        """

        super().__init__(
            patch_size=patch_size,
            encoder_dim=encoder_dim,
            encoder_layers=encoder_layers,
            encoder_nhead=encoder_nhead,
            decoder_layers=decoder_layers,
            decoder_nhead=decoder_nhead,
            mask_ratio=mask_ratio,
            max_len=max_len,
            device=device,
        )

        self.contrastive_head = ContrastiveHead(
            input_dim=encoder_dim,
            proj_dim=contrastive_dim,
        ).to(self.device)

        self.contrastive_dim = contrastive_dim

        self.snr_min = snr_min
        self.snr_max = snr_max
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
        return_reconstruction: bool = True,
        return_contrastive: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ContraWiMAE model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            mask_ratio: Masking ratio (uses self.mask_ratio if None)
            return_reconstruction: Whether to return reconstruction
            return_contrastive: Whether to return contrastive features

        Returns:
            Dictionary containing encoded features and optionally reconstruction/contrastive features
        """

        output = super().forward(x, mask_ratio, return_reconstruction)

        if return_contrastive:
            encoded_features = output["encoded_features"]
            contrastive_features = self.contrastive_head(encoded_features)
            output["contrastive_features"] = contrastive_features

        return output

    def forward_with_augmentation(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
        return_reconstruction: bool = True,
        noise_prob: float = 1.0,
        freq_shift_prob: float = 0.0,
        phase_rot_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with augmentation for contrastive learning.

        Args:
            x: Input tensor
            mask_ratio: Masking ratio
            return_reconstruction: Whether to return reconstruction
            noise_prob: Probability of applying noise injection
            freq_shift_prob: Probability of applying frequency shift
            phase_rot_prob: Probability of applying phase rotation

        Returns:
            Dictionary containing outputs for original and augmented inputs
        """

        x_aug = apply_channel_augmentations(
            x,
            noise_prob=noise_prob,
            freq_shift_prob=freq_shift_prob,
            phase_rot_prob=phase_rot_prob,
            snr_min=self.snr_min,
            snr_max=self.snr_max,
        )

        output_orig = self.forward(
            x,
            mask_ratio=mask_ratio,
            return_reconstruction=return_reconstruction,
            return_contrastive=True,
        )

        output_aug = self.forward(
            x_aug,
            mask_ratio=mask_ratio,
            return_reconstruction=return_reconstruction,
            return_contrastive=True,
        )

        combined_output = {
            "original": output_orig,
            "augmented": output_aug,
        }

        return combined_output

    def compute_contrastive_loss(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of features.

        Args:
            features1: First set of features
            features2: Second set of features
            temperature: Temperature parameter (uses self.temperature if None)

        Returns:
            Contrastive loss
        """
        if temperature is None:
            temperature = self.temperature

        return self.contrastive_head.compute_contrastive_loss(
            features1, features2, temperature
        )

    def get_contrastive_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get contrastive embeddings from encoded features.

        The contrastive head already performs mean pooling over patches.

        Args:
            x: Input tensor

        Returns:
            Contrastive embeddings
        """
        encoded_features = self.encode(x)
        return self.contrastive_head(encoded_features)

    def save_checkpoint(self, filepath: str, **kwargs):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            **kwargs: Additional data to save
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "patch_size": self.patch_size,
            "encoder_dim": self.encoder_dim,
            "encoder_layers": len(self.encoder.transformer.layers),
            "encoder_nhead": self.encoder.transformer.layers[0].self_attn.num_heads,
            "decoder_layers": len(self.decoder.transformer.layers),
            "decoder_nhead": self.decoder.transformer.layers[0].self_attn.num_heads,
            "mask_ratio": self.mask_ratio,
            "max_len": self.max_len,
            "contrastive_dim": self.contrastive_head.proj_dim,
            "temperature": self.temperature,
            "snr_min": self.snr_min,
            "snr_max": self.snr_max,
            **kwargs,
        }

        torch.save(checkpoint, filepath)

    @classmethod
    def from_checkpoint(
        cls, filepath: str, device: Optional[torch.device] = None
    ) -> "ContraWiMAE":
        """
        Load model from checkpoint.

        Args:
            filepath: Path to checkpoint file
            device: Device to load model on

        Returns:
            Loaded ContraWiMAE model
        """

        checkpoint = torch.load(filepath, map_location=device, weights_only=True)

        patch_size = checkpoint["patch_size"]
        encoder_dim = checkpoint["encoder_dim"]
        encoder_layers = checkpoint["encoder_layers"]
        encoder_nhead = checkpoint["encoder_nhead"]
        decoder_layers = checkpoint["decoder_layers"]
        decoder_nhead = checkpoint["decoder_nhead"]
        mask_ratio = checkpoint["mask_ratio"]
        contrastive_dim = checkpoint.get("contrastive_dim", encoder_dim)
        temperature = checkpoint.get("temperature", 0.1)
        snr_min = checkpoint.get("snr_min", 0.0)
        snr_max = checkpoint.get("snr_max", 30.0)
        max_len = checkpoint.get("max_len", 128)

        model = cls(
            patch_size=patch_size,
            encoder_dim=encoder_dim,
            encoder_layers=encoder_layers,
            encoder_nhead=encoder_nhead,
            decoder_layers=decoder_layers,
            decoder_nhead=decoder_nhead,
            mask_ratio=mask_ratio,
            contrastive_dim=contrastive_dim,
            temperature=temperature,
            snr_min=snr_min,
            snr_max=snr_max,
            max_len=max_len,
            device=device,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.

        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        base_info.update(
            {
                "model_type": "ContraWiMAE",
                "contrastive_dim": self.contrastive_head.proj_dim,
                "temperature": self.temperature,
                "snr_min": self.snr_min,
                "snr_max": self.snr_max,
            }
        )

        return base_info

    def compute_training_losses(
        self,
        x: torch.Tensor,
        criterion: nn.Module,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for ContraWiMAE with 2 forward passes.

        Args:
            x: Input tensor
            criterion: Loss function for reconstruction
            mask_ratio: Masking ratio (uses self.mask_ratio if None)

        Returns:
            Dictionary containing:
                - reconstruction_loss: Loss on original masked patches
                - contrastive_loss: Loss between visible patches of original and augmented
                - total_loss: Combined loss
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        x_aug = apply_channel_augmentations(
            x, snr_min=self.snr_min, snr_max=self.snr_max
        )

        orig_output = self.forward(x, mask_ratio=mask_ratio, return_contrastive=True)
        orig_reconstructed = orig_output["reconstructed_patches"]
        orig_ids_mask = orig_output["ids_mask"]
        orig_encoded = orig_output["encoded_features"]

        aug_output = self.forward(x_aug, mask_ratio=mask_ratio, return_contrastive=True)
        aug_encoded = aug_output["encoded_features"]

        if orig_ids_mask.shape[1] > 0:
            batch_size = x.shape[0]
            batch_indices = (
                torch.arange(batch_size, device=x.device)
                .unsqueeze(-1)
                .expand(-1, orig_ids_mask.shape[1])
            )

            patches = self.patcher(x)
            target_masked = patches[batch_indices, orig_ids_mask]
            recon_masked = orig_reconstructed[batch_indices, orig_ids_mask]

            reconstruction_loss = criterion(recon_masked, target_masked)
        else:
            reconstruction_loss = torch.tensor(0.0, device=x.device, requires_grad=True)

        z_i = self.contrastive_head(orig_encoded)
        z_j = self.contrastive_head(aug_encoded)

        contrastive_loss = self.compute_contrastive_loss(
            z_i, z_j, temperature=self.temperature
        )

        return {
            "reconstruction_loss": reconstruction_loss,
            "contrastive_loss": contrastive_loss,
            "total_loss": reconstruction_loss + contrastive_loss,
        }

    def compute_validation_losses(
        self,
        x: torch.Tensor,
        criterion: nn.Module,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute validation losses for ContraWiMAE.

        Args:
            x: Input tensor
            criterion: Loss function for reconstruction
            mask_ratio: Masking ratio (uses self.mask_ratio if None)

        Returns:
            Dictionary containing:
                - masked_recon_loss: Reconstruction loss on masked patches (primary)
                - full_recon_loss: Reconstruction loss on full input (secondary)
                - contrastive_loss: Contrastive loss between original and augmented
                - masked_loss: Combined masked reconstruction + contrastive loss
                - full_loss: Combined full reconstruction + contrastive loss
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        patches = self.patcher(x)

        masked_output = self.forward_with_augmentation(x, mask_ratio=mask_ratio)
        orig_masked_output = masked_output["original"]
        aug_masked_output = masked_output["augmented"]

        orig_ids_mask = orig_masked_output["ids_mask"]
        aug_ids_mask = aug_masked_output["ids_mask"]

        if orig_ids_mask.shape[1] > 0:
            batch_size = patches.shape[0]
            batch_indices = (
                torch.arange(batch_size, device=x.device)
                .unsqueeze(-1)
                .expand(-1, orig_ids_mask.shape[1])
            )
            orig_recon_masked = orig_masked_output["reconstructed_patches"][
                batch_indices, orig_ids_mask
            ]
            target_masked = patches[batch_indices, orig_ids_mask]
            orig_masked_recon_loss = criterion(orig_recon_masked, target_masked)
        else:
            orig_masked_recon_loss = torch.tensor(0.0, device=x.device)

        if aug_ids_mask.shape[1] > 0:
            batch_size = patches.shape[0]
            batch_indices = (
                torch.arange(batch_size, device=x.device)
                .unsqueeze(-1)
                .expand(-1, aug_ids_mask.shape[1])
            )
            aug_recon_masked = aug_masked_output["reconstructed_patches"][
                batch_indices, aug_ids_mask
            ]
            target_masked = patches[batch_indices, aug_ids_mask]
            aug_masked_recon_loss = criterion(aug_recon_masked, target_masked)
        else:
            aug_masked_recon_loss = torch.tensor(0.0, device=x.device)

        masked_recon_loss = (orig_masked_recon_loss + aug_masked_recon_loss) / 2

        full_output = self.forward_with_augmentation(x, mask_ratio=0.0)
        orig_full_output = full_output["original"]
        aug_full_output = full_output["augmented"]

        orig_full_recon_loss = criterion(
            orig_full_output["reconstructed_patches"], patches
        )
        aug_full_recon_loss = criterion(
            aug_full_output["reconstructed_patches"], patches
        )
        full_recon_loss = (orig_full_recon_loss + aug_full_recon_loss) / 2

        orig_features = orig_masked_output["contrastive_features"]
        aug_features = aug_masked_output["contrastive_features"]

        contrastive_loss = self.compute_contrastive_loss(
            orig_features, aug_features, temperature=self.temperature
        )

        return {
            "masked_recon_loss": masked_recon_loss,
            "full_recon_loss": full_recon_loss,
            "contrastive_loss": contrastive_loss,
            "masked_loss": masked_recon_loss + contrastive_loss,
            "full_loss": full_recon_loss + contrastive_loss,
        }
