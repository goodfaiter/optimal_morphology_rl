import torch
from optimal_morphology_rl.transfer.encoder import LatentSpaceEncoder, LatentSpaceDecoder
from optimal_morphology_rl.transfer.wrapper import EncoderWrapper, DecoderWrapper
import numpy as np
import os


class EncoderTrainer:
    def __init__(self, teacher_input_size, student_input_size, latent_size, device):
        self.teacher_encoder = LatentSpaceEncoder(teacher_input_size, latent_size, layers=[41]).to(device)
        self.teacher_decoder = LatentSpaceDecoder(latent_size, teacher_input_size, layers=[41]).to(device)

        self.student_encoder = LatentSpaceEncoder(student_input_size, latent_size, layers=[41]).to(device)
        self.student_decoder = LatentSpaceDecoder(latent_size, student_input_size, layers=[41]).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.teacher_encoder.parameters())
            + list(self.teacher_decoder.parameters())
            + list(self.student_encoder.parameters())
            + list(self.student_decoder.parameters()),
            lr=1e-3,
        )

        with torch.no_grad():
            self.teacher_mean = torch.zeros(teacher_input_size, device=device)
            self.teacher_std = torch.ones(teacher_input_size, device=device)
            self.student_mean = torch.zeros(student_input_size, device=device)
            self.student_std = torch.ones(student_input_size, device=device)

    def _normalize(self, batch: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (batch - mean) / std

    def _compute_reconstruction_loss(self, z_t, z_s, target_t, target_s):
        """Compute reconstruction loss for both robots"""
        recon_t = self.teacher_decoder(z_t)
        recon_s = self.student_decoder(z_s)

        # MSE loss for reconstruction
        loss_t = torch.nn.MSELoss()(recon_t, target_t)
        loss_s = torch.nn.MSELoss()(recon_s, target_s)

        return loss_t + loss_s

    def _compute_alignment_loss(self, z_t, z_s):
        """Compute latent space alignment loss"""

        loss_align = torch.nn.MSELoss()(z_t, z_s)  # simple L2 loss for alignment (alternative to MMD)

        return loss_align

    def _compute_regularization_loss(self, z_t, z_s):
        """Prevent latent space collapse"""
        # L2 regularization
        l2_loss = torch.mean(z_t**2) + torch.mean(z_s**2)

        # Variance penalty to prevent constant latents
        var_penalty = -torch.std(z_t) - torch.std(z_s)

        return l2_loss + var_penalty

    def _compute_expressivity_loss(self, z_t, z_s):
        """Encourage latents to use full range of values"""

        # Penalize low variance along batch dimension
        var_t = torch.var(z_t, dim=0).mean()
        var_s = torch.var(z_s, dim=0).mean()

        # Target variance (normalized data → ~1.0)
        target_var = 1.0
        var_loss = torch.abs(var_t - target_var) + torch.abs(var_s - target_var)

        # Also penalize latents that are too close to zero
        # mean_abs_t = torch.abs(z_t).mean()
        # mean_abs_s = torch.abs(z_s).mean()

        # Encourage mean absolute value > 0.5 (for normalized range)
        # magnitude_loss = torch.relu(0.5 - mean_abs_t) + torch.relu(0.5 - mean_abs_s)

        return var_loss
    
    def _info_nce_loss(self, z_t, z_s, temperature=0.1):
        """
        InfoNCE loss for paired data.
        Positive pairs: (z_t[i], z_s[i]) for same i
        Negative pairs: (z_t[i], z_s[j]) for i ≠ j
        """
        batch_size = z_t.shape[0]
        
        # Normalize embeddings (optional but recommended)
        z_t = torch.nn.functional.normalize(z_t, dim=1)
        z_s = torch.nn.functional.normalize(z_s, dim=1)
        
        # Compute similarity matrix [batch_size, batch_size]
        similarity = torch.matmul(z_t, z_s.T) / temperature
        
        # Labels: positive pairs are on diagonal
        labels = torch.arange(batch_size, device=z_t.device)
        
        # Cross-entropy loss (each row: find correct student for each teacher)
        loss_t = torch.nn.CrossEntropyLoss()(similarity, labels)
        
        # Also compute from student to teacher (symmetric loss)
        loss_s = torch.nn.CrossEntropyLoss()(similarity.T, labels)
        
        return (loss_t + loss_s) / 2

    def train_step(self, batch_t, batch_s):
        self.optimizer.zero_grad()
        # Inputs are normalized once up-front in `train()`; use batches directly here.
        z_t = self.teacher_encoder(batch_t)
        z_s = self.student_encoder(batch_s)

        recon_loss = self._compute_reconstruction_loss(z_t, z_s, batch_t, batch_s)
        align_loss = self._compute_alignment_loss(z_t, z_s)
        # reg_loss = self._compute_regularization_loss(z_t, z_s)
        # exp_loss = self._compute_expressivity_loss(z_t, z_s)
        nce_loss = self._info_nce_loss(z_t, z_s)

        # Total loss
        total_loss = 1.0 * recon_loss + 0.1 * align_loss + 0.1 * nce_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "align": align_loss.item(),
            # "reg": reg_loss.item(),
            # "expressivity": exp_loss.item(),
            "nce": nce_loss.item(),
        }

    def train(self, teacher_data, student_data, epochs=100, batch_size=64, verbose=True):
        """Main training loop"""

        # Compute and store normalization statistics once for the full datasets
        self.teacher_mean[:] = teacher_data.mean(dim=0)
        self.teacher_std[:] = teacher_data.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.student_mean[:] = student_data.mean(dim=0)
        self.student_std[:] = student_data.std(dim=0, unbiased=False).clamp_min(1e-6)

        # Normalize datasets once up-front so training batches are already normalized
        teacher_data = (teacher_data - self.teacher_mean) / self.teacher_std
        student_data = (student_data - self.student_mean) / self.student_std

        # Create data loaders from normalized data
        teacher_dataset = torch.utils.data.TensorDataset(teacher_data)
        student_dataset = torch.utils.data.TensorDataset(student_data)

        teacher_loader = torch.utils.data.DataLoader(teacher_dataset, batch_size=batch_size, shuffle=False)
        student_loader = torch.utils.data.DataLoader(student_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        for epoch in range(epochs):
            epoch_losses = []

            # Create iterator that yields batches from both datasets
            for (batch_t,), (batch_s,) in zip(teacher_loader, student_loader):
                losses = self.train_step(batch_t, batch_s)
                epoch_losses.append(losses)

            # Average losses for the epoch
            avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()}

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch}/{epochs} - "
                    f"Total: {avg_losses['total']:.4f}, "
                    f"Recon: {avg_losses['recon']:.4f}, "
                    f"Align: {avg_losses['align']:.4f}, "
                    # f"Reg: {avg_losses['reg']:.4f}, "
                    # f"Expressivity: {avg_losses['expressivity']:.4f}, "
                    f"nce: {avg_losses['nce']:.4f}"
                )

        self.save_models("/workspace/data/transfer/models")

    def save_models(self, save_folder_path: str) -> None:
        os.makedirs(save_folder_path, exist_ok=True)

        wrapped_teacher_encoder = EncoderWrapper(self.teacher_encoder, self.teacher_mean, self.teacher_std)
        wrapped_teacher_decoder = DecoderWrapper(self.teacher_decoder, self.teacher_mean, self.teacher_std)
        wrapped_student_encoder = EncoderWrapper(self.student_encoder, self.student_mean, self.student_std)
        wrapped_student_decoder = DecoderWrapper(self.student_decoder, self.student_mean, self.student_std)

        wrapped_teacher_encoder.freeze()
        wrapped_teacher_decoder.freeze()
        wrapped_student_encoder.freeze()
        wrapped_student_decoder.freeze()

        wrapped_teacher_encoder.trace_and_save(
            f"{save_folder_path}/{self.teacher_encoder.input_size}_{self.teacher_encoder.latent_size}_encoder.pt"
        )
        wrapped_teacher_decoder.trace_and_save(
            f"{save_folder_path}/{self.teacher_decoder.latent_size}_{self.teacher_decoder.output_size}_decoder.pt"
        )
        wrapped_student_encoder.trace_and_save(
            f"{save_folder_path}/{self.student_encoder.input_size}_{self.student_encoder.latent_size}_encoder.pt"
        )
        wrapped_student_decoder.trace_and_save(
            f"{save_folder_path}/{self.student_decoder.latent_size}_{self.student_decoder.output_size}_decoder.pt"
        )

        wrapped_teacher_encoder.unfreeze()
        wrapped_teacher_decoder.unfreeze()
        wrapped_student_encoder.unfreeze()
        wrapped_student_decoder.unfreeze()


# Example usage with frozen input saving
if __name__ == "__main__":
    # Parameters
    LATENT_DIM = 32
    BATCH_SIZE = 64
    EPOCHS = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load teacher and student observation NPZs (observations only)
    teacher_path = "/workspace/data/transfer/training_data/hand_simple_recording_obs41.npz"
    student_path = "/workspace/data/transfer/training_data/hand_simple_recording_obs41_obs29.npz"

    print(f"Loading teacher observations from {teacher_path}")
    tdata = np.load(teacher_path)
    teacher_obs = tdata["observations"]  # (T, E, obs_dim)

    print(f"Loading student observations from {student_path}")
    sdata = np.load(student_path)
    student_obs = sdata["observations"]  # (T, E, obs_dim)

    # flatten time and env dims into samples
    T, E, teacher_obs_dim = teacher_obs.shape
    Ts, Es, student_obs_dim = student_obs.shape

    assert T == Ts, "Teacher and student must have same number of timesteps"
    assert E == Es, "Teacher and student must have same number of environments"

    teacher_flat = teacher_obs.reshape(T * E, teacher_obs_dim)
    student_flat = student_obs.reshape(Ts * Es, student_obs_dim)

    teacher_data = torch.zeros((T * E, teacher_obs_dim), device=device, dtype=torch.float32)
    student_data = torch.zeros((Ts * Es, student_obs_dim), device=device, dtype=torch.float32)

    with torch.no_grad():
        teacher_data.copy_(torch.from_numpy(teacher_flat))
        student_data.copy_(torch.from_numpy(student_flat))

    # Down sample
    teacher_data = teacher_data[::32, :]
    student_data = student_data[::32, :]

    # Initialize trainer using detected dims
    trainer = EncoderTrainer(
        teacher_input_size=teacher_obs_dim,
        student_input_size=student_obs_dim,
        latent_size=LATENT_DIM,
        device=device,
    )

    # Train using loaded data
    trainer.train(teacher_data, student_data, epochs=EPOCHS, batch_size=BATCH_SIZE)
