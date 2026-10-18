"""
EGNN training script (LayerNorm version)

Features:
- LayerNorm (replaces BatchNorm) for stable validation behaviour
- clamped sin/cos radial embeddings
- residuals, coordinate scaling
- OneCycleLR warmup + cosine anneal (step per batch)
- gradient clipping, checkpointing, 90/10 split
- standardize energies (cm^-1) and print MAE in cm^-1 each epoch
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# This script runs bare-metal on the GPU node (no apptainer container, see
# train_egnn.sh), so system_config is reached via its real host path rather
# than the /home/pipeline_config bind every containerized step uses.
sys.path.insert(0, '/gscratch/mccoy/pot_training/pipeline_config')
from system_config import SYSTEM_TAG, ATOMS

# ---------------------------
# Utilities
# ---------------------------
def ensure_float32(t):
    if not torch.is_floating_point(t):
        return t.float()
    return t.to(torch.float32)

# stable sin/cos embedding that accepts (B,N,N) or (B,N,N,1)
def build_sin_cos_embedding(d, n_freqs=3, include_orig=True, clamp_val=1e2):
    """
    d: torch tensor, either shape (B,N,N) or (B,N,N,1)
    Returns: (B,N,N, D_emb) where D_emb = (1 if include_orig else 0) + 2*n_freqs
    """
    if d.dim() == 3:
        d = d.unsqueeze(-1)  # -> (B,N,N,1)
    # frequencies 1..n_freqs
    freq = torch.arange(1, n_freqs + 1, device=d.device, dtype=d.dtype).view(1,1,1,n_freqs)
    angles = d * freq  # (B,N,N,n_freqs)
    angles = torch.clamp(angles, -clamp_val, clamp_val)
    sin_emb = torch.sin(angles)
    cos_emb = torch.cos(angles)
    emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (B,N,N,2*n_freqs)
    if include_orig:
        emb = torch.cat([d, emb], dim=-1)  # (B,N,N,1+2*n_freqs)
    return emb

# ---------------------------
# EGNN Layer with LayerNorm + Residuals
# ---------------------------
class EGNNLayerLN(nn.Module):
    def __init__(self, hidden_dim=64, n_freqs=3, coord_rescale=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_freqs = n_freqs
        self.coord_rescale = coord_rescale
        self.d_emb_dim = 1 + 2 * n_freqs

        # message MLP: input = xi || xj || d_emb
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.d_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # coordinate MLP -> scalar per edge
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

        # feature update MLP (residual)
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # LayerNorm applied after residual add
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, coords):
        """
        x: (B, N, H)
        coords: (B, N, 3)
        """
        B, N, _ = coords.shape

        # pairwise differences r_ij = r_j - r_i  (B,N,N,3)
        rij = coords.unsqueeze(2) - coords.unsqueeze(1)
        dij = torch.norm(rij, dim=-1, keepdim=True)  # (B,N,N,1)

        # radial embedding (clamped inside)
        d_emb = build_sin_cos_embedding(dij, n_freqs=self.n_freqs, include_orig=True)  # (B,N,N,D)

        # prepare node pair features
        xi = x.unsqueeze(2).expand(-1, -1, N, -1)  # (B,N,N,H)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B,N,N,H)
        edge_in = torch.cat([xi, xj, d_emb], dim=-1)  # (B,N,N,2H+D)

        # messages
        m = self.message_mlp(edge_in)  # (B,N,N,H)

        # coordinate update (scaled residual)
        coord_coef = self.coord_mlp(m).squeeze(-1)  # (B,N,N)
        delta_coords = (coord_coef.unsqueeze(-1) * rij).sum(dim=2)  # (B,N,3)
        coords = coords + self.coord_rescale * delta_coords

        # aggregate messages
        m_sum = m.sum(dim=2)  # (B,N,H)

        # feature update with residual + LayerNorm
        dx = self.feature_mlp(m_sum)  # (B,N,H)
        x = x + dx
        x = self.ln(x)

        return x, coords

# ---------------------------
# Full EGNN model (LayerNorm)
# ---------------------------
class EGNNModelLN(nn.Module):
    def __init__(self, n_atoms, atom_types_list, hidden_dim=64, num_layers=3, n_freqs=3, coord_rescale=0.01):
        super().__init__()
        self.n_atoms = n_atoms

        # store atom types as buffer (1D vector length n_atoms)
        atom_types = torch.as_tensor(atom_types_list, dtype=torch.float32).view(-1)
        self.register_buffer("atom_types", atom_types)

        # atom input embedding: accepts shape (B,N,1)
        self.embedding = nn.Linear(1, hidden_dim)
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # stack of layers
        self.layers = nn.ModuleList([
            EGNNLayerLN(hidden_dim=hidden_dim, n_freqs=n_freqs, coord_rescale=coord_rescale)
            for _ in range(num_layers)
        ])

        # output per-atom -> sum
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, coords, z=None):
        """
        coords: (B, n_atoms, 3)
        z: optional per-sample atom types (B, n_atoms, 1). If None, use internal atom_types buffer.
        returns: (B,1)
        """
        B = coords.shape[0]
        if z is None:
            z = self.atom_types.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)  # (B, N, 1)
        x = self.embedding(z)  # (B,N,H)
        x = self.input_mlp(x)
        for layer in self.layers:
            x, coords = layer(x, coords)
        per_atom = self.out_mlp(x).squeeze(-1)  # (B,N)
        energy = per_atom.sum(dim=1, keepdim=True)  # (B,1)
        return energy

# ---------------------------
# Training function with OneCycleLR + LayerNorm
# ---------------------------
def train_egnn_with_ln_onecycle(
    system_name,
    coords,                     # numpy or torch array (N_samples, n_atoms, 3)
    atom_types,                 # list len n_atoms (e.g., [8,1,1,...])
    energies_cm1,               # numpy or torch array (N_samples,)
    hidden_dim=64,
    num_layers=3,
    n_freqs=3,
    coord_rescale=0.01,
    batch_size=128,
    max_epochs=200,
    max_lr=1e-3,
    pct_start=0.05,
    weight_decay=1e-6,
    grad_clip=1.0,
    save_dir="./checkpoints_ln",
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    print("Device:", device)

    # ensure numpy arrays
    coords = np.asarray(coords, dtype=np.float32)
    energies_cm1 = np.asarray(energies_cm1, dtype=np.float32)
    n_samples, n_atoms, _ = coords.shape

    # shuffle and split (90/10)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    coords = coords[indices]
    energies_cm1 = energies_cm1[indices]

    n_train = int(0.9 * n_samples)
    n_val = n_samples - n_train

    coords_train = coords[:n_train]
    coords_val = coords[n_train:]
    energies_train = energies_cm1[:n_train]
    energies_val = energies_cm1[n_train:]

    # compute mean/std on training energies
    mu = float(np.mean(energies_train))
    sigma = float(np.std(energies_train))
    if sigma == 0 or math.isnan(sigma):
        raise ValueError("Computed sigma is zero or NaN. Check energies input.")

    print(f"Training samples: {n_train}, Validation samples: {n_val}")
    print(f"Energy mu={mu:.3f} cm^-1, sigma={sigma:.3f} cm^-1")

    # normalized targets (full dataset order)
    y_all_norm = ((energies_cm1 - mu) / sigma).astype(np.float32).reshape(-1, 1)

    # tensors
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    # expand atom_types to (N_samples, n_atoms, 1)
    atom_types_arr = np.asarray(atom_types, dtype=np.float32).reshape(1, n_atoms, 1)
    atom_types_batch = np.repeat(atom_types_arr, n_samples, axis=0)
    atom_types_tensor = torch.tensor(atom_types_batch, dtype=torch.float32)
    targets_norm_tensor = torch.tensor(y_all_norm, dtype=torch.float32)

    # dataset and loaders
    dataset = TensorDataset(coords_tensor, atom_types_tensor, targets_norm_tensor)
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # model + optimizer
    model = EGNNModelLN(n_atoms=n_atoms, atom_types_list=atom_types, hidden_dim=hidden_dim,
                        num_layers=num_layers, n_freqs=n_freqs, coord_rescale=coord_rescale).to(device)

    # initialize optimizer with smaller base lr; OneCycle will schedule up to max_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr / 10.0, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=pct_start, anneal_strategy='cos', div_factor=10.0, final_div_factor=1e4)


    loss_fn = nn.L1Loss()  # MAE in normalized space

    best_val_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        # TRAIN
        model.train()
        running_loss = 0.0
        running_phys_mae_sum = 0.0
        running_count = 0

        for batch_idx, batch in enumerate(train_loader):
            c_batch, z_batch, y_norm_batch = batch
            c_batch = c_batch.to(device)
            z_batch = z_batch.to(device)
            y_norm_batch = y_norm_batch.to(device)

            optimizer.zero_grad()
            pred_norm = model(c_batch, z_batch)           # (B,1)
            loss = loss_fn(pred_norm, y_norm_batch)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            scheduler.step()

            # bookkeeping
            bsz = c_batch.shape[0]
            running_loss += loss.item() * bsz
            with torch.no_grad():
                pred_cm1 = (pred_norm * sigma + mu).squeeze(-1)*219474.63136320
                true_cm1 = (y_norm_batch * sigma + mu).squeeze(-1)*219474.63136320
                batch_mae = torch.mean(torch.abs(pred_cm1 - true_cm1)).item()
                running_phys_mae_sum += batch_mae * bsz
            running_count += bsz

        avg_train_loss = running_loss / running_count
        avg_train_mae = running_phys_mae_sum / running_count

        # VALIDATION
        model.eval()
        val_loss_sum = 0.0
        val_phys_mae_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                c_batch, z_batch, y_norm_batch = batch
                c_batch = c_batch.to(device)
                z_batch = z_batch.to(device)
                y_norm_batch = y_norm_batch.to(device)

                pred_norm = model(c_batch, z_batch)
                loss = loss_fn(pred_norm, y_norm_batch)

                bsz = c_batch.shape[0]
                val_loss_sum += loss.item() * bsz

                pred_cm1 = (pred_norm * sigma + mu).squeeze(-1)*219474.63136320
                true_cm1 = (y_norm_batch * sigma + mu).squeeze(-1)*219474.63136320
                val_phys_mae_sum += torch.mean(torch.abs(pred_cm1 - true_cm1)).item() * bsz
                val_count += bsz

        avg_val_loss = val_loss_sum / val_count
        avg_val_mae = val_phys_mae_sum / val_count

        print(f"Epoch {epoch:4d}/{max_epochs} | TrainLoss(norm)={avg_train_loss:.6e} | "
              f"TrainMAE(cm^-1)={avg_train_mae:.3f} | ValLoss(norm)={avg_val_loss:.6e} | ValMAE(cm^-1)={avg_val_mae:.3f}")

        # checkpoint best by normalized val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "mu": mu,
                "sigma": sigma
            }, os.path.join(save_dir, f"{system_name}_egnn_1-5mill_new_mobml_ccsdt_tz_64_3_3_1e6decay_ln_best.pth"))
            print("  -> Saved new best model.")

    # save latest
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "mu": mu,
        "sigma": sigma
    }, os.path.join(save_dir, f"{system_name}_egnn_1-5mill_new_mobml_ccsdt_tz_64_3_3_1e6decay_ln_latest.pth"))

    return model, mu, sigma

# ---------------------------
# Example driver (edit to run)
# ---------------------------
if __name__ == "__main__":
    system_name = SYSTEM_TAG
    n_atoms = len(ATOMS)

    # Real host paths -- this script runs bare-metal, not through the
    # apptainer container the rest of the pipeline uses, so pca_and_fps's
    # output isn't reachable via /home/pca_and_fps here.
    pca_and_fps_dir = '/gscratch/mccoy/pot_training/pca_and_fps'

    cds = np.load(f'{pca_and_fps_dir}/output/{system_name}/{system_name}_egnn_training_cds.npy')
    coords = torch.tensor(cds,dtype=torch.float64)

    en = np.load(f'{pca_and_fps_dir}/output/{system_name}/{system_name}_egnn_training_energies.npy')
    energies = torch.tensor(en,dtype=torch.float64)

    ATOM_NUMBERS = {'H': 1.0, 'C': 6.0, 'N': 7.0, 'O': 8.0} #UPDATE HERE if a new element appears
    atom_types = torch.tensor([[ATOM_NUMBERS[a]] for a in ATOMS])

    model, mu, sigma = train_egnn_with_ln_onecycle(
        system_name, coords, atom_types, energies,
        hidden_dim=64, num_layers=3, n_freqs=3,
        batch_size=128, max_epochs=200, max_lr=1e-3,
        coord_rescale=0.01, grad_clip=1.0, save_dir="./checkpoints_ln_onecycle"
    )

    print("Done. mu, sigma:", mu, sigma)


