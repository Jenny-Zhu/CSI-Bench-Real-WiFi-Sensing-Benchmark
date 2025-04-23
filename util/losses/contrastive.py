import torch
import torch.nn as nn
import torch.nn.functional as F

class NtXentLoss(nn.Module):
    """
    The NT-Xent Loss from SimCLR:
    Normalized Temperature-scaled Cross Entropy loss,
    correlating positive pairs across a batch.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        # Cosine similarity
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / self.temperature

        # Mask out diagonal
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))

        # Positive pairs: (i, i+B), (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ])

        return F.cross_entropy(sim, labels)

class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE Loss using cross-entropy
    over a similarity matrix of (2B x 2B).
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Concat features, compute pairwise dot products
        features = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.matmul(features, features.t()) / self.temperature

        # Zero diagonal to avoid self-sim
        diag_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(diag_mask, -1e9)

        # Labels: each sample should match its counterpart at offset B
        # i.e. 0->B, 1->B+1, ...
        labels = torch.arange(z1.size(0), 2 * z1.size(0), device=sim_matrix.device)
        labels = torch.cat([labels, torch.arange(0, z1.size(0), device=sim_matrix.device)])

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class TripletLoss(nn.Module):
    """
    Standard Triplet Loss: margin-based separation of anchor-positive
    from anchor-negative.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)
        loss = torch.relu(dist_pos - dist_neg + self.margin).mean()
        return loss

class EntLoss(nn.Module):
    """
    Entropy-based loss for self-supervised tasks, combining KL,
    entropic components, and a kernel-density similarity measure.
    """
    def __init__(self, args, lam1, lam2, pqueue=None):
        super().__init__()
        self.lam1 = lam1
        self.lam2 = lam2
        self.pqueue = pqueue
        self.args = args

    def forward(self, feat1, feat2, use_queue=False):
        # Probability distributions
        probs1 = F.softmax(feat1, dim=-1)
        probs2 = F.softmax(feat2, dim=-1)

        # Symmetric KL
        loss = {}
        loss['kl'] = 0.5 * (KL(probs1, probs2, self.args) + KL(probs2, probs1, self.args))

        # Sharpened probs
        sharpened1 = F.softmax(feat1 / self.args.tau, dim=-1)
        sharpened2 = F.softmax(feat2 / self.args.tau, dim=-1)

        # EH, HE
        loss['eh'] = 0.5 * (EH(sharpened1, self.args) + EH(sharpened2, self.args))
        loss['he'] = 0.5 * (HE(sharpened1, self.args) + HE(sharpened2, self.args))

        # Combined final
        loss['final'] = loss['kl'] + ((1 + self.lam1) * loss['eh'] - self.lam2 * loss['he'])

        # Cosine-sim kernel density
        loss['kde'] = cosine_similarity_loss(feat1, feat2)

        # Nuclear norm (optional)
        loss['n-norm'] = -0.5 * (torch.norm(sharpened1, 'nuc') + torch.norm(sharpened2, 'nuc')) * 0.001

        # Merge final + KDE
        loss['final-kde'] = loss['kde'] * 100 + loss['final']  # (plus optional 'n-norm')

        return loss

# Helper functions for EntLoss
def KL(probs1, probs2, args):
    """KL Divergence between two probability distributions (batch-wise)."""
    kl = (probs1 * (probs1 + args.EPS).log() - probs1 * (probs2 + args.EPS).log()).sum(dim=1)
    return kl.mean()

def CE(probs1, probs2, args):
    """Cross-entropy between two probability distributions (batch-wise)."""
    ce = -(probs1 * (probs2 + args.EPS).log()).sum(dim=1)
    return ce.mean()

def HE(probs, args):
    """Entropy of the average distribution."""
    mean = probs.mean(dim=0)
    return -(mean * (mean + args.EPS).log()).sum()

def EH(probs, args):
    """Average entropy across batch."""
    ent = -(probs * (probs + args.EPS).log()).sum(dim=1)
    return ent.mean()

def cosine_similarity_loss(output_net, target_net, eps=1e-7):
    """
    Creates a probability distribution from the pairwise cos-sim of
    feature vectors in output_net vs. themselves, and target_net vs. themselves,
    then compute KL divergence between those distributions.
    """
    # Norm
    def safe_norm(x):
        return x / (x.norm(dim=1, keepdim=True) + eps)

    out = safe_norm(output_net)
    tgt = safe_norm(target_net)

    # Cos-sim in [0,1]
    model_similarity = torch.mm(out, out.t())
    target_similarity = torch.mm(tgt, tgt.t())

    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Prob normalize row-wise
    model_similarity = model_similarity / model_similarity.sum(dim=1, keepdim=True).clamp_min(eps)
    target_similarity = target_similarity / target_similarity.sum(dim=1, keepdim=True).clamp_min(eps)

    # KL divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))
    return loss

def gaussian_noise(std, csi, epsilon, win_len, feature_size):
    """Creates random normal noise and adds scaled noise to 'csi'."""
    noise = torch.normal(0, std, size=(1, win_len, feature_size), device=csi.device)
    return csi + epsilon * noise
