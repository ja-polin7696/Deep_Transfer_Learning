"""Adversarial adaptation to train target encoder."""

import os
import torch
import torch.optim as optim
from torch import nn

import params
from utils import make_variable


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain using ADDA adversarial training."""

    # Ensure models are in training mode
    tgt_encoder.train()
    critic.train()

    # Loss + optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(
        tgt_encoder.parameters(),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2)
    )
    optimizer_critic = optim.Adam(
        critic.parameters(),
        lr=params.d_learning_rate,
        betas=(params.beta1, params.beta2)
    )

    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    print("\n=== Training encoder for target domain ===\n")

    # ===========================
    # ADDA TRAINING LOOP
    # ===========================
    for epoch in range(params.num_epochs):

        for step, ((images_src, _), (images_tgt, _)) in \
                enumerate(zip(src_data_loader, tgt_data_loader)):

            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # ------------------------------
            # (1) Train Discriminator (Critic)
            # ------------------------------
            optimizer_critic.zero_grad()

            feat_src = src_encoder(images_src).detach()
            feat_tgt = tgt_encoder(images_tgt).detach()

            feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

            pred_concat = critic(feat_concat)

            # Correct domain labels
            label_src = torch.zeros(feat_src.size(0)).long()
            label_tgt = torch.ones(feat_tgt.size(0)).long()
            label_concat = torch.cat((label_src, label_tgt), dim=0)
            label_concat = make_variable(label_concat)

            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()
            optimizer_critic.step()

            # Critic accuracy
            pred_cls = torch.argmax(pred_concat, dim=1)
            acc = (pred_cls == label_concat).float().mean()

            # ------------------------------
            # (2) Train Target Encoder (Generator)
            # ------------------------------
            optimizer_tgt.zero_grad()

            feat_tgt = tgt_encoder(images_tgt)
            pred_tgt = critic(feat_tgt)

            # Fool the discriminator → pretend target = source (label = 0)
            fool_labels = torch.zeros(feat_tgt.size(0)).long()
            fool_labels = make_variable(fool_labels)

            loss_tgt = criterion(pred_tgt, fool_labels)
            loss_tgt.backward()
            optimizer_tgt.step()

            # ------------------------------
            # (3) Logging
            # ------------------------------
            if (step + 1) % params.log_step == 0:
                print(
                    "Epoch [{}/{}] Step [{}/{}]: "
                    "d_loss={:.5f} g_loss={:.5f} acc={:.5f}".format(
                        epoch + 1,
                        params.num_epochs,
                        step + 1,
                        len_data_loader,
                        loss_critic.item(),
                        loss_tgt.item(),
                        acc.item()
                    )
                )

        # Save model each epoch
        if (epoch + 1) % params.save_step == 0:
            torch.save(critic.state_dict(),
                       os.path.join(params.model_root,
                                    f"ADDA-critic-{epoch+1}.pt"))
            torch.save(tgt_encoder.state_dict(),
                       os.path.join(params.model_root,
                                    f"ADDA-target-encoder-{epoch+1}.pt"))

    # Save final models
    torch.save(critic.state_dict(),
               os.path.join(params.model_root, "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(),
               os.path.join(params.model_root, "ADDA-target-encoder-final.pt"))

    print("\n=== Finished Adversarial Adaptation ===\n")
    return tgt_encoder
