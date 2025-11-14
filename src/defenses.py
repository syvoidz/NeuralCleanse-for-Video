import torch
from torch import optim
import math
from .utils import evaluate_clean_acc, evaluate_asr, inject_trigger_video


def unlearning_defense(model, unlearning_loader, test_loader, trigger_data, args):
    print("\n--- Starting FINAL OPTIMIZED Active Unlearning Defense ---")
    device = torch.device(args.device)
    model.to(device)

    params_to_optimize = []
    print("Unfreezing parameters in: cnn.layer4, lstm, fc")
    for name, param in model.named_parameters():
        if 'cnn.layer4' in name or 'lstm' in name or 'fc' in name:
            param.requires_grad = True
            params_to_optimize.append(param)
        else:
            param.requires_grad = False
    optimizer = optim.Adam(params_to_optimize, lr=args.lr_mitigate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(args.nb_epochs_mitigate):
        model.train()
        model.cnn.conv1.eval();
        model.cnn.bn1.eval();
        model.cnn.layer1.eval()
        model.cnn.layer2.eval();
        model.cnn.layer3.eval()

        for images, labels in unlearning_loader:
            images, labels = images.to(device), labels.to(device)

            batch_size = images.shape[0]
            # 向上取整，确保至少有1个干净样本
            num_clean = math.ceil(batch_size * 0.25)  # 25% for ACC
            num_poison = batch_size - num_clean  # 75% for ASR

            # --- 前一部分(25%)：用于“巩固”ACC ---
            clean_images = images[:num_clean]
            clean_labels = labels[:num_clean]

            # --- 后一部分(75%)：用于“解毒”ASR ---
            images_to_poison = images[num_clean:]
            original_labels_for_poisoned = labels[num_clean:]

            # 如果后一部分为空，则跳过
            if images_to_poison.shape[0] == 0:
                continue

            antidote_images = inject_trigger_video(images_to_poison, trigger_data)

            combined_images = torch.cat([clean_images, antidote_images], dim=0)
            combined_labels = torch.cat([clean_labels, original_labels_for_poisoned], dim=0)

            optimizer.zero_grad()
            outputs = model(combined_images)
            loss = criterion(outputs, combined_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 2 == 0 or epoch == args.nb_epochs_mitigate - 1:
            acc = evaluate_clean_acc(model, test_loader, device)
            asr = evaluate_asr(model, test_loader, trigger_data, args.target_label, device)
            print(f"Epoch {epoch + 1}/{args.nb_epochs_mitigate} | Clean ACC: {acc:.2f}% | ASR: {asr:.2f}%")

    print("\n--- Mitigation finished! ---")
    return model
