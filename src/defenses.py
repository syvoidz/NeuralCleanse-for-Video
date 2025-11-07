import torch
from torch import optim
from utils import evaluate_clean_acc, evaluate_asr, inject_trigger_video


def unlearning_defense(model, unlearning_loader, test_loader, trigger_data, args):
    """
    执行主动对抗（Unlearning）防御，对模型的指定部分进行微调。
    """
    print("\n--- Starting Targeted Active Unlearning Defense ---")
    device = torch.device(args.device)
    model.to(device)

    # 1. 精准选择需要优化的参数
    params_to_optimize = []
    print("Unfreezing parameters in: cnn.layer4, lstm, fc")
    for name, param in model.named_parameters():
        if 'cnn.layer4' in name or 'lstm' in name or 'fc' in name:
            param.requires_grad = True
            params_to_optimize.append(param)
        else:
            param.requires_grad = False

    # 2. 设置优化器和调度器
    optimizer = optim.Adam(params_to_optimize, lr=args.lr_mitigate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # 3. 准备触发器
    # 我们假设 trigger_data 是一个字典 {'mask': tensor, 'pattern': tensor}
    reconstructed_trigger = trigger_data['mask'] * trigger_data['pattern']
    reconstructed_trigger_gpu = reconstructed_trigger.to(device)
    trigger_size = (reconstructed_trigger.shape[1], reconstructed_trigger.shape[2])

    # 4. 训练循环
    for epoch in range(args.nb_epochs_mitigate):
        model.train()
        # 确保冻结的部分处于评估模式
        model.cnn.conv1.eval()
        model.cnn.bn1.eval()
        model.cnn.layer1.eval()
        model.cnn.layer2.eval()
        model.cnn.layer3.eval()

        running_loss = 0.0
        for images, labels in unlearning_loader:
            images, labels = images.to(device), labels.to(device)

            # 制造抗体数据
            antidote_images = inject_trigger_video(images, reconstructed_trigger_gpu, trigger_size)

            optimizer.zero_grad()
            outputs = model(antidote_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # 5. 定期评估
        if (epoch + 1) % 2 == 0 or epoch == args.nb_epochs_mitigate - 1:
            acc = evaluate_clean_acc(model, test_loader, device)
            # 使用组合后的触发器进行评估
            asr = evaluate_asr(model, test_loader, reconstructed_trigger, args.target_label, device)
            print(f"Epoch {epoch + 1}/{args.nb_epochs_mitigate} | Clean ACC: {acc:.2f}% | ASR: {asr:.2f}%")

    print("\n--- Mitigation finished! ---")
    return model
