# -*- coding: utf-8 -*-

def main():
    import os
    import numpy as np
    import scipy.io
    import DataPreparator
    import Models
    import torch

    from torch import nn
    from sklearn.model_selection import KFold
    from torch.utils.data import DataLoader

    # 配置--------------------------
    ## 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Will proceed with Nvidia CUDA')
    elif torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            device = torch.device('mps')
            print('Will proceed with Metal Performance Shaders of Apple Silicon')
        else:
            print('MPS failed, use CPU alternatively')
    else:
        device = torch.device('cpu')
        print('Will proceed with CPU')

    ## 配置参数:数据
    SNR_start = -16
    SNR_stop = 2
    SNR_step = 0.5
    SNR_LEVELs = np.arange(SNR_start, SNR_stop + SNR_step, SNR_step)  # 合成训练数据信噪比, dB, 峰值信噪比从-13到2dB
    dtype = np.float32
    fsample = 256  # 数据采样率 Hz
    samp_num = 512
    ##配置参数：训练
    N_FOLD = 10  # 10折交叉验证
    EPOCH_NUM = 200
    Train_BATCH_SIZE = 2 ** 8  # 增大batch size
    test_BATCH_SIZE = 2 ** 7
    # 配置结束------------------------

    # === 创建输出目录 ===
    os.makedirs('output/trained_models', exist_ok=True)
    os.makedirs('output/results', exist_ok=True)

    # 0 初始化输出
    test_arti_eeg_all = np.zeros((0, samp_num))
    test_arti_truth_all = np.zeros((0, samp_num))
    test_arti_estimate_all = np.zeros((0, samp_num))
    test_snr_all = np.zeros((0, 1))
    test_fold_idx_all = np.zeros((0, 1))

    # 初始化评估指标存储
    nmse_all = []
    correlation_all = []

    # 1 准备数据===============================
    print('1. 开始准备眼电噪声估计模型数据...')
    eeg_data = scipy.io.loadmat('eeg.mat')['data']  # 每行一条数据
    eog_data = scipy.io.loadmat('eog.mat')['data']  # 每行一条数据
    arti_eeg_std, arti_std, eeg_std, scale, snr = DataPreparator.data_sythesize(eeg_data, eog_data, SNR_LEVELs, fsample,
                                                                                dtype)

    # 2 十折交叉验证训练========================
    print(f"2. 开始{N_FOLD}折交叉验证训练...")
    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=18931226)
    fold_histories = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(arti_eeg_std)):

        # 2.1 定义数据加载器
        train_dataset = DataPreparator.ArtiEEGDataset(arti_eeg_std[train_idx, :], arti_std[train_idx, :],
                                                      snr[train_idx])
        test_dataset = DataPreparator.ArtiEEGDataset(arti_eeg_std[test_idx, :], arti_std[test_idx, :], snr[train_idx])

        train_loader = DataLoader(train_dataset, batch_size=Train_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_BATCH_SIZE)

        # 2.2 定义模型
        model = Models.SNN_CNN_EEG_Denoise(
            input_dim=512,
            time_steps=4
        )

        # 2.3 定义损失函数
        criterion = Models.TemporalMSELoss()

        # 2.4 定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 2.5 训练模型
        model = Models.train_model(model, train_loader, test_loader, criterion, optimizer, device, EPOCH_NUM, fold_idx)

        # 2.6 评估模型
        model.eval()
        print(f'    · Fold {fold_idx} 正在测试模型')

        # 初始化当前fold的指标
        fold_nmse = []
        fold_correlation = []

        with torch.no_grad():
            for test in test_loader:

                test_arti_eeg, test_arti_truth, test_snr = test
                test_arti_eeg = test_arti_eeg.to(device)

                test_arti_estimate = model(test_arti_eeg)

                # 提取测试结果
                test_arti_eeg = test_arti_eeg.cpu()
                test_arti_eeg_all = np.append(test_arti_eeg_all, test_arti_eeg.numpy(), axis=0)

                test_arti_truth = test_arti_truth.cpu()
                test_arti_truth_all = np.append(test_arti_truth_all, test_arti_truth.numpy(), axis=0)

                test_arti_estimate = test_arti_estimate.cpu()
                test_arti_estimate_all = np.append(test_arti_estimate_all, test_arti_estimate.numpy(), axis=0)

                test_snr = test_snr.cpu()
                test_snr_all = np.append(test_snr_all, test_snr.numpy().reshape((test_snr.shape[0], 1)), axis=0)

                test_fold_idx_all = np.append(test_fold_idx_all, fold_idx * np.ones((test_snr.shape[0], 1)), axis=0)

                # === 新增：计算NMSE和相关性 ===
                batch_clean_eeg = test_arti_eeg.numpy() - test_arti_truth.numpy()
                batch_denoised_eeg = test_arti_eeg.numpy() - test_arti_estimate.numpy()

                # 计算NMSE
                batch_mse = np.mean((batch_clean_eeg - batch_denoised_eeg) ** 2, axis=1)
                batch_signal_power = np.mean(batch_clean_eeg ** 2, axis=1)
                batch_nmse = batch_mse / batch_signal_power
                fold_nmse.extend(batch_nmse)

                # 计算相关性
                for i in range(len(batch_clean_eeg)):
                    corr = np.corrcoef(batch_clean_eeg[i], batch_denoised_eeg[i])[0, 1]
                    if not np.isnan(corr):
                        fold_correlation.append(corr)

        # 计算当前fold的平均指标
        avg_fold_nmse = np.mean(fold_nmse) if fold_nmse else 0
        avg_fold_correlation = np.mean(fold_correlation) if fold_correlation else 0

        nmse_all.append(avg_fold_nmse)
        correlation_all.append(avg_fold_correlation)

        print(f'    · Fold {fold_idx} 评估指标: NMSE = {avg_fold_nmse:.4f}, Correlation = {avg_fold_correlation:.4f}')

        # 2.7 保存当前fold模型
        model_save_path = os.path.join('output', 'trained_models', f'Model_fold_{fold_idx}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'    · Fold {fold_idx} 模型权重已保存')

    # 3 保存测试结果和评估指标
    scipy.io.savemat(os.path.join('output', 'results', 'test_results.mat'), {
        'test_arti_eeg_all': test_arti_eeg_all,
        'test_arti_truth_all': test_arti_truth_all,
        'test_arti_estimate_all': test_arti_estimate_all,
        'test_snr_all': test_snr_all
    })

    # 保存评估指标
    scipy.io.savemat(os.path.join('output', 'results', 'evaluation_metrics.mat'), {
        'nmse_all': np.array(nmse_all),
        'correlation_all': np.array(correlation_all),
        'fold_indices': np.arange(N_FOLD)
    })

    # 打印最终评估结果
    print("\n" + "=" * 60)
    print("最终评估结果汇总")
    print("=" * 60)
    print(f"归一化均方误差 (NMSE): {np.mean(nmse_all):.4f} ± {np.std(nmse_all):.4f}")
    print(f"相关系数 (Correlation): {np.mean(correlation_all):.4f} ± {np.std(correlation_all):.4f}")
    print("=" * 60)

    print('\n各Fold详细结果:')
    for fold in range(N_FOLD):
        print(f'Fold {fold}: NMSE = {nmse_all[fold]:.4f}, Correlation = {correlation_all[fold]:.4f}')

    print('Script execution completed!')


if __name__ == "__main__":  # execute main() if run as script
    main()