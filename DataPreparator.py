#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:18:20 2025

@author: mingqi.zhao
"""
#import sklearn.model_selection as ms
import random
import numpy as np
#import scipy.io as sio
import scipy.signal as signal
import scipy.stats as stats

#import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm


def data_sythesize(signal, arti, snr_values, fs, dtype=np.double):
    """ 合成信号并控制信噪比
        输入EEG为M1*N矩阵，包含M1条N1个采样点的数据
        输入arti为M2*N矩阵，包含M2条N1个采样点的数据
        输入snr_values为1*L的向量，表示有L种信噪比dB
        输入fs为信号采样率
        
        对于M2条arti数据，随机从M1条EEG数据中选取M2条EEG数据，与对应的M2个arti以snr_db中的信噪比数值混合
        最终返回arti_eeg_all, arti_all, eeg_all 均为M2*N*L矩阵，分别为对应的带噪声脑电、噪声和纯EEG
        十折交叉验证抽样时，从M2维度中抽，抽取后将L维度展开
        
    
        数据生成过程
        1. 所有数据标准化
        2. 估算所有数据功率
        3. 眼电噪声有3400条，乘-1反相，增加进去后扩充到6800条
        4. 对于每种信噪比，随机抽取M2条脑电，根据信噪比和当前信号噪声功率估算噪声的幅度缩放系数
        5. 根据系数生成每种信噪比的合成信号-噪声数据对
        6. 再次标准化，用于训练测试
        
    """
    arti = np.append(arti, -1*arti, axis=0)
    
    signal_num = np.size(signal, axis=0)
    arti_num = np.size(arti, axis=0)
    samp_num = np.size(arti, axis=1)
    snr_num = np.size(snr_values)
    all_num = arti_num*snr_num
    
    #输入信号标准化
    signal = stats.zscore(signal, axis=1)
    arti = stats.zscore(arti, axis=1)
    
    signal_power = get_mean_power(signal)
    arti_power = get_peak_power(arti, fs)
    
    #arti_eeg_all = np.full((arti_num, samp_num, snr_num), np.nan)
    #arti_all = np.full((arti_num, samp_num, snr_num), np.nan)
    #eeg_all = np.full((arti_num, samp_num, snr_num), np.nan)
    arti_eeg_all = np.zeros((0, samp_num))
    arti_all = np.zeros((0, samp_num))
    eeg_all = np.zeros((0, samp_num))
    scale_all = np.zeros((0, samp_num))
    snr_all = np.zeros((0, 1))
    
    progress_bar = tqdm(range(0,snr_num))
    for iter_snr in progress_bar:
        progress_bar.set_description('    · 正在合成数据')
        # 从M1条脑电中随机抽取M2条脑电
        if arti_num > signal_num:
            samp_n = arti_num//signal_num # 计算抽样次数
            samp_res = arti_num%signal_num # 计算抽样后还不足的余数
            
            idx = random.sample(range(0,signal_num),signal_num)
            for iter_samp in range(0, samp_n-1):
                tmp_idx = random.sample(range(0,signal_num),signal_num)
                idx = idx.extend(tmp_idx)
                
            tmp_idx = random.sample(range(0,signal_num),samp_res)
            idx.extend(tmp_idx)
            
        else:
            idx = random.sample(range(0,signal_num),arti_num)
        
            
        signal_sub = signal[idx,:]
        signal_power_sub = signal_power[idx]
        
        snr = snr_values[iter_snr]
        snr_linear = 10**(snr/10)
        tmp_scale_factors = np.sqrt(signal_power_sub/(arti_power*snr_linear))
        
        
        #合成信号,并标准化
        tmp_scale_factors = tmp_scale_factors.repeat(samp_num)
        tmp_scale_factors = tmp_scale_factors.reshape(arti_num,samp_num)
        tmp_arti = tmp_scale_factors*arti #缩放噪声
        tmp_arti_eeg = signal_sub + tmp_arti #合成
        tmp_eeg = signal_sub
        
        tmp_arti_eeg = stats.zscore(tmp_arti_eeg, axis=1) #标准化带噪声脑电
        tmp_arti = stats.zscore(tmp_arti,axis=1) #标准化缩放后的噪声
        tmp_snr = np.repeat(snr, arti_num)
        
        #存储配对的数据
        #arti_eeg_all[:,:,iter_snr] = tmp_arti_eeg
        arti_eeg_all = np.append(arti_eeg_all, tmp_arti_eeg, axis=0)
        #arti_all[:,:,iter_snr] = tmp_arti
        arti_all = np.append(arti_all, tmp_arti, axis=0)
        #eeg_all[:,:,iter_snr] = tmp_eeg
        eeg_all = np.append(eeg_all, tmp_eeg, axis=0)
        scale_all = np.append(scale_all, tmp_scale_factors, axis=0)
        snr_all = np.append(snr_all, tmp_snr)
        
        
    #保留配对的情况下随机打乱次序，防止不同信噪比的同一噪声样本跨训练集和测试集，造成泄漏 【效果不好则关闭】
    idx = random.sample(range(0,all_num),all_num)
    arti_eeg_all = arti_eeg_all[idx, :]
    arti_all = arti_all[idx, :]
    eeg_all = eeg_all[idx, :]
    scale_all = scale_all[idx, :]
    snr_all = snr_all[idx]
    
    #设置数据精度
    if arti_eeg_all.dtype != dtype:
        arti_eeg_all = arti_eeg_all.astype(np.float32)
    
    if arti_all.dtype != dtype:
        arti_all = arti_all.astype(np.float32)
        
    if eeg_all.dtype != dtype:
        eeg_all = eeg_all.astype(np.float32)
        
    if scale_all.dtype != dtype:
        scale_all = scale_all.astype(np.float32)
        
    if snr_all.dtype != dtype:
        snr_all = snr_all.astype(np.float32)
    
    all_num = np.size(arti_eeg_all, axis=0)
    samp_num = np.size(arti_eeg_all, axis=1)
    print(f"    · 合成结束，共获得{all_num}条数据，每条样本采样点数为{samp_num}，数据类型为{arti_eeg_all.dtype}")
    return arti_eeg_all, arti_all, eeg_all, scale_all, snr_all

def get_mean_power(data):
    mean_power = np.mean(data**2, axis=1)
    return mean_power

def get_peak_power(data,fs):
    cutoff = 0.25
    nyquist = 0.5*fs
    norm_cutoff = cutoff/nyquist
    n = 4
    b, a = signal.butter(n, norm_cutoff, btype='lowpass') #创建0.5Hz低通滤波器提取直流分量
    
    data_dc = signal.filtfilt(b, a, data, axis=0)
    data_dc = np.mean(data_dc, axis=1)
    
    peak_p = abs(np.max(data, axis=1) - data_dc)
    peak_n = abs(data_dc - np.min(data, axis=1))
    
    return np.max(np.array([peak_p**2, peak_n**2]), axis=0)
    
class ArtiEEGDataset(Dataset):
    def __init__(self, arti_eeg, arti, snr):
        '''参数:
        arti_eeg: 带噪声的脑电信号
        arti: 对应的噪声信号
        '''
        self.arti_eeg = arti_eeg
        self.arti = arti
        self.snr = snr
        
    def __len__(self):
        return len(self.arti_eeg)
    
    def __getitem__(self, index):
        return self.arti_eeg[index,:], self.arti[index,:], self.snr[index]
        
    
    