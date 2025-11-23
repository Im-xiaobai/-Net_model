#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 15:27:42 2025

@author: mingqi.zhao 这是个基础的代码，效果不是很好，可以修改调整成标准的AutoEncoderDecoder模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn

from tqdm import tqdm

from snntorch import surrogate

batch_size = 2
time_steps = 50  # 模拟50个时间步
num_inputs = 1  # 输入通道数（单通道EEG）
num_hidden = 16  # 隐藏层通道数
num_outputs = 2  # 输出类别数

# 生成模拟数据：随机脉冲输入
# 形状: [批量大小, 时间步长, 通道数, 高度, 宽度]
# 对于1D数据，我们使用高度=1，宽度=序列长度
spike_data = torch.rand(batch_size, time_steps, num_inputs, 1, 100) > 0.9  # 稀疏脉冲
spike_data = spike_data.float()

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN_CNN_EEG_Denoise(nn.Module):
    def __init__(self, input_dim=512, time_steps=8, num_hidden=32):
        super(SNN_CNN_EEG_Denoise, self).__init__()

        self.time_steps = time_steps
        self.input_dim = input_dim

        # 脉冲神经元参数
        beta = 0.5
        spike_grad = surrogate.fast_sigmoid()

        # 编码器部分: 1D CNN + SNN
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=num_hidden,
            kernel_size=3,
            stride=1,
            padding=1  # 修改为padding=1保持维度
        )
        # 移除 init_hidden=True，手动管理膜电位
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv1d(
            in_channels=num_hidden,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # 自适应池化到固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)

        # 瓶颈层: SNN处理
        self.bottleneck_fc = nn.Linear(64 * 128, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # 解码器部分: 重建EEG信号
        self.decoder_fc1 = nn.Linear(256, 512)
        self.decoder_fc2 = nn.Linear(512, input_dim)

    def forward(self, x):
        # 输入x: [batch_size, 512] - 带噪声的EEG
        batch_size = x.shape[0]

        # 为SNN准备时间维度
        x = x.unsqueeze(1)  # [batch, 1, 512]
        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1)  # [T, batch, 1, 512]

        # 手动初始化膜电位
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # 存储每个时间步的编码特征
        encoded_features = []

        # 时间步循环
        for step in range(self.time_steps):
            # 获取当前时间步的输入 [batch, 1, 512]
            cur_input = x_seq[step]

            # 第一层: CNN + SNN
            conv1_out = self.conv1(cur_input)  # [batch, 32, 512]
            spk1, mem1 = self.lif1(conv1_out, mem1)

            # 第二层: CNN + SNN
            conv2_out = self.conv2(spk1)  # [batch, 64, 512]
            spk2, mem2 = self.lif2(conv2_out, mem2)

            # 池化
            pooled = self.adaptive_pool(spk2)  # [batch, 64, 128]
            pooled_flat = pooled.contiguous().view(batch_size, -1)  # [batch, 64*128]

            # 瓶颈层
            bottleneck_out = self.bottleneck_fc(pooled_flat)
            spk3, mem3 = self.lif3(bottleneck_out, mem3)

            encoded_features.append(spk3)

        # 对时间维度取平均
        temporal_avg = torch.stack(encoded_features).mean(dim=0)  # [batch, 256]

        # 解码器: 重建信号
        decoded = self.decoder_fc1(temporal_avg)
        decoded = torch.relu(decoded)
        reconstructed = self.decoder_fc2(decoded)  # [batch, 512]

        return reconstructed

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
    
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations and reshape for multi-head
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out(context)
    
class MultiHeadAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, num_heads=4):
        super(MultiHeadAutoEncoder, self).__init__()

        # Encoder: 2D representation network
        self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        MultiHeadSelfAttention(hidden_dim, num_heads),
        nn.Linear(hidden_dim, latent_dim)
        )
    
        # Decoder: 2D representation network
        self.decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        MultiHeadSelfAttention(hidden_dim, num_heads),
        nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        # Reshape input to (batch_size, seq_len, 1) for attention
        x = x.unsqueeze(-1)
        
        # Encoder
        latent = self.encoder(x)
        
        # Decoder
        reconstructed = self.decoder(latent)
        
        # Reshape back to (batch_size, input_dim)
        return reconstructed.squeeze(-1)

class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        """
        Self-attention mechanism for input features.
    
        Args:
            input_dim (int): Size of input data (e.g., 784 for MNIST).
            attention_dim (int): Dimensionality of attention queries/keys/values.
        """
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # Linear layers for query, key, value
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        
        # Output projection to maintain input_dim
        self.out = nn.Linear(attention_dim, input_dim)
        
        self.scale = attention_dim ** -0.5  # Scaling factor for dot-product attention

    def forward(self, x):
        """
        Forward pass for self-attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Attention-weighted output of shape (batch_size, input_dim).
        """
        batch_size = x.size(0)
        
        # Compute Q, K, V
        q = self.query(x).view(batch_size, 1, self.attention_dim)  # (batch_size, 1, attention_dim)
        k = self.key(x).view(batch_size, 1, self.attention_dim)    # (batch_size, 1, attention_dim)
        v = self.value(x).view(batch_size, 1, self.attention_dim)  # (batch_size, 1, attention_dim)
        
        # Scaled dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (batch_size, 1, 1)
        attn_weights = F.softmax(scores, dim=-1)              # (batch_size, 1, 1)
        attn_output = torch.bmm(attn_weights, v)              # (batch_size, 1, attention_dim)
        
        # Project back to input_dim
        attn_output = self.out(attn_output.squeeze(1))         # (batch_size, input_dim)
        
        return attn_output

#-----
# basic encoder decoder
#-----
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim, attention_dim):
        """
        Autoencoder-Decoder model with self-attention on input layer.
    
        Args:
            input_dim (int): Size of input data (e.g., 784 for MNIST).
            hidden_dim1 (int): Size of first hidden layer.
            hidden_dim2 (int): Size of second hidden layer.
            latent_dim (int): Size of latent space.
            attention_dim (int): Dimensionality of attention mechanism.
        """
        super(Autoencoder, self).__init__()
        
        # Self-attention layer applied to input
        self.attention = SelfAttention(input_dim, attention_dim)
        
        # Encoder: input -> hidden1 -> hidden2 -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim)
        )
        
        # Decoder: latent -> hidden2 -> hidden1 -> output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            #nn.Sigmoid()  # Output in [0, 1] for normalized inputs
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder with attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Reconstructed output of shape (batch_size, input_dim).
        """
        # Apply attention to input
        attention = self.attention(x)
        
        # Pass through encoder and decoder
        latent = self.encoder(attention)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        """
        Encode input to latent space with attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim).
        """
        x = self.attention(x)
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent representation to output.
        
        Args:
            z (torch.Tensor): Latent tensor of shape (batch_size, latent_dim).
        
        Returns:
            torch.Tensor: Reconstructed output of shape (batch_size, input_dim).
        """
        return self.decoder(z)

# --------------------
# 改进的模型架构
# --------------------
class EnhancedNoiseExtractor(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[512, 384, 256, 128], num_heads=8, dropout_rate=0.3):
        super().__init__()
        # 编码器 - 更深的网络结构
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.latent_dim = hidden_dims[-1]
        self.attention = EnhancedAttentionBlock(self.latent_dim, num_heads, dropout_rate)
        
        # 修改TCN分支以确保输出维度为latent_dim
        self.tcn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # 将序列长度池化为1
            nn.Flatten(),  # 展平为 [batch_size, 32]
            nn.Linear(32, self.latent_dim)  # 投影到与latent_dim相同的维度
        )
        
        # 改进的噪声提取器 - 使用UNet风格的跳跃连接
        noise_extractor_layers = []
        skip_connections = []
        self.skip_layers = nn.ModuleList()
        
        # 将reversed(hidden_dims)转换为列表以便索引访问
        reversed_dims = list(reversed(hidden_dims))
        
        # 下采样路径中保存中间层输出
        for i, dim in enumerate(hidden_dims):
            if i < len(hidden_dims) - 1:
                skip_connections.append(dim)
                self.skip_layers.append(nn.Linear(dim, reversed_dims[i+1]))
        
        # 上采样路径与跳跃连接
        decoder_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]
        for i, dim in enumerate(decoder_dims):
            if i < len(skip_connections):
                # 与跳跃连接融合
                noise_extractor_layers.extend([
                    nn.Linear(prev_dim + skip_connections[i], dim),  # +skip_connection维度
                    nn.BatchNorm1d(dim) if dim != input_dim else nn.Identity(),
                    nn.LeakyReLU(0.2) if dim != input_dim else nn.Identity(),
                    nn.Dropout(dropout_rate) if dim != input_dim else nn.Identity()
                ])
            else:
                noise_extractor_layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim) if dim != input_dim else nn.Identity(),
                    nn.LeakyReLU(0.2) if dim != input_dim else nn.Identity(),
                    nn.Dropout(dropout_rate) if dim != input_dim else nn.Identity()
                ])
            prev_dim = dim
        
        # 最终输出层使用Tanh激活以约束范围
        noise_extractor_layers.append(nn.Tanh())
        
        self.noise_decoder_layers = nn.ModuleList(noise_extractor_layers)
        

    def forward(self, x):
        # 编码器路径，保存中间层结果用于跳跃连接
        batch_size = x.shape[0]
        intermediate_outputs = []
        
        # 使用编码器获取特征
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU) and i < len(self.encoder) - 2:
                intermediate_outputs.append(x)
        
        encoded = x
        
        # 注意力处理
        if len(encoded.shape) == 2:
            encoded_seq = encoded.unsqueeze(1)
            attended = self.attention(encoded_seq)
            encoded_final = attended.squeeze(1)
        else:
            attended = self.attention(encoded)
            encoded_final = attended.mean(dim=1)
        
        # 修改TCN分支处理
        tcn_input = x.unsqueeze(1)  # [B, 1, F]
        tcn_output = self.tcn(tcn_input)  # 现在是 [B, latent_dim]
        
        # 现在可以安全地融合特征
        fused_features = encoded_final + tcn_output
        
        # 与编码器中间层输出进行跳跃连接
        x = fused_features
        skip_idx = len(intermediate_outputs) - 1
        
        # 解码器路径，融合跳跃连接
        for i in range(0, len(self.noise_decoder_layers), 4):
            if skip_idx >= 0 and i < len(self.noise_decoder_layers) - 4:
                skip = self.skip_layers[skip_idx](intermediate_outputs[skip_idx])
                x = torch.cat([x, skip], dim=1)
                skip_idx -= 1
            
            # 应用解码器层
            for j in range(min(4, len(self.noise_decoder_layers) - i)):
                x = self.noise_decoder_layers[i + j](x)
        
        # 应用最后的Tanh激活
        if isinstance(self.noise_decoder_layers[-1], nn.Tanh):
            extracted_noise = self.noise_decoder_layers[-1](x)
        else:
            extracted_noise = x
            
        return extracted_noise
    
# --------------------
# 改进的注意力模块
# --------------------
class EnhancedAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        # 第一个残差连接和norm
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 第二个残差连接和norm
        ff_output = self.ffn(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x
    

#
# Combined spectral and temporal loss
#
class TemporalMSELoss(nn.Module):
    """
    A PyTorch module for computing a combined Temporal MSE and Spectral MSE loss.
    
    Args:
        alpha (float): Weight for temporal MSE.
        beta (float): Weight for spectral MSE.
    """
    def __init__(self):
        super(TemporalMSELoss, self).__init__()
        
    
    def forward(self, pred, target):
        """
        Compute the combined loss.
        
        Args:
            pred (torch.Tensor): Predicted signal (batch_size, sequence_length).
            target (torch.Tensor): Target signal (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: Combined loss value.
        """
        # Ensure inputs are float tensors
        pred = pred.float()
        target = target.float()
        
        # 1. Temporal MSE
        return torch.mean((pred - target) ** 2)
        
class CombinedTemporalSpectralLoss(nn.Module):
    """
    A PyTorch module for computing a combined Temporal MSE and Spectral MSE loss.
    
    Args:
        alpha (float): Weight for temporal MSE.
        beta (float): Weight for spectral MSE.
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super(CombinedTemporalSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, target):
        """
        Compute the combined loss.
        
        Args:
            pred (torch.Tensor): Predicted signal (batch_size, sequence_length).
            target (torch.Tensor): Target signal (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: Combined loss value.
        """
        # Ensure inputs are float tensors
        pred = pred.float()
        target = target.float()
        
        # 1. Temporal MSE
        temporal_mse = torch.mean((pred - target) ** 2)
        
        # 2. Spectral MSE
        # Compute FFT and magnitude
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Compute Spectral MSE
        spectral_mse = torch.mean((pred_mag - target_mag) ** 2)
        
        # 3. Combine losses
        combined_loss = self.alpha * temporal_mse + self.beta * spectral_mse
        
        return combined_loss

# --------------------
# 训练函数
# --------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epoch_num = 10, fold_idx = 0):
    
    # internal configs
    patience = 5
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 定义学习率调度器
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', step_size=5, gamma=0.1)
    
    # 最佳模型初始化
    best_model_state = None
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch_idx in range(1,epoch_num):
        # 开启训练模式
        model.train()
        train_loss = 0.0
        
        # 训练阶段
        batch_progress = tqdm(train_loader)
        for batch in batch_progress:
            # 将数据移动到设备
            inputs,labels,_ = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # 打印训练进度
            batch_progress.set_description(f'    · Fold {fold_idx} 训练第 {epoch_idx}/{epoch_num} Epoch [train loss: {train_loss:.0f}]')
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss= 0.0
        
        with torch.no_grad():
            for val in val_loader:
                inputs,labels,_ = val
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # 调整学习率
        #lr_scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            print(f'    · Fold {fold_idx} 第 {epoch_idx}/{epoch_num} Epoch，保存最佳模型，best_val_loss={best_val_loss:.4f}')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'    · Fold {fold_idx} 第 {epoch_idx}/{epoch_num} Epoch未观测到loss改进,实施早停')
            break
    model.load_state_dict(best_model_state)
    return model
   