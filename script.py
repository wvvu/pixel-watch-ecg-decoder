# -*- coding: utf-8 -*-
"""
ecg_decode.py — Pixel Watch 2 (Wear OS) · Fitbit ECG `ecg-*.dat` 解码器
=====================================================================
破解自 com.fitbit.ecg 的 libecg.so (fb_ecg_lib, "ECGlib") 静态逆向 + 样本统计验证。

使用:
    python ecg_decode.py path/to/ecg-001.dat [path/to/ecg-002.dat ...]
    python ecg_decode.py path/to/ecg-001.dat --plot --outdir ./ --compare

依赖: numpy, matplotlib (画图时)
输出: 波形 numpy 数组 (fs=250 Hz), PNG/SVG 图, 终端打印格式摘要
"""
import os
import struct
import sys

import numpy as np

MAGIC = b'BIFA'
FS_HZ = 250.0  # Fitbit ECG 采样率


class EcgDatFile:
    """解析一个 ecg-*.dat 文件（19,858 字节固定长度 / BIFA 格式）"""

    def __init__(self, path):
        self.path = path
        raw = open(path, 'rb').read()
        self.file_size = len(raw)

        # --- 文件根头 (8 字节) ---
        assert raw[0:4] == MAGIC, f'bad magic: {raw[0:4]!r}'
        self.magic = raw[0:4]
        self.data_size = struct.unpack_from('<I', raw, 4)[0]  # 数据区总长(含校验和)
        self.body = raw[8:8 + self.data_size]
        self.padding = raw[8 + self.data_size:]              # 尾部零填充

        b = self.body

        # --- 固定头字段 (已确认) ---
        self.ver_flags = b[0:4]          # 01 02 03 01 -> 版本/标志
        self.fixed12 = b[4:16]           # 00 41 07 00 00 00 00 00 00 4f 1b bc(部分)
        # 元数据浮点/整数字段 (逆向确认区, 语义部分待定)
        meta = {}
        meta['f32@0x15'] = struct.unpack_from('<f', b, 0x15)[0]
        meta['u32@0x19'] = struct.unpack_from('<I', b, 0x19)[0]
        meta['f32@0x1D'] = struct.unpack_from('<f', b, 0x1D)[0]
        meta['f32@0x21'] = struct.unpack_from('<f', b, 0x21)[0]
        meta['f32@0x25'] = struct.unpack_from('<f', b, 0x25)[0]
        self.meta = meta

        # --- 采样数与波形 (逆向确认: 0xDD 起 u32 样本数, 0xE1 起 int16 LE) ---
        self.n_samples = struct.unpack_from('<I', b, 0xDD)[0]
        assert 0 < self.n_samples < 0x1D4D, f'样本数异常: {self.n_samples}'
        wave_raw = b[0xE1:0xE1 + self.n_samples * 2]
        self.samples = np.frombuffer(wave_raw, dtype='<i2').astype(np.float32)
        self.duration_s = len(self.samples) / FS_HZ

        # --- 波形之后: 帧时间戳表 + 注解区 + 校验和 ---
        pos = 0xE1 + self.n_samples * 2
        tail = b[pos:]
        n_frames = struct.unpack_from('<I', tail, 0)[0]
        if 0 < n_frames < 10000 and 4 + n_frames * 8 + 40 < len(tail):
            self.n_frames = n_frames
            arr = np.frombuffer(tail, dtype='<u4', count=1 + n_frames * 2)
            self.frame_ts_us = arr[1:1 + n_frames * 2:2].astype(np.int64)
            rest = tail[4 + n_frames * 8:]
        else:
            self.n_frames = 0
            self.frame_ts_us = np.array([], dtype=np.int64)
            rest = tail
        self.annotation = rest[:-32]   # 注解区 (2026-09-16 已破译, 见下)
        self.checksum = rest[-32:]     # 32B: 前 16B 有效 + 16B 零

        # --- 注解区结构 (双样本验证) ---
        ann = self.annotation
        self.frame_flags = np.frombuffer(ann[:469], dtype=np.uint8)   # 469 帧有效标志 (1=有效块; 每块 16 样本/64ms)
        self.ann_u32 = struct.unpack_from('<I', ann, 469)[0]          # 0x1D50 = 7504 = 469×16 (样本位宽常量)
        self.ann_zeros = ann[473:491]                                 # 18B 零
        self.ann_quality = struct.unpack_from('<3f', ann, 491)        # 3×f32: 信号占比 / 伪影占比 (≈1 互补) / 噪声水平
        offs = np.frombuffer(ann[503:], dtype='<u2')
        z = np.flatnonzero(offs == 0)                                 # 0x0000 终止符
        self.qrs_onsets = offs[:z[0]] if len(z) else offs             # 每跳 QRS onset 样本索引 (峰前 36-40ms)
        self.ann_tail = ann[503 + (len(self.qrs_onsets) + 1) * 2:]    # 尾部 16B (含公共 u32 0x00065B8F)

    def summary(self):
        ts = self.frame_ts_us
        lines = [
            f'文件       : {self.path}',
            f'文件大小   : {self.file_size} B (固定 19858)',
            f'magic      : {self.magic!r}  data_size = {self.data_size}',
            f'样本数     : {self.n_samples}  ({self.duration_s:.2f} s @ {FS_HZ:.0f} Hz)',
            f'波形       : int16 LE @ 0xE1, 幅值 [{int(self.samples.min())}, {int(self.samples.max())}]',
            f'时间戳帧   : {self.n_frames} 条 × 8B  (us, '
            f'{ts[0]/1e6:.3f}s ~ {ts[-1]/1e6:.3f}s)' if len(ts) else f'时间戳帧   : 0',
            f'注解区    : {len(self.annotation)} B | 帧标志 {int((self.frame_flags==1).sum())}/{len(self.frame_flags)} 有效'
            f' | 质量 {self.ann_quality[0]:.3f}/{self.ann_quality[1]:.3f}/{self.ann_quality[2]:.2e}'
            f' | QRS onset {len(self.qrs_onsets)} 跳' if len(self.frame_flags) else f'注解区     : {len(self.annotation)} B',
            f'校验和   : {self.checksum[:16].hex()}…',
        ]
        return '\n'.join(lines)


def plot_wave(f, outpng=None, outsvg=None, title_suffix='', show_meta=True):
    """画全览 + 2 秒细节; 可叠加两条波形"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    s = f.samples
    t = np.arange(len(s)) / FS_HZ

    if show_meta:
        fig, ax = plt.subplots(3, 1, figsize=(15, 8))
        ax[0].plot(t, s, lw=0.4)
        ax[0].set_title(f'{os.path.basename(f.path)} - full 30s{title_suffix}')
        ax[0].set_xlim(0, t[-1])
        ax[1].plot(t[:2000], s[:2000], lw=0.5)
        ax[1].set_title('first 8s')
        ax[2].plot(t[6000:7100], s[6000:7100], lw=0.7)
        ax[2].set_title('24-28.4s (R wave detail)')
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(t, s, lw=0.5)
        ax.set_title(f'{os.path.basename(f.path)} - full 30s{title_suffix}')
        fig.tight_layout()

    if outpng:
        fig.savefig(outpng, dpi=110)
    if outsvg:
        fig.savefig(outsvg)
    plt.close(fig)


def first_rpeak(s, fs=FS_HZ):
    """稳健 R 峰定位: 掩蔽饱和伪影(|s|>30000) -> 去基线 -> 正向最大偏转.

    修复: 旧实现 argmax(|s|) 会被 ecg-001 里的 -32k 接触丢失谷拉偏
    (锁定 29.776s 的 -32768 样本). R 波是 QRS 中最大的正向偏转,
    掩蔽饱和样本后取 argmax(x) 即得真 R 峰.
    """
    s = np.asarray(s, dtype=np.float32)
    sat = np.abs(s) > 30000               # 饱和伪影 (接触丢失)
    x = s.copy()
    x[sat] = np.nan
    base = float(np.nanmedian(x))         # 去基线
    x = x - base
    x[sat] = -np.inf                      # 排除候选
    i = int(np.argmax(x))
    if x[i] <= 0:                         # 无正峰兜底
        x2 = s.copy()
        x2[sat] = 0.0
        return int(np.argmax(np.abs(x2 - np.median(x2))))
    return i


def compare_waves(f1, f2, outpng=None):
    """两条录音对比: 对齐到第一个 R 峰 (饱和样本已排除)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p1, p2 = first_rpeak(f1.samples), first_rpeak(f2.samples)
    a = f1.samples[max(0, p1 - 100):p1 + 200]
    b = f2.samples[max(0, p2 - 100):p2 + 200]
    t1, t2 = p1 / FS_HZ, p2 / FS_HZ

    fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=False)
    ax[0].plot(np.arange(len(a)) / FS_HZ, a, lw=0.6)
    ax[0].axvline(100 / FS_HZ, color='r', lw=0.8, ls='--', label=f'R peak @ {t1:.3f}s')
    ax[0].set_title(f'{os.path.basename(f1.path)} - R-peak aligned (1.2s, saturation excluded)')
    ax[0].legend(fontsize=8)
    ax[1].plot(np.arange(len(b)) / FS_HZ, b, lw=0.6, color='tab:green')
    ax[1].axvline(100 / FS_HZ, color='r', lw=0.8, ls='--', label=f'R peak @ {t2:.3f}s')
    ax[1].set_title(f'{os.path.basename(f2.path)} - R-peak aligned (1.2s, saturation excluded)')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    if outpng:
        fig.savefig(outpng, dpi=110)
    plt.close(fig)
    return t1, t2


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Fitbit ECG .dat 解码器')
    ap.add_argument('files', nargs='+', help='ecg-*.dat 路径')
    ap.add_argument('--plot', action='store_true', help='出波形图')
    ap.add_argument('--svg', action='store_true', help='同时存 SVG')
    ap.add_argument('--outdir', default='.', help='输出目录')
    ap.add_argument('--compare', action='store_true', help='两条录音 R 峰对比')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fs = []
    for p in args.files:
        f = EcgDatFile(p)
        fs.append(f)
        print(f.summary())
        print('-' * 56)
        if args.plot:
            base = os.path.splitext(os.path.basename(p))[0]
            plot_wave(f,
                      outpng=os.path.join(args.outdir, base + '_wave.png'),
                      outsvg=os.path.join(args.outdir, base + '_wave.svg') if args.svg else None)
            print('已输出:', os.path.join(args.outdir, base + '_wave.png'))
    if args.compare and len(fs) >= 2:
        out = os.path.join(args.outdir, 'compare_rpeak.png')
        compare_waves(fs[0], fs[1], outpng=out)
        print('已输出:', out)


if __name__ == '__main__':
    main()