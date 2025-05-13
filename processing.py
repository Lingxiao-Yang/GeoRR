import os
import numpy as np
import fcwt
import pywt
import matplotlib.pyplot as plt
from scipy import signal


class FilterHR:
    @staticmethod
    def _confidence_score(spectrum: np.ndarray, threshold: float = 0.6667) -> float:
        peaks, _ = signal.find_peaks(spectrum)
        if len(peaks) == 0:
            return 0.0
        max_peak = spectrum[peaks].max()
        peak_candidates = peaks[spectrum[peaks] >= threshold * max_peak]
        return max_peak / spectrum[peak_candidates].sum()

    @staticmethod
    def _lowpass_filter(current: np.ndarray, cutoff: float, fs: int) -> np.ndarray:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)
        return signal.lfilter(b, a, current)

    def __init__(
        self,
        channels: list[str],
        intensity: np.ndarray,
        sample_rate: int = 2e4,
        output_DIR: str | None = None,
        verbose: bool = False
    ):
        self.channels = list(channels)
        self.I = np.asarray(intensity, dtype=np.float32)
        self.sample_rate = sample_rate
        self.output_DIR = output_DIR
        self.verbose = verbose

    def run(self):
        n_channels, T = self.I.shape

        hr   = np.full(n_channels, np.nan, dtype=np.float32)      # Heart Rate predictions
        conf = np.zeros(n_channels,         dtype=np.float32)     # Confidence scores, see _confidence_score()

        for n in range(n_channels):
            # ---------- (0) pre‑processing --------------------------
            sig_filt = self.I[n] - self.I[n].mean()          # detrend (zero‑mean)
            sig_filt = self._lowpass_filter(sig_filt, 400, self.sample_rate)  # lowpass filter

            # ---------- (1) wide‑band CWT wavelet filter ------------
            # Option 1: use fcwt
            #! Question: Only Morlet wavelet defined
            # _freqs, coeffs = fcwt.cwt(input=sig_filt, fs=10000, f0=7, f1=54, fn=10)
            
            
            # Option 2: use pywt
            dt = 1 / self.sample_rate
            freqs = np.linspace(7, 54, 10)
            fc = pywt.central_frequency('gaus1')
            scales = fc / (freqs * dt)
            
            coeffs, _freqs = pywt.cwt(data=sig_filt, scales=scales, wavelet='gaus1', sampling_period=dt)

            # Pass coeffs through a ReLU Function
            coeffs[coeffs < 0] = 0.0
            
            if self.verbose:
                print(f"Channel {n + 1}:")
                print("coeffs.shape", coeffs.shape)
                print("coeffs.max", coeffs.max())
                print("coeffs.min", coeffs.min())
                print("freqs.shape", _freqs.shape)
                print("_freqs.max", _freqs.max())
                print("_freqs.min", _freqs.min())
                print()
                
                plt.figure(figsize=(18, 6))
                plt.imshow(
                    np.abs(coeffs),
                    extent=[0, T, _freqs.min(), _freqs.max()],
                    aspect='auto',
                    origin='lower',
                    cmap='jet'
                )
                plt.colorbar(label='Magnitude')
                plt.title(f"{self.channels[n]} CWT Coefficients")
                plt.xlabel("Time (frames)")
                plt.ylabel("Frequency (Hz)")
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_DIR, f"cwt_{self.channels[n]}.png"))
                plt.show()

            sig_filt  = np.abs(coeffs).sum(axis=0)       # wide‑band envelope
            
            # Scale to 0-1
            sig_filt  = (sig_filt - sig_filt.min()) / (sig_filt.max() - sig_filt.min())
            
            # Envelope detection
            cutoff_hz = 300 / 60.0 #300 bpm
            envelop = sig_filt
            envelop  = self._lowpass_filter(envelop, cutoff_hz, self.sample_rate)
            # Scale to 0-1
            envelop  = (envelop - envelop.min()) / (envelop.max() - envelop.min())
            
            plt.figure(figsize=(12, 4))
            plt.plot(sig_filt)
            plt.plot(envelop, color='red', linestyle='--')
            plt.title(f"{self.channels[n]} CWT filtered signal")
            plt.xlabel("Frame")
            plt.ylabel("Intensity")
            plt.xlim(0, T)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_DIR, f"filtered_{self.channels[n]}.png"))
            if self.verbose:
                plt.show()
            else:
                plt.close()
            

        

            # ---------- (3) FFT peak search -------------------------
            freqs = np.fft.fftfreq(len(envelop), 1 / self.sample_rate) * 60
            spectrum = np.abs(np.fft.fft(envelop))
            # band‑pass mask (50–190 bpm)
            band_mask          = (freqs >= 50) & (freqs <= 190)
            spectrum[~band_mask] = 0.0

            # restrict to positive freqs for convenience
            pos = freqs > 0
            if not spectrum[pos].any():
                conf[n] = 0.0
                continue

            pk_idx  = np.argmax(spectrum[pos])
            hr[n]   = freqs[pos][pk_idx]
            conf[n] = self._confidence_score(spectrum[pos])

            # ---------- optional plotting ---------------------------
            if self.output_DIR:
                os.makedirs(self.output_DIR, exist_ok=True)
                plt.figure(figsize=(12, 4))
                plt.plot(freqs[pos], spectrum[pos])
                plt.title(f"{self.channels[n]} - HR {hr[n]:.1f} bpm  (conf {conf[n]:.2f})")
                plt.xlabel("Frequency (bpm)")
                plt.ylabel("Amplitude")
                plt.xlim(50, 190)
                plt.tight_layout()
                fname = f"spectrum_{self.channels[n]}.png"
                plt.savefig(os.path.join(self.output_DIR, fname))
                plt.close()

        return hr, conf
