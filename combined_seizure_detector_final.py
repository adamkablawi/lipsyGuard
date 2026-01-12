"""
Combined Seizure Detection System
Combines ML-based accelerometer detection (70%) with HRV-based detection (30%)

Requirements:
- triaxial_seizure_model.keras (trained model)
- triaxial_seizure_encoder.pkl (label encoder)
- axis_normalization_stats.pkl (normalization statistics)
"""

import numpy as np
import pickle
import tensorflow as tf
from scipy import signal
from scipy.signal import find_peaks


class AccelerometerSeizureDetector:
    """ML-based accelerometer seizure detector using trained CNN model"""

    def __init__(self, model_path='triaxial_seizure_model.keras',
                 encoder_path='triaxial_seizure_encoder.pkl',
                 stats_path='axis_normalization_stats.pkl'):
        self.model = tf.keras.models.load_model(model_path)

        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)

        with open(stats_path, 'rb') as f:
            self.axis_stats = pickle.load(f)

    def predict(self, x_data, y_data, z_data):
        """Predict seizure probability from accelerometer data"""
        x_data = np.array(x_data, dtype=float)
        y_data = np.array(y_data, dtype=float)
        z_data = np.array(z_data, dtype=float)

        if len(x_data) != 200 or len(y_data) != 200 or len(z_data) != 200:
            raise ValueError(f"Expected 200 samples per axis")

        # Stack and normalize
        X = np.stack([x_data, y_data, z_data], axis=-1)
        X_normalized = np.zeros_like(X)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            mean = self.axis_stats[axis]['mean']
            std = self.axis_stats[axis]['std']
            X_normalized[:, i] = (X[:, i] - mean) / std

        X_normalized = np.expand_dims(X_normalized, axis=0)

        # Get prediction
        predictions_proba = self.model.predict(X_normalized, verbose=0)
        proba = predictions_proba[0]

        predicted_class_idx = np.argmax(proba)
        predicted_class = self.encoder.classes_[predicted_class_idx]

        # Get seizure probability
        seizure_idx = np.where(self.encoder.classes_ == 'seizure')[0]
        if len(seizure_idx) > 0:
            seizure_probability = float(proba[seizure_idx[0]] * 100)
        else:
            seizure_probability = 0.0

        return {
            'seizure_probability': seizure_probability,
            'is_seizure': seizure_probability >= 50.0,
            'predicted_class': predicted_class,
            'confidence': float(proba[predicted_class_idx] * 100)
        }


class HRVSeizureDetector:
    """Real-time HRV-based seizure detector using PPG signal processing"""

    def __init__(self, sampling_rate=16.0, seizure_threshold=50.0):
        self.sampling_rate = sampling_rate
        self.seizure_threshold = seizure_threshold

    def preprocess_ppg(self, ppg_data):
        """Filter PPG signal (0.5-8 Hz bandpass)"""
        ppg = np.array(ppg_data, dtype=float)

        if len(ppg) != 200:
            return None, f"Expected 200 samples, got {len(ppg)}"

        # Bandpass filter
        nyquist = self.sampling_rate / 2
        low = max(0.01, min(0.5 / nyquist, 0.99))
        high = max(low + 0.01, min(8.0 / nyquist, 0.99))

        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, ppg), "OK"

    def extract_rr_intervals(self, ppg_filtered):
        """Extract RR intervals from filtered PPG"""
        ppg_norm = (ppg_filtered - np.mean(ppg_filtered)) / np.std(ppg_filtered)

        # Find peaks
        min_distance = int(self.sampling_rate * 0.33)
        peaks, _ = find_peaks(ppg_norm, distance=min_distance, prominence=0.3)

        if len(peaks) < 3:
            return None, f"Only {len(peaks)} peaks found"

        # Calculate RR intervals in ms
        rr_ms = np.diff(peaks / self.sampling_rate) * 1000
        rr_valid = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]

        if len(rr_valid) < 2:
            return None, "Too few valid intervals"

        return rr_valid, f"{len(rr_valid)} intervals from {len(peaks)} peaks"

    def calculate_hrv_features(self, rr_intervals):
        """Calculate time and frequency domain HRV features"""
        rr = np.array(rr_intervals)

        # Time domain
        mean_rr = np.mean(rr)
        std_rr = np.std(rr, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(rr)**2))
        nn50 = np.sum(np.abs(np.diff(rr)) > 50)
        pnn50 = (nn50 / len(np.diff(rr)) * 100) if len(rr) > 1 else 0
        cv = (std_rr / mean_rr * 100) if mean_rr > 0 else 0
        mean_hr = 60000 / mean_rr if mean_rr > 0 else 0

        features = {
            'mean_rr': mean_rr,
            'std_rr': std_rr,
            'rmssd': rmssd,
            'pnn50': pnn50,
            'cv': cv,
            'mean_hr': mean_hr
        }

        # Frequency domain (if enough data)
        if len(rr) >= 10:
            try:
                t_orig = np.cumsum(rr) / 1000
                t_uniform = np.arange(0, t_orig[-1], 1.0 / self.sampling_rate)
                rr_uniform = np.interp(t_uniform, t_orig, rr)
                rr_detrended = signal.detrend(rr_uniform)

                nperseg = min(256, len(rr_detrended))
                freqs, psd = signal.welch(rr_detrended, fs=self.sampling_rate, nperseg=nperseg)

                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs < 0.4)

                lf_power = np.trapezoid(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
                hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0

                features['lf_power'] = lf_power
                features['hf_power'] = hf_power
                features['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else 0
            except:
                pass

        return features

    def calculate_seizure_score(self, features):
        """Rule-based seizure scoring from HRV features"""
        score = 0.0
        indicators = []

        if features['std_rr'] > 150:
            score += 35
            indicators.append('very_high_variability')
        elif features['std_rr'] > 100:
            score += 25
            indicators.append('high_variability')

        if features['std_rr'] < 20:
            score += 20
            indicators.append('low_variability')

        if features['rmssd'] > 200:
            score += 30
            indicators.append('extreme_rmssd')
        elif features['rmssd'] > 100:
            score += 20
            indicators.append('high_rmssd')
        elif features['rmssd'] < 15:
            score += 20
            indicators.append('low_parasympathetic')

        if features['cv'] > 40:
            score += 20
            indicators.append('very_high_cv')
        elif features['cv'] > 30:
            score += 15
            indicators.append('excessive_cv')

        if features['mean_hr'] > 120:
            score += 20
            indicators.append('tachycardia')
        elif features['mean_hr'] > 100:
            score += 10
            indicators.append('elevated_hr')

        if 'lf_hf_ratio' in features:
            if features['lf_hf_ratio'] > 3.0:
                score += 20
                indicators.append('high_sympathetic')
            elif features['lf_hf_ratio'] < 0.5:
                score += 15
                indicators.append('low_sympathetic')

        return min(score, 100), indicators

    def predict(self, ppg_data):
        """Predict seizure from PPG data"""
        ppg_filtered, msg = self.preprocess_ppg(ppg_data)
        if ppg_filtered is None:
            return {'error': msg, 'seizure_probability': 0.0, 'is_seizure': False}

        rr_intervals, msg = self.extract_rr_intervals(ppg_filtered)
        if rr_intervals is None:
            return {'error': msg, 'seizure_probability': 0.0, 'is_seizure': False}

        features = self.calculate_hrv_features(rr_intervals)
        score, indicators = self.calculate_seizure_score(features)

        return {
            'seizure_probability': float(score),
            'is_seizure': score >= self.seizure_threshold,
            'confidence': float(score if score >= self.seizure_threshold else 100 - score),
            'features': features,
            'indicators': indicators
        }


class CombinedSeizureDetector:
    """
    Combined seizure detector with weighted predictions:
    - Accelerometer ML: 70%
    - HRV Analysis: 30%
    """

    def __init__(self, model_path='triaxial_seizure_model.keras',
                 encoder_path='triaxial_seizure_encoder.pkl',
                 stats_path='axis_normalization_stats.pkl',
                 accel_weight=0.7, hrv_weight=0.3):

        print("Initializing Combined Seizure Detector...")
        print(f"  Accelerometer weight: {accel_weight*100:.0f}%")
        print(f"  HRV weight: {hrv_weight*100:.0f}%")

        self.accel_detector = AccelerometerSeizureDetector(model_path, encoder_path, stats_path)
        self.hrv_detector = HRVSeizureDetector()

        self.accel_weight = accel_weight
        self.hrv_weight = hrv_weight

        print("✓ Combined detector ready!\n")

    def predict(self, x_data, y_data, z_data, ppg_data, verbose=True):
        """
        Make combined prediction from accelerometer and PPG data

        Args:
            x_data: X-axis accelerometer (200 samples at 16Hz)
            y_data: Y-axis accelerometer (200 samples)
            z_data: Z-axis accelerometer (200 samples)
            ppg_data: PPG data (200 samples at 16Hz)
            verbose: Print results

        Returns:
            dict: Combined prediction with detailed breakdown
        """
        # Get individual predictions
        accel_result = self.accel_detector.predict(x_data, y_data, z_data)
        hrv_result = self.hrv_detector.predict(ppg_data)

        # Handle HRV errors gracefully
        if 'error' in hrv_result:
            hrv_prob = 0.0
            hrv_error = hrv_result['error']
        else:
            hrv_prob = hrv_result['seizure_probability']
            hrv_error = None

        # Calculate weighted probability
        combined_probability = (
            self.accel_weight * accel_result['seizure_probability'] +
            self.hrv_weight * hrv_prob
        )

        is_seizure = combined_probability >= 50.0
        confidence = combined_probability if is_seizure else (100 - combined_probability)

        result = {
            'combined_seizure_probability': float(combined_probability),
            'is_seizure': is_seizure,
            'confidence': float(confidence),
            'classification': 'SEIZURE' if is_seizure else 'NORMAL',
            'accelerometer': {
                'seizure_probability': accel_result['seizure_probability'],
                'weight': self.accel_weight * 100,
                'contribution': accel_result['seizure_probability'] * self.accel_weight,
                'predicted_class': accel_result['predicted_class']
            },
            'hrv': {
                'seizure_probability': hrv_prob,
                'weight': self.hrv_weight * 100,
                'contribution': hrv_prob * self.hrv_weight,
                'error': hrv_error,
                'indicators': hrv_result.get('indicators', []),
                'features': hrv_result.get('features', {})
            }
        }

        if verbose:
            self._print_result(result)

        return result

    def predict_batch(self, x_batch, y_batch, z_batch, ppg_batch):
        """Predict for multiple samples"""
        results = []
        for i in range(len(x_batch)):
            result = self.predict(
                x_batch[i], y_batch[i], z_batch[i], ppg_batch[i],
                verbose=False
            )
            results.append(result)
        return results

    def _print_result(self, result):
        """Print formatted result"""
        print("=" * 70)
        print("COMBINED SEIZURE DETECTION RESULT")
        print("=" * 70)

        print(f"\n{'CLASSIFICATION:':<25} {result['classification']}")
        print(f"{'Seizure Probability:':<25} {result['combined_seizure_probability']:.1f}%")
        print(f"{'Confidence:':<25} {result['confidence']:.1f}%")

        print("\n" + "-" * 70)
        print("COMPONENT BREAKDOWN")
        print("-" * 70)

        # Accelerometer
        accel = result['accelerometer']
        print(f"\n1. ACCELEROMETER ML (Weight: {accel['weight']:.0f}%)")
        print(f"   Seizure Probability:  {accel['seizure_probability']:.1f}%")
        print(f"   Contribution:         {accel['contribution']:.1f}%")
        print(f"   Predicted Class:      {accel['predicted_class']}")

        # HRV
        hrv = result['hrv']
        print(f"\n2. HRV ANALYSIS (Weight: {hrv['weight']:.0f}%)")

        if hrv['error']:
            print(f"   Error:                {hrv['error']}")
            print(f"   Contribution:         0.0%")
        else:
            print(f"   Seizure Probability:  {hrv['seizure_probability']:.1f}%")
            print(f"   Contribution:         {hrv['contribution']:.1f}%")

            if hrv['indicators']:
                print(f"   Indicators:           {', '.join(hrv['indicators'])}")

            if hrv['features']:
                f = hrv['features']
                print(f"   HR:                   {f['mean_hr']:.1f} bpm")
                print(f"   SDNN:                 {f['std_rr']:.1f} ms")
                print(f"   RMSSD:                {f['rmssd']:.1f} ms")

        print("\n" + "=" * 70)


def generate_synthetic_accel(data_type='normal'):
    """Generate synthetic accelerometer data"""
    t = np.linspace(0, 12.5, 200)

    if data_type == 'normal':
        x = np.random.normal(0, 0.05, 200)
        y = np.random.normal(0, 0.05, 200)
        z = np.random.normal(0, 0.1, 200)
    else:
        x = np.sin(2 * np.pi * 5 * t) * 2.0 + np.random.normal(0, 0.5, 200)
        y = np.cos(2 * np.pi * 5 * t) * 2.0 + np.random.normal(0, 0.5, 200)
        z = np.sin(2 * np.pi * 3 * t) * 1.5 + 9.8 + np.random.normal(0, 0.3, 200)

    return x, y, z


def generate_synthetic_ppg(data_type='normal'):
    """Generate synthetic PPG data"""
    t = np.linspace(0, 12.5, 200)

    if data_type == 'normal':
        ppg = np.zeros(200)
        for i in range(16):
            pulse_time = i * 0.8
            if pulse_time < 12.5:
                ppg += np.exp(-((t - pulse_time) ** 2) / 0.015)
        ppg += 0.1 * np.sin(2 * np.pi * 0.25 * t)
        ppg += np.random.normal(0, 0.02, 200)
    else:
        ppg = np.zeros(200)
        times = [0, 0.5, 1.3, 1.8, 2.6, 3.1, 4.0, 4.5, 5.3, 6.0, 6.8, 7.5, 8.2, 9.0, 9.7, 10.5, 11.2, 12.0]
        for pulse_time in times:
            if pulse_time < 12.5:
                amplitude = 0.7 + np.random.uniform(-0.3, 0.3)
                ppg += amplitude * np.exp(-((t - pulse_time) ** 2) / 0.015)
        ppg += 0.2 * np.sin(2 * np.pi * 0.4 * t)
        ppg += np.random.normal(0, 0.05, 200)

    ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
    return ppg * 1000 + 2000


def main():
    """Example usage"""
    print("\n" + "=" * 70)
    print("COMBINED SEIZURE DETECTION SYSTEM")
    print("Accelerometer ML (70%) + HRV Analysis (30%)")
    print("=" * 70 + "\n")

    # Initialize
    try:
        detector = CombinedSeizureDetector(
            model_path='triaxial_seizure_model.keras',
            encoder_path='triaxial_seizure_encoder.pkl',
            stats_path='axis_normalization_stats.pkl',
            accel_weight=0.7,
            hrv_weight=0.3
        )
    except Exception as e:
        print(f"Error loading model files: {e}")
        print("\nRequired files:")
        print("  - triaxial_seizure_model.keras")
        print("  - triaxial_seizure_encoder.pkl")
        print("  - axis_normalization_stats.pkl")
        return

    # Example 1: Normal
    print("\nExample 1: Normal Activity")
    print("-" * 70)
    x_n, y_n, z_n = generate_synthetic_accel('normal')
    ppg_n = generate_synthetic_ppg('normal')
    detector.predict(x_n, y_n, z_n, ppg_n)

    # Example 2: Seizure
    print("\n\nExample 2: Seizure-like Activity")
    print("-" * 70)
    x_s, y_s, z_s = generate_synthetic_accel('seizure')
    ppg_s = generate_synthetic_ppg('seizure')
    detector.predict(x_s, y_s, z_s, ppg_s)

    # Example 3: Batch
    print("\n\nExample 3: Batch Processing")
    print("-" * 70)
    results = detector.predict_batch(
        [x_n, x_s, x_n],
        [y_n, y_s, y_n],
        [z_n, z_s, z_n],
        [ppg_n, ppg_s, ppg_n]
    )

    for i, r in enumerate(results):
        print(f"\nSample {i+1}: {r['classification']} ({r['combined_seizure_probability']:.1f}%)")
        print(f"  Accel: {r['accelerometer']['seizure_probability']:.1f}% | HRV: {r['hrv']['seizure_probability']:.1f}%")

    print("\n" + "=" * 70)
    print("USAGE:")
    print("  detector = CombinedSeizureDetector()")
    print("  result = detector.predict(x, y, z, ppg)  # All 200 samples at 16Hz")
    print("=" * 70)


if __name__ == "__main__":
    main()
