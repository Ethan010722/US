import tkinter as tk
from tkinter import filedialog, simpledialog
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, correlate
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import matplotlib.gridspec as gridspec
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def select_input_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title="Select Input File", filetypes=[("Text files", "*.txt")])

def extract_segments_by_type(input_text, num_points=10):
    segments = {1: [], 2: [], 3: [], 4: []}
    lines = input_text.split()
    i = 0
    while i < len(lines) - 2:
        if lines[i] == '0' and lines[i + 2] == '0':
            try:
                segment_type = int(lines[i + 1])
                if segment_type in segments:
                    i += 3
                    segment = []
                    for _ in range(num_points):
                        if i >= len(lines) or lines[i] == '255':
                            break
                        segment.append(int(lines[i]))
                        i += 1
                    if len(segment) == num_points:
                        segments[segment_type].append(segment)
                    else:
                        print(f"Skipping incomplete segment, type {segment_type}, length {len(segment)}")
                else:
                    i += 1
            except ValueError:
                i += 1
        else:
            i += 1
    return segments

def create_best_signal_dict(segments_by_type, num_points):
    best_signals_dict = {}
    
    valid_segments = [segments_by_type[t] for t in [1, 2, 3, 4] if segments_by_type[t]]
    if not valid_segments:
        raise ValueError("No valid segments to create best signal dictionary")
    
    width = min(len(s) for s in valid_segments)
    height = num_points
    
    for col in range(width):
        col_signals = {}
        for t in [1, 2, 3, 4]:
            if segments_by_type[t] and col < len(segments_by_type[t]):
                col_signals[t] = np.array(segments_by_type[t][col]) - 96
        
        best_signals_dict[col] = col_signals
    
    return best_signals_dict, width, height

def create_best_signal_image(best_signals_dict, width, height):
    best_signal_image = np.zeros((height, width), dtype=np.uint8)
    
    for col in range(width):
        col_signals = best_signals_dict[col]
        if not col_signals:
            continue
            
        signals = np.array([col_signals[t] for t in col_signals.keys()])
        if signals.size:
            significance = np.abs(signals)
            max_idx = np.argmax(significance, axis=0)
            best_signals = signals[max_idx, np.arange(height)]
            enhanced = np.abs(best_signals) ** 2
            normalized = (enhanced / 100) * 255
            best_signal_image[:, col] = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return best_signal_image

def plot_grayscale_image(grayscale_image, num_points, segments_length, title_suffix=""):
    sampling_interval_us = 0.2778
    speed_of_sound = 1540
    distance_multiplier = 100
    time_divisor = 25
    
    distance_max_cm = (num_points * sampling_interval_us * 1e-6 * speed_of_sound * distance_multiplier / 2)
    time_max_s = segments_length / time_divisor
    extent = [0, time_max_s, distance_max_cm, 0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(grayscale_image, cmap='gray', aspect='auto', extent=extent, origin='upper')
    
    ax.set_ylabel('Distance (cm)')
    ax.set_xlabel('Time (s)')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    plt.colorbar(cax, ax=ax, label='Intensity')
    fig.text(0.5, 0.02, f'M mode {title_suffix} (3.6MHz)', ha='center', va='center')
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig, ax, extent

class MuscleThicknessAnalyzer:
    def __init__(self, image, best_signals_dict, time_points, distance_per_pixel, time_max_s):
        self.image = image
        self.best_signals_dict = best_signals_dict
        self.time_points = time_points
        self.distance_per_pixel = distance_per_pixel
        self.height, self.width = image.shape
        self.upper_boundary = None
        self.lower_boundary = None
        self.thickness_values = None
        self.samples_per_second = self.width / time_max_s
        self.signal_threshold = 0.8

    def analyze_muscle_boundaries(self):
        if not self.best_signals_dict:
            print("No signal data available")
            return
            
        self.upper_boundary = np.zeros(self.width)
        self.lower_boundary = np.zeros(self.width)
        
        first_frame_signals = self.find_initial_boundaries()
        if first_frame_signals is None:
            print("Failed to determine initial boundaries")
            return
            
        self.track_boundaries_with_correlation(first_frame_signals)
        self.calculate_thickness()
        
    def find_initial_boundaries(self):
        first_frame_signals = {}
        for channel in self.best_signals_dict[0].keys():
            first_frame_signals[channel] = self.best_signals_dict[0][channel]
        
        if not first_frame_signals:
            return None
            
        combined_signal = np.zeros(self.height)
        for channel in first_frame_signals:
            combined_signal += np.abs(first_frame_signals[channel])
            
        peaks = self.find_signal_peaks(combined_signal)
        
        if len(peaks) < 2:
            print("Not enough signal peaks detected in the first frame")
            self.manual_select_boundaries()
            return first_frame_signals
            
        strongest_peaks = sorted(peaks, key=lambda p: combined_signal[p], reverse=True)[:2]
        strongest_peaks.sort()
        
        if len(peaks) > 2:
            strongest_signal = combined_signal[strongest_peaks[0]]
            
            close_peaks = []
            for p in peaks:
                if p != strongest_peaks[0] and p != strongest_peaks[1]:
                    if combined_signal[p] >= self.signal_threshold * strongest_signal:
                        close_peaks.append(p)
            
            if close_peaks:
                self.manual_select_boundaries()
            else:
                self.upper_boundary[0] = strongest_peaks[0]
                self.lower_boundary[0] = strongest_peaks[1]
        else:
            self.upper_boundary[0] = strongest_peaks[0]
            self.lower_boundary[0] = strongest_peaks[1]
            
        return first_frame_signals
        
    def find_signal_peaks(self, signal, min_distance=10):
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if signal[i] > 0.3 * np.max(signal):
                    peaks.append(i)
        
        if len(peaks) > 0:
            filtered_peaks = [peaks[0]]
            for i in range(1, len(peaks)):
                if peaks[i] - filtered_peaks[-1] >= min_distance:
                    filtered_peaks.append(peaks[i])
            return filtered_peaks
        return []
        
    def manual_select_boundaries(self):
        root = tk.Tk()
        root.withdraw()
        
        upper_depth = simpledialog.askinteger("Manual Selection", f"Enter upper boundary depth (0-{self.height-1})",
                                           minvalue=0, maxvalue=self.height-1)
        if upper_depth is None:
            upper_depth = self.height // 4
            
        lower_depth = simpledialog.askinteger("Manual Selection", f"Enter lower boundary depth (0-{self.height-1})",
                                           minvalue=upper_depth+1, maxvalue=self.height-1)
        if lower_depth is None:
            lower_depth = self.height * 3 // 4
            
        self.upper_boundary[0] = upper_depth
        self.lower_boundary[0] = lower_depth
        
    def track_boundaries_with_correlation(self, first_frame_signals):
        upper_template_center = int(self.upper_boundary[0])
        lower_template_center = int(self.lower_boundary[0])
        
        window_size = 20
        half_window = window_size // 2
        
        upper_start = max(0, upper_template_center - half_window)
        upper_end = min(self.height, upper_template_center + half_window)
        lower_start = max(0, lower_template_center - half_window)
        lower_end = min(self.height, lower_template_center + half_window)
        
        upper_template = np.zeros(upper_end - upper_start)
        lower_template = np.zeros(lower_end - lower_start)
        
        for channel in first_frame_signals:
            upper_template += np.abs(first_frame_signals[channel][upper_start:upper_end])
            lower_template += np.abs(first_frame_signals[channel][lower_start:lower_end])
        
        for col in range(1, self.width):
            col_signals = {}
            for channel in self.best_signals_dict[col]:
                col_signals[channel] = self.best_signals_dict[col][channel]
            
            if not col_signals:
                self.upper_boundary[col] = self.upper_boundary[col-1]
                self.lower_boundary[col] = self.lower_boundary[col-1]
                continue
                
            combined_signal = np.zeros(self.height)
            for channel in col_signals:
                combined_signal += np.abs(col_signals[channel])
                
            search_range = 30
            
            prev_upper = int(self.upper_boundary[col-1])
            upper_search_start = max(0, prev_upper - search_range)
            upper_search_end = min(self.height - len(upper_template), prev_upper + search_range)
            
            if upper_search_start < upper_search_end:
                upper_corr = np.zeros(upper_search_end - upper_search_start)
                
                for i in range(upper_search_start, upper_search_end):
                    signal_segment = combined_signal[i:i+len(upper_template)]
                    corr = np.sum(signal_segment * upper_template)
                    upper_corr[i - upper_search_start] = corr
                
                if np.max(upper_corr) > 0:
                    best_match = upper_search_start + np.argmax(upper_corr)
                    self.upper_boundary[col] = best_match
                else:
                    self.upper_boundary[col] = self.upper_boundary[col-1]
            else:
                self.upper_boundary[col] = self.upper_boundary[col-1]
            
            prev_lower = int(self.lower_boundary[col-1])
            lower_search_start = max(0, prev_lower - search_range)
            lower_search_end = min(self.height - len(lower_template), prev_lower + search_range)
            
            if lower_search_start < lower_search_end:
                lower_corr = np.zeros(lower_search_end - lower_search_start)
                
                for i in range(lower_search_start, lower_search_end):
                    signal_segment = combined_signal[i:i+len(lower_template)]
                    corr = np.sum(signal_segment * lower_template)
                    lower_corr[i - lower_search_start] = corr
                
                if np.max(lower_corr) > 0:
                    best_match = lower_search_start + np.argmax(lower_corr)
                    self.lower_boundary[col] = best_match
                else:
                    self.lower_boundary[col] = self.lower_boundary[col-1]
            else:
                self.lower_boundary[col] = self.lower_boundary[col-1]

    def calculate_thickness(self):
        if self.upper_boundary is None or self.lower_boundary is None:
            print("Boundaries not determined")
            return
            
        self.thickness_values = (self.lower_boundary - self.upper_boundary) * self.distance_per_pixel
        print(f"Calculated muscle thickness for {len(self.thickness_values)} time points")
        if len(self.thickness_values) > 0:
            print(f"Sample thickness values: {self.thickness_values[:5]} cm")

    def plot_boundaries_on_image(self, fig, ax):
        if self.upper_boundary is None or self.lower_boundary is None:
            return
            
        x_vals = self.time_points
        upper_y_vals = self.upper_boundary * self.distance_per_pixel
        lower_y_vals = self.lower_boundary * self.distance_per_pixel
        
        ax.plot(x_vals, upper_y_vals, 'r-', linewidth=2, label='Upper Boundary')
        ax.plot(x_vals, lower_y_vals, 'b-', linewidth=2, label='Lower Boundary')
        ax.legend(loc='upper right')

    def plot_thickness_over_time(self):
        if self.thickness_values is None or len(self.thickness_values) == 0:
            print("No valid thickness data for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mean_thickness = np.mean(self.thickness_values)
        std_thickness = np.std(self.thickness_values)
        threshold = 2
        
        normal_mask = np.abs(self.thickness_values - mean_thickness) <= threshold * std_thickness
        
        ax.scatter(self.time_points[normal_mask], self.thickness_values[normal_mask], 
                  color='blue', alpha=0.7, label='Measured Thickness')
        ax.scatter(self.time_points[~normal_mask], self.thickness_values[~normal_mask], 
                  color='gray', alpha=0.3, label='Outliers')
        
        smooth_y = lowess(self.thickness_values[normal_mask], 
                         self.time_points[normal_mask],
                         frac=0.15,
                         it=3,
                         delta=0.1 * np.mean(np.diff(self.time_points)),
                         return_sorted=False)
        
        f = interp1d(self.time_points[normal_mask], smooth_y, 
                    kind='linear', fill_value='extrapolate')
        smooth_y_full = f(self.time_points)
        
        ax.plot(self.time_points, smooth_y_full, color='red', 
               linewidth=2, label='Smoothed Fit')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Muscle Thickness (cm)')
        ax.set_title('Muscle Thickness Over Time')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        mean_thickness = np.mean(self.thickness_values)
        ax.text(0.05, 0.95, f'Mean Thickness: {mean_thickness:.4f} cm', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig, ax

    def plot_comprehensive_results(self):
        if self.thickness_values is None:
            print("No thickness data available")
            return
            
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        
        ax1 = plt.subplot(gs[0])
        ax1.imshow(self.image, cmap='gray', aspect='auto', 
                  extent=[0, self.time_points[-1], self.height * self.distance_per_pixel, 0], 
                  origin='upper')
        
        x_vals = self.time_points
        upper_y_vals = self.upper_boundary * self.distance_per_pixel
        lower_y_vals = self.lower_boundary * self.distance_per_pixel
        
        ax1.plot(x_vals, upper_y_vals, 'r-', linewidth=2, label='Upper Boundary')
        ax1.plot(x_vals, lower_y_vals, 'b-', linewidth=2, label='Lower Boundary')
        
        ax1.set_title('Muscle Boundaries Detection')
        ax1.set_ylabel('Distance (cm)')
        ax1.legend(loc='upper right')
        
        ax2 = plt.subplot(gs[1])
        ax2.scatter(self.time_points, self.thickness_values, color='blue', alpha=0.7, label='Measured Thickness')
        
        smooth_y = lowess(self.thickness_values, self.time_points, 
                         frac=0.15, 
                         it=3,
                         delta=0.1 * np.mean(np.diff(self.time_points)),
                         return_sorted=False)
        ax2.plot(self.time_points, smooth_y, color='red', linewidth=2, label='Smoothed Fit')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Thickness (cm)')
        ax2.set_title('Muscle Thickness Over Time')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        mean_thickness = np.mean(self.thickness_values)
        ax2.text(0.05, 0.95, f'Mean Thickness: {mean_thickness:.4f} cm', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig

def main():
    input_file = select_input_file()
    if not input_file:
        print("No input file selected. Exiting.")
        return
    
    with open(input_file, "r") as file:
        input_text = file.read()
    
    num_points = 500
    segments_by_type = extract_segments_by_type(input_text, num_points)
    
    # Original four-channel images are not displayed
    
    try:
        best_signals_dict, width, height = create_best_signal_dict(segments_by_type, num_points)
        best_signal_image = create_best_signal_image(best_signals_dict, width, height)
        
        if width > 0:
            sampling_interval_us = 0.2778
            speed_of_sound = 1540
            distance_multiplier = 100
            time_divisor = 25
            
            distance_max_cm = (num_points * sampling_interval_us * 1e-6 * speed_of_sound * distance_multiplier / 2) 
            distance_per_pixel = distance_max_cm / num_points
            time_max_s = width / time_divisor
            time_points = np.linspace(0, time_max_s, width)
            
            fig, ax, extent = plot_grayscale_image(best_signal_image, num_points, width, "(Best Signal)")
            
            analyzer = MuscleThicknessAnalyzer(best_signal_image, best_signals_dict, time_points, distance_per_pixel, time_max_s)
            analyzer.analyze_muscle_boundaries()
            
            analyzer.plot_boundaries_on_image(fig, ax)
            plt.draw()
            
            analyzer.plot_thickness_over_time()
            analyzer.plot_comprehensive_results()
        else:
            print("Cannot determine width of best signal image.")
    except ValueError as e:
        print(f"Error creating best signal image: {e}")
    except Exception as e:
        import traceback
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    
    plt.show()

if __name__ == "__main__":
    main()
