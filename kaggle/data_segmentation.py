from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from typing import List, Tuple, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGDataProcessor:

    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the EEG data processor with configuration.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.marker_mapping = {
            '/Marker/1': 0,  # Left motor imagery
            '/Marker/2': 1,  # Right motor imagery  
            '/Marker/3': 2,  # Relaxed state
        }
        
        self.feature_columns = [
            f'{band}_{electrode}' 
            for band in ['Delta', 'Theta', 'Alpha', 'Beta']
            for electrode in ['TP9', 'AF7', 'AF8', 'TP10']
        ]
        
        self.marker_data = pd.DataFrame()


    def _load_single_file(self, file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and separate regular data and marker data from a single CSV file.
        """
        try:
            df = pd.read_csv(file_path, dtype={"Elements": str})
            
            # Separate regular data (with milliseconds) from marker data
            has_milliseconds = df['TimeStamp'].str.contains(r'\.', na=False)
            
            regular_df = self._process_dataframe_chunk(df[has_milliseconds])
            marker_df = self._process_dataframe_chunk(df[~has_milliseconds])
            
            logger.info(f"Loaded {len(regular_df)} regular rows and {len(marker_df)} marker rows from {file_path.name}")
            return regular_df, marker_df
            
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            return pd.DataFrame(), pd.DataFrame()
        

    def _process_dataframe_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts all timestamps to np.int64 format and removes all duplicates.
        """
        if df.empty:
            return df
            
        df = df.copy()
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        return df.drop_duplicates(subset=['TimeStamp'], keep='first')
    

    def load_and_combine_data(self) -> pd.DataFrame:
        """
        Loads and combines all CSV files in the raw data directory.
        """
        csv_files = list(self.data_dir.glob('*.csv'))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
            
        regular_dfs, marker_dfs = [], []
        
        for file in csv_files:
            regular_df, marker_df = self._load_single_file(file)
            if not regular_df.empty:
                regular_dfs.append(regular_df)
            if not marker_df.empty:
                marker_dfs.append(marker_df)
                
        if not regular_dfs:
            raise ValueError("No regular data loaded from any files")
            
        # Combines and removes duplicated rows
        combined_df = pd.concat(regular_dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['TimeStamp'], keep='first')
        logger.info(f"Combined {len(combined_df)} total regular rows from {len(regular_dfs)} files")
        
        # Handle marker data
        if marker_dfs:
            self.marker_data = pd.concat(marker_dfs, ignore_index=True)
            self.marker_data = self.marker_data.drop_duplicates(subset=['TimeStamp'], keep='first')
            logger.info(f"Combined {len(self.marker_data)} total marker rows")
        else:
            logger.info("No marker data found")
        
        return combined_df
    

    # Preprocessing data from main segment (raw temporal data without markers)
    def _validate_segment_boundaries(self, marker_time: pd.Timestamp, start_ms: int, end_ms: int, 
                                   min_time: pd.Timestamp, max_time: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Checks if the onset time of a movement falls within the predefined 400ms window.

        This function calculates the time window around a marker by adding -200ms to the marker to 
        determine the start offset and +200ms to determine the end offset. 

        Note: For motor imagery classification, we analyze a 400ms window (-200ms to +200ms) around each marker event.
        This window captures the critical neural patterns associated with movement imagination while avoiding
        contamination from preceding/following events in the experimental paradigm.
        """
        seg_start = marker_time + pd.Timedelta(milliseconds=start_ms)
        seg_end = marker_time + pd.Timedelta(milliseconds=end_ms)
        
        if seg_start < min_time or seg_end > max_time:
            raise ValueError(f"Segment at {marker_time} outside data range")
            
        return seg_start, seg_end
    

    def _extract_segmentdata_helper(self, df: pd.DataFrame, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> pd.DataFrame:
        """
        Extracts EEG data between validated start and end timestamps.

        This function uses the boundaries determined by _validate_segment_boundaries
        to locate the closest data points to the specified timestamps and extract the
        corresponding EEG segment. It ensures the segment includes at least 10 samples 
        and that the requested range falls within the bounds of the available data.
        """
        start_idx = max(0, df.index.searchsorted(seg_start, side='left'))
        end_idx = min(len(df), df.index.searchsorted(seg_end, side='right'))
        
        if start_idx >= end_idx:
            raise ValueError("Invalid segment indices")
            
        segment = df.iloc[start_idx:end_idx]
        
        if len(segment) < 10:  # Minimum samples check
            raise ValueError(f"Segment too short ({len(segment)} samples)")
            
        return segment


    def _resample_segment(self, X: np.ndarray, target_samples: int) -> np.ndarray:
        """
        Resamples a segment of EEG data to a fixed number of samples.

        This function is used to ensure that the number of samples between each marker-defined
        segment is consistent. If the number of rows in the segment is greater than the target, 
        the function downsamples by selecting evenly spaced indices. If the segment is shorter 
        than the target, it upsamples using linear interpolation across each EEG channel.
        """
        if len(X) == target_samples:
            return X
            
        elif len(X) > target_samples:
            indices = np.linspace(0, len(X) - 1, target_samples, dtype=int)
            return X[indices]
        
        else:
            old_indices = np.arange(len(X))
            new_indices = np.linspace(0, len(X) - 1, target_samples)
            
            X_resampled = np.zeros((target_samples, X.shape[1]))
            for ch in range(X.shape[1]):
                f = interp1d(old_indices, X[:, ch], kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
                X_resampled[:, ch] = f(new_indices)

            return X_resampled


    # Preprocessing marker data 
    def _get_valid_markers(self, marker_column: str = 'Elements') -> pd.DataFrame:
        """
        Filters rows containing  marker strings (e.g., 'Marker/1', 'Marker/2', 'Marker/3') from the 
        specified column ('Elements') containing the marker data.
        """
        marker_mask = (self.marker_data[marker_column]
                      .fillna('')
                      .astype(str)
                      .str.contains(r'Marker/[123]', na=False))
        
        marker_positions = self.marker_data[marker_mask].copy()
        
        # Standardize marker format and filters valid markers 
        marker_positions[marker_column] = (marker_positions[marker_column].astype(str).str.replace(r'^/?', '/', regex=True))

        valid_markers = marker_positions[marker_positions[marker_column].isin(self.marker_mapping.keys())]
        
        logger.info(f"Found {len(valid_markers)} relevant markers")

        if len(valid_markers) > 0:
            marker_counts = valid_markers[marker_column].value_counts()
            logger.info(f"Marker distribution:\n{marker_counts}")
        else:
            logger.warning("No valid markers found!")
            
        return valid_markers


    def extract_marker_segments(self, df: pd.DataFrame, marker_column: str = 'Elements', 
                        start_ms: int = -200, end_ms: int = 200, target_samples: int = 100) -> List[Tuple[np.ndarray, int]]:
        """
        Extract segments of data around each marker with fixed length.
        """
        segments = []
        
        # Prepare dataframe
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('TimeStamp').sort_index()
        
        min_time, max_time = df.index.min(), df.index.max()
        logger.info(f"Data time range: {min_time} to {max_time}")
        
        # Get valid markers
        marker_positions = self._get_valid_markers(marker_column)
        if marker_positions.empty:
            return []
        
        # Check for missing feature columns
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return []
        
        # Process each marker
        for _, row in marker_positions.iterrows():
            try:
                marker_time = row['TimeStamp']
                marker = row[marker_column]
                
                # Validate and extract segment
                seg_start, seg_end = self._validate_segment_boundaries(marker_time=marker_time, 
                                                                       start_ms=start_ms, end_ms=end_ms, 
                                                                       min_time=min_time, max_time=max_time)
                
                segment = self._extract_segmentdata_helper(df=df, seg_start=seg_start, seg_end=seg_end)
                
                # Extract features and check for NaN
                X = segment[self.feature_columns].values
                if np.isnan(X).any():
                    logger.warning(f"NaN values found in segment at {marker_time}, skipping")
                    continue
                
                # Resample to target size
                X = self._resample_segment(X, target_samples)
                
                if X.shape[0] != target_samples:
                    logger.warning(f"Could not resample segment to target size {target_samples}, got {X.shape[0]}, skipping")
                    continue
                
                # Get label and add segment
                y = self.marker_mapping[marker]
                segments.append((X, y))
                logger.info(f"Successfully extracted segment at {marker_time} with shape {X.shape}")
                
            except Exception as e:
                logger.error(f"Error processing segment at {marker_time}: {e}")
                continue
        
        logger.info(f"Extracted {len(segments)} valid segments")
        return segments


    def _create_metadata(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Returns a dictionary containing the metadata of the processed data. 
        """
        return {
            'n_samples': len(X),
            'n_channels': X.shape[1],
            'n_timesteps': X.shape[2],
            'n_classes': len(np.unique(y)),
            'class_mapping': {v: k for k, v in self.marker_mapping.items()},
            'feature_columns': self.feature_columns,
            'class_names': {
                0: 'Left motor imagery',
                1: 'Right motor imagery', 
                2: 'Relaxed state'
            }
        }


    def process_and_save(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all data and save in format compatible with EEGNet.
        """
        df = self.load_and_combine_data()
        segments = self.extract_marker_segments(df)
        
        if not segments:
            raise ValueError("No valid segments found!")
            
        # Combine all segments
        X = np.stack([s[0] for s in segments])
        y = np.array([s[1] for s in segments])
        
        # Standardize data
        X_reshaped = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape[0], X.shape[2], X.shape[1]).transpose(0, 2, 1)
        
        # Save processed data
        np.save(self.output_dir / 'X.npy', X)
        np.save(self.output_dir / 'y.npy', y)
        
        # Save metadata
        metadata = self._create_metadata(X, y)
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Processed and saved {len(X)} segments")
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y


def main():
    data_dir = 'data/raw/' # replace with own path 
    output_dir = 'data/processed' # replace with own path 
    
    processor = EEGDataProcessor(data_dir=data_dir, output_dir=output_dir)
    X, y = processor.process_and_save()

    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    logger.info(f"Unique labels: {np.unique(y)}")


if __name__ == "__main__":
    main()
