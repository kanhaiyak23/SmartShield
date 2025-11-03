"""Data preprocessing for UNSW-NB15 and network traffic."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger('SmartShield')

class DataPreprocessor:
    """Preprocess network traffic data for ML models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_cols = None
        self.categorical_cols = None
        
    def encode_categorical(self, df, columns):
        """Encode categorical features."""
        encoded_df = df.copy()
        
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                encoded_df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                unique_vals = df[col].astype(str).unique()
                known_vals = set(self.label_encoders[col].classes_)
                unknown_vals = set(unique_vals) - known_vals
                
                if unknown_vals:
                    logger.warning(f"Found unknown categories in {col}: {unknown_vals}")
                    # Map unknown to 'unknown'
                    encoded_df[col] = encoded_df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in known_vals else -1
                    )
                else:
                    encoded_df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return encoded_df
    
    def fit(self, df):
        """Fit preprocessor on training data (identify columns, fit encoders/scalers)."""
        logger.info("Fitting preprocessor on training data...")
        
        # Remove unnecessary columns
        cols_to_drop = ['id', 'sttl', 'dttl']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # Identify categorical and numeric columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove label columns from features
        if 'label' in self.categorical_cols:
            self.categorical_cols.remove('label')
        if 'attack_cat' in self.categorical_cols:
            self.categorical_cols.remove('attack_cat')
        if 'label' in self.numeric_cols:
            self.numeric_cols.remove('label')
        if 'attack_cat' in self.numeric_cols:
            self.numeric_cols.remove('attack_cat')
        
        logger.info(f"Categorical columns: {self.categorical_cols}")
        logger.info(f"Numeric columns: {len(self.numeric_cols)} (total: {len(self.numeric_cols) + len(self.categorical_cols)})")
        
        # Prepare training data for fitting
        df_work = df.copy()
        
        # Drop label columns
        cols_to_drop_labels = ['label', 'attack_cat']
        cols_to_drop_labels = [col for col in cols_to_drop_labels if col in df_work.columns]
        if cols_to_drop_labels:
            df_work = df_work.drop(columns=cols_to_drop_labels)
        
        # Handle missing values
        df_work = df_work.fillna(0)
        
        # FIT categorical encoders (only on training data)
        if self.categorical_cols:
            categorical_cols_in_df = [col for col in self.categorical_cols if col in df_work.columns]
            for col in categorical_cols_in_df:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit encoder on training data
                    self.label_encoders[col].fit(df_work[col].astype(str))
        
        # FIT numeric scaler (only on training data)
        self.numeric_cols = [col for col in self.numeric_cols if col in df_work.columns]
        if self.numeric_cols:
            self.scaler.fit(df_work[self.numeric_cols])
        
        logger.info("Preprocessor fitted successfully")
        return self
    
    def transform(self, df):
        """Transform data using fitted preprocessor."""
        logger.info("Transforming data with fitted preprocessor...")
        
        if self.numeric_cols is None or self.categorical_cols is None:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        # Remove unnecessary columns
        cols_to_drop = ['id', 'sttl', 'dttl']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df_work = df.drop(columns=cols_to_drop)
        
        # Drop label columns
        cols_to_drop_labels = ['label', 'attack_cat']
        cols_to_drop_labels = [col for col in cols_to_drop_labels if col in df_work.columns]
        if cols_to_drop_labels:
            df_work = df_work.drop(columns=cols_to_drop_labels)
        
        # Handle missing values
        df_work = df_work.fillna(0)
        
        # TRANSFORM categorical features (using fitted encoders)
        if self.categorical_cols:
            categorical_cols_in_df = [col for col in self.categorical_cols if col in df_work.columns]
            for col in categorical_cols_in_df:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_vals = df_work[col].astype(str).unique()
                    known_vals = set(self.label_encoders[col].classes_)
                    unknown_vals = set(unique_vals) - known_vals
                    
                    if unknown_vals:
                        logger.warning(f"Found unknown categories in {col}: {unknown_vals}")
                        df_work[col] = df_work[col].apply(
                            lambda x: self.label_encoders[col].transform([x])[0] 
                            if x in known_vals else -1
                        )
                    else:
                        df_work[col] = self.label_encoders[col].transform(df_work[col].astype(str))
        
        # TRANSFORM numeric features (using fitted scaler)
        numeric_cols_in_df = [col for col in self.numeric_cols if col in df_work.columns]
        if numeric_cols_in_df:
            df_work[numeric_cols_in_df] = self.scaler.transform(df_work[numeric_cols_in_df])
        
        logger.info(f"Transformed dataset shape: {df_work.shape}")
        return df_work
    
    def preprocess_unsw_nb15(self, df):
        """Preprocess UNSW-NB15 dataset (legacy method - fits and transforms in one step).
        DEPRECATED: Use fit() and transform() separately to avoid data leakage.
        """
        logger.warning("Using deprecated preprocess_unsw_nb15(). Use fit() and transform() separately to avoid data leakage.")
        # For backward compatibility, fit and transform in one step
        self.fit(df)
        return self.transform(df)
    
    def map_packet_to_unsw_features(self, packet_features):
        """Map single packet features to UNSW-NB15 feature space."""
        # UNSW-NB15 features that need to be created/approximated from single packet
        unsw_features = {}
        
        # Extract basic packet info
        packet_length = packet_features.get('packet_length', 0)
        protocol = packet_features.get('protocol', 'other').lower()
        src_port = packet_features.get('src_port', 0)
        dst_port = packet_features.get('dst_port', 0)
        
        # Flow-based features (approximated for single packet)
        # dur: duration of flow (single packet = 0)
        unsw_features['dur'] = 0.0
        
        # proto: protocol (tcp, udp, icmp, etc.) - already have
        # Map protocol to UNSW-NB15 format
        protocol_map = {
            'tcp': 'tcp',
            'udp': 'udp',
            'icmp': 'icmp',
            'arp': 'arp',
            'other': '-'
        }
        unsw_features['proto'] = protocol_map.get(protocol, '-')
        
        # service: derived from port (common services)
        service = '-'
        if dst_port:
            if dst_port == 80 or dst_port == 443:
                service = 'http'
            elif dst_port == 21:
                service = 'ftp'
            elif dst_port == 22:
                service = 'ssh'
            elif dst_port == 25:
                service = 'smtp'
            elif dst_port == 53:
                service = 'dns'
            elif dst_port == 5353:
                service = 'dns'  # mDNS
        unsw_features['service'] = service
        
        # state: connection state (approximate based on protocol)
        if protocol == 'tcp':
            state = 'FIN'  # Default, could be improved
        elif protocol == 'udp':
            state = 'INT'  # INT = intermediate
        else:
            state = 'INT'
        unsw_features['state'] = state
        
        # spkts: source packets (1 for single packet)
        unsw_features['spkts'] = 1
        
        # dpkts: destination packets (1 for single packet)
        unsw_features['dpkts'] = 1
        
        # sbytes: source bytes (approximate - use packet_length)
        unsw_features['sbytes'] = packet_length
        
        # dbytes: destination bytes (approximate - use packet_length)
        unsw_features['dbytes'] = packet_length
        
        # rate: packet rate (approximate as packet_length / 1 second = packet_length)
        unsw_features['rate'] = float(packet_length)
        
        # sload: source load (bytes/sec, approximate)
        unsw_features['sload'] = float(packet_length)
        
        # dload: destination load (bytes/sec, approximate)
        unsw_features['dload'] = float(packet_length)
        
        # sloss: source packet loss (0 for single packet)
        unsw_features['sloss'] = 0
        
        # dloss: destination packet loss (0 for single packet)
        unsw_features['dloss'] = 0
        
        # sinpkt: source inter-packet arrival time (approximate)
        unsw_features['sinpkt'] = 0.0
        
        # dinpkt: destination inter-packet arrival time (approximate)
        unsw_features['dinpkt'] = 0.0
        
        # sjit: source jitter (0 for single packet)
        unsw_features['sjit'] = 0.0
        
        # djit: destination jitter (0 for single packet)
        unsw_features['djit'] = 0.0
        
        # swin: source TCP window size
        unsw_features['swin'] = packet_features.get('tcp_window', 0)
        
        # stcpb: source TCP base sequence number
        unsw_features['stcpb'] = packet_features.get('tcp_seq', 0)
        
        # dtcpb: destination TCP base sequence number
        unsw_features['dtcpb'] = packet_features.get('tcp_ack', 0)
        
        # dwin: destination TCP window size
        unsw_features['dwin'] = packet_features.get('tcp_window', 0)
        
        # tcprtt: TCP round trip time (0 for single packet)
        unsw_features['tcprtt'] = 0.0
        
        # synack: TCP SYN-ACK time (0 for single packet)
        unsw_features['synack'] = 0.0
        
        # ackdat: TCP ACK data time (0 for single packet)
        unsw_features['ackdat'] = 0.0
        
        # smean: mean source packet size
        unsw_features['smean'] = float(packet_length)
        
        # dmean: mean destination packet size
        unsw_features['dmean'] = float(packet_length)
        
        # trans_depth: transaction depth (1 for single packet)
        unsw_features['trans_depth'] = 1
        
        # response_body_len: response body length (0 for single packet)
        unsw_features['response_body_len'] = 0
        
        # Connection tracking features (approximated)
        # ct_srv_src: connections to same service from same source
        unsw_features['ct_srv_src'] = 1
        
        # ct_state_ttl: connection state + TTL
        ttl = packet_features.get('ip_ttl', 64)
        unsw_features['ct_state_ttl'] = 1  # Simplified
        
        # ct_dst_ltm: connections to same destination in last time window
        unsw_features['ct_dst_ltm'] = 1
        
        # ct_src_dport_ltm: connections from same source to same destination port
        unsw_features['ct_src_dport_ltm'] = 1
        
        # ct_dst_sport_ltm: connections to same destination from same source port
        unsw_features['ct_dst_sport_ltm'] = 1
        
        # ct_dst_src_ltm: connections to same destination from same source
        unsw_features['ct_dst_src_ltm'] = 1
        
        # is_ftp_login: is FTP login (0)
        unsw_features['is_ftp_login'] = 0
        
        # ct_ftp_cmd: FTP command count (0)
        unsw_features['ct_ftp_cmd'] = 0
        
        # ct_flw_http_mthd: HTTP method count (0)
        unsw_features['ct_flw_http_mthd'] = 0
        
        # ct_src_ltm: connections from same source in last time window
        unsw_features['ct_src_ltm'] = 1
        
        # ct_srv_dst: connections to same service to same destination
        unsw_features['ct_srv_dst'] = 1
        
        # is_sm_ips_ports: same source IP and port (0 or 1)
        unsw_features['is_sm_ips_ports'] = 0
        
        return unsw_features
    
    def preprocess_packet_features(self, features):
        """Preprocess features from captured packets."""
        # First map packet features to UNSW-NB15 feature space
        if isinstance(features, dict):
            # Map single packet to UNSW-NB15 features
            unsw_features = self.map_packet_to_unsw_features(features)
            df = pd.DataFrame([unsw_features])
        else:
            # If already a DataFrame, map each row
            df = features.copy()
            if len(df) > 0:
                mapped_features = []
                for idx, row in df.iterrows():
                    mapped = self.map_packet_to_unsw_features(row.to_dict())
                    mapped_features.append(mapped)
                df = pd.DataFrame(mapped_features)
        
        # Ensure we have all required columns
        if self.numeric_cols is None or self.categorical_cols is None:
            logger.error("Preprocessor not properly initialized. Models may not be trained yet.")
            # Return empty dataframe with correct structure
            return df
        
        # Fill missing columns with 0
        all_cols = set(self.numeric_cols) | set(self.categorical_cols)
        for col in all_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Encode categorical features
        if self.categorical_cols:
            categorical_cols_in_df = [col for col in self.categorical_cols if col in df.columns]
            if categorical_cols_in_df:
                df = self.encode_categorical(df, categorical_cols_in_df)
        
        # Normalize numeric features
        numeric_cols_in_df = [col for col in self.numeric_cols if col in df.columns]
        if numeric_cols_in_df:
            try:
                df[numeric_cols_in_df] = self.scaler.transform(df[numeric_cols_in_df])
            except Exception as e:
                logger.error(f"Error normalizing features: {e}")
                # Fill with zeros if normalization fails
                df[numeric_cols_in_df] = 0
        
        # Ensure column order matches training exactly
        feature_cols = self.numeric_cols + self.categorical_cols
        logger.debug(f"Expected features: {len(feature_cols)} (numeric: {len(self.numeric_cols)}, categorical: {len(self.categorical_cols)})")
        logger.debug(f"Features in DataFrame before alignment: {len(df.columns)}")
        
        # Reorder columns to match training order
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing features in packet data: {missing_cols}")
            for col in missing_cols:
                df[col] = 0
        
        # Select only the columns we need, in the right order
        # Ensure all feature_cols exist in df
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_cols]
        
        # Log feature mismatch if any
        if len(df.columns) != len(feature_cols):
            missing = set(feature_cols) - set(df.columns)
            extra = set(df.columns) - set(feature_cols)
            logger.error(
                f"Feature mismatch: Expected {len(feature_cols)} features, got {len(df.columns)}. "
                f"Missing: {missing}, Extra: {extra}"
            )
            logger.error(f"Expected columns: {feature_cols}")
            logger.error(f"Got columns: {list(df.columns)}")
        
        logger.debug(f"Final feature count: {len(df.columns)} (expected: {len(feature_cols)})")
        return df
    
    def prepare_train_test(self, df, target_col='label', test_size=0.2, random_state=42):
        """Prepare train/test split."""
        X = df.drop(columns=[col for col in [target_col, 'attack_cat'] if col in df.columns])
        y = df[target_col] if target_col in df.columns else None
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            return X_train, X_test, None, None
    
    def get_feature_names(self):
        """Get feature names in correct order."""
        return self.numeric_cols + self.categorical_cols

