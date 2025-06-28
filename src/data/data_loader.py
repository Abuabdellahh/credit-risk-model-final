import pandas as pd
from typing import Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw transaction data from CSV file.
        """
        logger.info(f"Loading data from {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

class FeatureEngineering:
    def __init__(self):
        self.snapshot_date = datetime.now()
        
    def calculate_rfm_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Recency, Frequency, and Monetary metrics for each customer.
        """
        logger.info("Calculating RFM metrics")
        
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Calculate Recency
        recency = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
        recency['Recency'] = (self.snapshot_date - recency['TransactionStartTime']).dt.days
        
        # Calculate Frequency
        frequency = df.groupby('CustomerId')['TransactionId'].nunique().reset_index()
        frequency.columns = ['CustomerId', 'Frequency']
        
        # Calculate Monetary value
        monetary = df.groupby('CustomerId')['Value'].sum().reset_index()
        monetary.columns = ['CustomerId', 'Monetary']
        
        # Merge all metrics
        rfm = recency.merge(frequency, on='CustomerId').merge(monetary, on='CustomerId')
        rfm = rfm[['CustomerId', 'Recency', 'Frequency', 'Monetary']]
        
        logger.info(f"Calculated RFM metrics for {len(rfm)} customers")
        return rfm

    def create_proxy_target(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Create high-risk proxy target using K-means clustering on RFM metrics.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        logger.info("Creating high-risk proxy target")
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(scaled_features)
        
        # Identify high-risk cluster (typically low frequency and low monetary)
        cluster_stats = rfm.groupby('Cluster').agg({
            'Frequency': 'mean',
            'Monetary': 'mean'
        })
        
        # Find the cluster with lowest frequency and monetary value
        high_risk_cluster = cluster_stats[
            (cluster_stats['Frequency'] == cluster_stats['Frequency'].min()) &
            (cluster_stats['Monetary'] == cluster_stats['Monetary'].min())
        ].index[0]
        
        # Create binary target variable
        rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
        
        logger.info(f"Created high-risk proxy target. High-risk cluster: {high_risk_cluster}")
        return rfm
