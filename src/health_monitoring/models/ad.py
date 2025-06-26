import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis

from src.health_monitoring.models.lstm import LSTMTimeSeriesEncoder
from src.health_monitoring.models.transformer import TransformerTimeSeriesEncoder
from src.health_monitoring.models.mlp import MlpEncoder
from src.health_monitoring.models.fusion import ConcatLateFusionModule, GatedLateFusionModule
from src.health_monitoring.models.gnn import EntityGNN


class AnomalyDetector(nn.Module):
    """Complete pipeline for anomaly detection"""

    def __init__(self,
                 ts_input_dim,
                 static_input_dim,
                 ts_hidden_dim=64,
                 static_hidden_dim=32,
                 fusion_dim=128,
                 gnn_hidden_dim=64,
                 gnn_output_dim=32,
                 ts_encoder_type='lstm',
                 gnn_type='gcn'):
        super().__init__()

        self.ts_encoder = TimeSeriesEncoder(
            ts_input_dim,
            ts_hidden_dim,
            encoder_type=ts_encoder_type
        )

        self.static_encoder = StaticFeatureEncoder(
            static_input_dim,
            static_hidden_dim
        )

        self.fusion = LateFusionModule(
            ts_hidden_dim,
            static_hidden_dim,
            fusion_dim,
            fusion_type='concat'
        )

        self.gnn = EntityGNN(
            fusion_dim,
            gnn_hidden_dim,
            gnn_output_dim,
            gnn_type=gnn_type
        )

        self.scaler = StandardScaler()
        self.anomaly_detector = None

    def forward(self, ts_data, static_data, edge_index, batch=None):
        """
        Args:
            ts_data: (batch_size, seq_len, ts_input_dim)
            static_data: (batch_size, static_input_dim)
            edge_index: (2, num_edges)
            batch: batch indicator for graphs
        Returns:
            (num_nodes, gnn_output_dim) or (num_graphs, gnn_output_dim)
        """
        # Encode time series
        ts_features = self.ts_encoder(ts_data)

        # Encode static features
        static_features = self.static_encoder(static_data)

        # Fuse features
        fused_features = self.fusion(ts_features, static_features)

        # Apply GNN
        gnn_features = self.gnn(fused_features, edge_index, batch)

        return gnn_features

    def fit_anomaly_detector(self, features, method='isolation_forest', contamination=0.05):
        """Fit anomaly detector on normal data"""
        features_np = features.detach().cpu().numpy()
        features_scaled = self.scaler.fit_transform(features_np)

        if method == 'isolation_forest':
            self.anomaly_detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            self.anomaly_detector.fit(features_scaled)

        elif method == 'mahalanobis':
            self.anomaly_detector = {
                'mean': np.mean(features_scaled, axis=0),
                'inv_cov': np.linalg.pinv(np.cov(features_scaled.T)),
                'contamination': contamination
            }

    def detect_anomalies(self, features, method='isolation_forest'):
        """Detect anomalies in features"""
        features_np = features.detach().cpu().numpy()
        features_scaled = self.scaler.transform(features_np)

        if method == 'isolation_forest':
            anomaly_scores = self.anomaly_detector.decision_function(features_scaled)
            anomalies = self.anomaly_detector.predict(features_scaled) == -1
            return anomalies, anomaly_scores

        elif method == 'mahalanobis':
            mean = self.anomaly_detector['mean']
            inv_cov = self.anomaly_detector['inv_cov']
            contamination = self.anomaly_detector['contamination']

            mahal_distances = [
                mahalanobis(x, mean, inv_cov) for x in features_scaled
            ]
            threshold = np.percentile(mahal_distances, (1 - contamination) * 100)
            anomalies = np.array(mahal_distances) > threshold

            return anomalies, np.array(mahal_distances)


# Example usage
def create_sample_data(num_entities=100, seq_len=50, ts_dim=5, static_dim=10):
    """Create sample data for demonstration"""

    # Time series data
    ts_data = torch.randn(num_entities, seq_len, ts_dim)

    # Static features
    static_data = torch.randn(num_entities, static_dim)

    # Create a simple graph (ring topology for demonstration)
    edge_index = []
    for i in range(num_entities):
        edge_index.append([i, (i + 1) % num_entities])
        edge_index.append([(i + 1) % num_entities, i])

    edge_index = torch.tensor(edge_index).t().contiguous()

    return ts_data, static_data, edge_index


# Example usage
if __name__ == "__main__":
    # Create sample data
    ts_data, static_data, edge_index = create_sample_data(
        num_entities=100,
        seq_len=50,
        ts_dim=5,
        static_dim=10
    )

    # Initialize model
    model = AnomalyDetector(
        ts_input_dim=5,
        static_input_dim=10,
        ts_hidden_dim=64,
        static_hidden_dim=32,
        fusion_dim=96,
        gnn_hidden_dim=64,
        gnn_output_dim=32,
        ts_encoder_type='lstm',  # or 'transformer'
        gnn_type='gcn'  # or 'gat'
    )

    # Forward pass
    with torch.no_grad():
        features = model(ts_data, static_data, edge_index)
        print(f"Output features shape: {features.shape}")

        # Fit anomaly detector on normal data
        model.fit_anomaly_detector(features, method='isolation_forest')

        # Detect anomalies
        anomalies, scores = model.detect_anomalies(features, method='isolation_forest')
        print(f"Detected {anomalies.sum()} anomalies out of {len(anomalies)} entities")