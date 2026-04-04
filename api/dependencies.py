from api.services.prediction_service import PredictionService

# Singleton instance to avoid reloading models on every request
_prediction_service = None

def get_prediction_service() -> PredictionService:
    """Provides a thread-safe singleton instance of the PredictionService."""
    global _prediction_service
    if _prediction_service is None:
        # Initializing with default paths as defined in the architecture
        _prediction_service = PredictionService(
            model_dir="data/models",
            graph_path="data/graphs/domain_graph.gpickle"
        )
    return _prediction_service
