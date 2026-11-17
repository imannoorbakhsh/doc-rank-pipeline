from doc_rank_pipeline.cli import evaluate_config
from doc_rank_pipeline.config import PipelineConfig, RetrievalConfig
from doc_rank_pipeline.evaluation import _query_metrics
from doc_rank_pipeline.ingestion import chunk_documents
from doc_rank_pipeline.models import SourceDocument
from doc_rank_pipeline.pipeline import DocumentRankPipeline


def test_chunking_preserves_stable_identifiers() -> None:
    chunks = chunk_documents(
        [SourceDocument("doc", "Document", "one two three four five")],
        PipelineConfig.from_dict({"name": "test", "chunking": {"size": 3, "overlap": 1}}).chunking,
    )
    assert [chunk.chunk_id for chunk in chunks] == ["doc:0", "doc:1"]


def test_hybrid_pipeline_returns_matching_document() -> None:
    result = evaluate_config("configs/default.json")
    assert result["overall"]["recall@10"] == 1.0
    assert set(result["by_category"]) == {"command-line", "containers", "filesystem", "iteration", "serialization"}


def test_retrieval_layers_are_switchable() -> None:
    config = PipelineConfig.from_dict(
        {"name": "dense", "retrieval": {"bm25_enabled": False, "dense_enabled": True, "fusion_enabled": False, "reranker_enabled": False}}
    )
    pipeline = DocumentRankPipeline.from_corpus(config, "data/sample_documents.json")
    assert pipeline.bm25 is None
    assert pipeline.dense is not None
    assert pipeline.reranker is None


def test_metrics_reward_first_relevant_result() -> None:
    metrics = _query_metrics(["relevant", "other"], frozenset({"relevant"}), (1, 10))
    assert metrics == {"recall@1": 1.0, "ndcg@1": 1.0, "recall@10": 1.0, "ndcg@10": 1.0, "mrr": 1.0}


def test_recall_measures_fraction_of_all_relevant_chunks() -> None:
    metrics = _query_metrics(["first", "other"], frozenset({"first", "second"}), (1, 10))
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@10"] == 0.5


def test_invalid_retrieval_config_is_rejected() -> None:
    try:
        RetrievalConfig(bm25_enabled=False, dense_enabled=False)
    except ValueError as error:
        assert "retriever" in str(error)
    else:
        raise AssertionError("expected invalid configuration to be rejected")
