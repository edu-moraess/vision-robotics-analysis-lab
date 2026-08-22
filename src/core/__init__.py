"""Core pipeline and timing utilities."""
__all__ = ["AnalysisPipeline", "AnalysisResult", "LatencyBreakdown"]

def __getattr__(name: str):
    if name in ("AnalysisPipeline", "AnalysisResult"):
        from .pipeline import AnalysisPipeline, AnalysisResult
        return {"AnalysisPipeline": AnalysisPipeline, "AnalysisResult": AnalysisResult}[name]
    if name == "LatencyBreakdown":
        from .timing import LatencyBreakdown
        return LatencyBreakdown
    raise AttributeError(name)
