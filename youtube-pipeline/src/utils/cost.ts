export const ESTIMATED_PIPELINE_COST_USD = {
  llm: 0.05,
  voice: 0.12,
  video: 0.5,
};

export function estimatePipelineCostUsd(): number {
  return (
    ESTIMATED_PIPELINE_COST_USD.llm +
    ESTIMATED_PIPELINE_COST_USD.voice +
    ESTIMATED_PIPELINE_COST_USD.video
  );
}
