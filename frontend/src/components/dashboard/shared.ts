import { HFPaper } from '@/lib/api';

/** Where a paper or repository was opened from, shown so the learner can find their way back. */
export type Origin = 'Daily Feed' | "What's Hot" | 'Saved' | 'Coding Agent';

/**
 * A paper handed to Agent Chat. Repositories are deliberately not: Agent Chat has no tool that
 * reads code, so questions about a repository go to the Coding Agent's repository agent, whose
 * answers cite file:line ranges it actually read.
 */
export type ChatContext = {
  kind: 'paper'; title: string; abstract: string | null; arxivId: string; arxivUrl: string; origin?: Origin;
};
export function arxivIdFromHotPaper(paper: HFPaper): string | null {
  const match = paper.arxiv_url?.match(/arxiv\.org\/(?:abs|pdf)\/(\d{4}\.\d{4,5}(?:v\d+)?)/i);
  return match?.[1] || (/^\d{4}\.\d{4,5}(?:v\d+)?$/.test(paper.id) ? paper.id : null);
}

export const AVAILABLE_AREAS = ['DL', 'LLM', 'Agent', 'ML', 'AI', 'NLP', 'CV'];
export const EXAMPLE_INTERESTS = [
  'machine learning and deep learning',
  'natural language processing and transformers',
  'computer vision and image recognition',
  'reinforcement learning and agents',
  'generative AI and large language models',
];
