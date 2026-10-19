'use client';

import { useState } from 'react';
import toast from 'react-hot-toast';
import { errorMessage, ExerciseBlank, GradeResult, practiceApi } from '@/lib/api';
import { LatexText } from './LatexText';

const VERDICT: Record<GradeResult['verdict'], [string, string]> = {
  correct: ['Correct', 'bg-emerald-100 text-emerald-800'],
  partially: ['Almost', 'bg-amber-100 text-amber-800'],
  incorrect: ['Not yet', 'bg-rose-100 text-rose-800'],
  missing: ['Empty', 'bg-gray-100 text-gray-600'],
};

/** Rebuild the lines that implement cited paper details; graded against the reference and the excerpt. */
export function ExercisePanel({ runId, evidence, executionEnabled }: {
  runId: string; evidence: { id: string; text: string; kind?: string }[]; executionEnabled: boolean;
}) {
  const [exercise, setExercise] = useState<{ template: string; blanks: ExerciseBlank[] } | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [results, setResults] = useState<Record<string, GradeResult>>({});
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const [execution, setExecution] = useState<{ passed?: boolean; stdout?: string; stderr?: string; error?: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const start = async () => {
    setBusy(true);
    try { setExercise(await practiceApi.exercise(runId)); } catch (error) { toast.error(errorMessage(error)); } finally { setBusy(false); }
  };
  const check = async () => {
    setBusy(true);
    try {
      const graded = await practiceApi.gradeExercise(runId, answers);
      setResults(Object.fromEntries(graded.results.map((item) => [item.id, item])));
      setExecution(graded.execution);
    } catch (error) { toast.error(errorMessage(error)); } finally { setBusy(false); }
  };

  if (!exercise) {
    return <div className="rounded-xl border border-dashed border-sky-300 p-3 text-sm">
      <p className="font-semibold">Practice: rebuild the key lines</p>
      <p className="text-xs text-gray-600">The lines that implement cited paper details are blanked out. Fill them in from the excerpts; answers are checked against the paper{executionEnabled ? ' and by running the program' : ''}.</p>
      <button type="button" onClick={start} disabled={busy} className="btn-primary btn-sm mt-2">{busy ? 'Preparing…' : 'Start exercise'}</button>
    </div>;
  }
  const blanksByLine = new Map(exercise.blanks.map((blank) => [blank.line, blank]));
  const byId = new Map(evidence.map((item) => [item.id, item]));
  return <div className="space-y-3 rounded-xl border border-sky-200 p-3 text-sm">
    <p className="font-semibold">Practice: rebuild the key lines</p>
    <div className="max-h-[28rem] overflow-auto rounded-lg bg-gray-50 py-2 font-mono text-xs">
      {exercise.template.split('\n').map((line, index) => {
        const blank = blanksByLine.get(index + 1);
        if (!blank) {
          return <div key={index} className="flex gap-3 px-2"><span className="w-8 shrink-0 select-none text-right text-gray-400">{index + 1}</span><span className="whitespace-pre-wrap break-all text-gray-700">{line || ' '}</span></div>;
        }
        const result = results[blank.id];
        const indent = line.slice(0, line.length - line.trimStart().length);
        return <div key={index} className="space-y-1 bg-sky-50 px-2 py-1">
          <div className="flex items-center gap-3">
            <span className="w-8 shrink-0 select-none text-right text-gray-400">{index + 1}</span>
            <span className="whitespace-pre text-gray-800">{indent}{blank.prefix ? `${blank.prefix} ` : ''}</span>
            <input aria-label={`Blank ${blank.id}`} value={answers[blank.id] || ''} onChange={(event) => setAnswers({ ...answers, [blank.id]: event.target.value })}
              maxLength={300} className="min-w-0 flex-1 rounded border border-sky-300 bg-white px-1.5 py-0.5 font-mono text-xs" placeholder={`blank ${blank.id}`} />
            {result && <span className={`shrink-0 rounded px-1.5 py-0.5 font-sans text-[10px] font-semibold ${VERDICT[result.verdict][1]}`}>{VERDICT[result.verdict][0]}</span>}
          </div>
          <div className="pl-11 font-sans text-[11px] text-gray-600">
            <span className="font-semibold">{blank.evidence_id}:</span> <LatexText text={blank.hint} />
            {byId.get(blank.evidence_id) && <details className="mt-0.5"><summary className="cursor-pointer text-sky-700">Show excerpt</summary>
              {byId.get(blank.evidence_id)!.kind === 'repo_code'
                ? <pre className="mt-1 max-h-40 overflow-auto rounded bg-gray-900 p-2 text-[11px] text-gray-100">{byId.get(blank.evidence_id)!.text}</pre>
                : <p className="mt-1"><LatexText text={byId.get(blank.evidence_id)!.text} /></p>}</details>}
            {result && <p className="mt-0.5 text-gray-700">{result.feedback}{' '}
              {result.verdict !== 'correct' && (revealed[blank.id]
                ? <code className="rounded bg-white px-1">{result.answer}</code>
                : <button type="button" onClick={() => setRevealed({ ...revealed, [blank.id]: true })} className="text-sky-700 underline">Show answer</button>)}</p>}
          </div>
        </div>;
      })}
    </div>
    <div className="flex items-center gap-3">
      <button type="button" onClick={check} disabled={busy || !Object.values(answers).some((value) => value.trim())} className="btn-primary btn-sm">{busy ? 'Checking…' : 'Check my answers'}</button>
      {!!Object.keys(results).length && <span className="text-xs text-gray-600">{Object.values(results).filter((r) => r.verdict === 'correct').length}/{exercise.blanks.length} correct</span>}
    </div>
    {execution && <p className={`text-xs ${execution.passed ? 'text-emerald-700' : 'text-amber-700'}`}>{execution.passed ? 'Your completed program ran and its self-check passed.' : `Your completed program did not pass: ${(execution.stderr || execution.error || '').slice(0, 300)}`}</p>}
  </div>;
}
