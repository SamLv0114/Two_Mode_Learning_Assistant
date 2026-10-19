'use client';

import { useState } from 'react';
import { PracticeRun } from '@/lib/api';
import { LatexText } from './LatexText';

/** Human-in-the-loop step: the learner approves or edits the plan before any code is written. */
export function PlanReview({ run, busy, onSubmit }: {
  run: PracticeRun; busy: boolean;
  onSubmit: (plan: { choices: Record<string, string>; simplifications: string[]; instructions: string }) => void;
}) {
  const manifest = run.result.manifest;
  const choices = manifest?.open_choices || [];
  const [answers, setAnswers] = useState<Record<string, string>>(
    Object.fromEntries(choices.map((choice) => [choice.id, choice.recommended])));
  const [custom, setCustom] = useState<Record<string, string>>({});
  const [simplifications, setSimplifications] = useState<string[]>(manifest?.simplifications || []);
  const [newSimplification, setNewSimplification] = useState('');
  const [instructions, setInstructions] = useState('');
  if (!manifest) return null;

  const submit = () => onSubmit({
    choices: Object.fromEntries(choices.map((choice) => [choice.id,
      answers[choice.id] === '__custom__' ? (custom[choice.id] || '').trim() || choice.recommended : answers[choice.id]])),
    simplifications, instructions: instructions.trim(),
  });

  return <div className="space-y-4 rounded-2xl border-2 border-violet-200 bg-violet-50/40 p-5 text-sm">
    <div>
      <h4 className="font-semibold text-violet-950">Review the plan before the agent writes code</h4>
      <p className="text-xs text-violet-800">The agent read the paper and will implement <strong>{manifest.method}</strong>. Check what it will follow, decide what the paper leaves open, and adjust the scope.</p>
    </div>
    {!!manifest.known_details?.length && <div>
      <p className="section-label">From the paper</p>
      <ul className="mt-1 list-disc space-y-1 pl-5">{manifest.known_details.map((item, index) => <li key={index}>
        <LatexText text={item.detail} /> <span className="text-xs text-gray-500">({item.evidence_ids.join(', ')})</span>
        {item.quote && <span className="block border-l-2 border-violet-300 pl-2 text-xs italic text-violet-900">“<LatexText text={item.quote} />”</span>}
      </li>)}</ul>
    </div>}
    {!!choices.length && <div className="space-y-3">
      <p className="section-label">The paper leaves these open — your call</p>
      {choices.map((choice) => <fieldset key={choice.id} className="rounded-xl border border-violet-200 bg-white p-3">
        <legend className="px-1 font-medium"><LatexText text={choice.question} /></legend>
        <div className="mt-1 flex flex-wrap gap-2">
          {choice.options.map((option) => <label key={option} className={`flex cursor-pointer items-center gap-1.5 rounded-lg border px-2 py-1 ${answers[choice.id] === option ? 'border-violet-500 bg-violet-50' : 'border-gray-200'}`}>
            <input type="radio" name={choice.id} checked={answers[choice.id] === option} onChange={() => setAnswers({ ...answers, [choice.id]: option })} />
            <LatexText text={option} />{option === choice.recommended && <span className="rounded bg-violet-100 px-1 text-[10px] font-semibold text-violet-700">suggested</span>}
          </label>)}
          <label className={`flex items-center gap-1.5 rounded-lg border px-2 py-1 ${answers[choice.id] === '__custom__' ? 'border-violet-500 bg-violet-50' : 'border-gray-200'}`}>
            <input type="radio" name={choice.id} checked={answers[choice.id] === '__custom__'} onChange={() => setAnswers({ ...answers, [choice.id]: '__custom__' })} />
            <input value={custom[choice.id] || ''} onFocus={() => setAnswers({ ...answers, [choice.id]: '__custom__' })} onChange={(event) => setCustom({ ...custom, [choice.id]: event.target.value })} maxLength={200} placeholder="Something else…" className="w-40 border-0 bg-transparent p-0 text-sm focus:ring-0" />
          </label>
        </div>
      </fieldset>)}
    </div>}
    <div>
      <p className="section-label">Simplified for teaching</p>
      <ul className="mt-1 space-y-1">{simplifications.map((item, index) => <li key={index} className="flex items-start justify-between gap-2 rounded-lg bg-white px-2 py-1">
        <span><LatexText text={item} /></span>
        <button type="button" onClick={() => setSimplifications(simplifications.filter((_, i) => i !== index))} className="text-xs text-gray-400 hover:text-rose-600" aria-label="Remove simplification">remove</button>
      </li>)}</ul>
      {simplifications.length < 8 && <div className="mt-1 flex gap-2">
        <input value={newSimplification} onChange={(event) => setNewSimplification(event.target.value)} maxLength={200} placeholder="Add a simplification, e.g. use a 2-layer MLP" className="input-field flex-1 !py-1.5 text-xs" />
        <button type="button" disabled={!newSimplification.trim()} onClick={() => { setSimplifications([...simplifications, newSimplification.trim()]); setNewSimplification(''); }} className="btn-secondary btn-sm">Add</button>
      </div>}
    </div>
    <div>
      <label htmlFor="plan-instructions" className="section-label">Anything else? (optional)</label>
      <textarea id="plan-instructions" value={instructions} onChange={(event) => setInstructions(event.target.value)} maxLength={500} placeholder="e.g. Keep it under 60 lines and print the loss at every step" className="input-field mt-1 h-16 text-sm" />
      <p className="text-[11px] text-gray-500">Instructions can change scope and style, not what the paper states.</p>
    </div>
    <button type="button" onClick={submit} disabled={busy} className="btn-primary">{busy ? 'Sending…' : 'Approve and write the code'}</button>
  </div>;
}
