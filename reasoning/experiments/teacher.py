"""Retry missing teacher exports once with a larger budget, preserving original data."""
import argparse
import fcntl
import getpass
import json
import os
import time
from reasoning.common import ROOT, Journal, digest, read_jsonl, write_json, code_hashes, api_credentials
from datetime import datetime, timezone
from pathlib import Path
from reasoning.datasets import problem_id
from reasoning.evaluation import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-key', action='store_true', help='Read API key privately from this terminal')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/v2/teacher_repair')
    args = parser.parse_args()
    source = ROOT / 'outputs/archive/data/math500/gpt51_teacher'
    output = args.output
    metadata = json.loads((source / 'attempts.jsonl.meta.json').read_text())
    key, base_url = api_credentials(metadata['base_url'])
    rows = read_jsonl(ROOT / 'data/original.jsonl')
    export_path = ROOT/'data/positive.jsonl'
    original = read_jsonl(export_path)
    exported = {r['id'] for r in original}
    missing = [{**r, 'id': problem_id(r['problem'])} for r in rows
               if problem_id(r['problem']) not in exported]
    config = dict(model=metadata['model'], base_url=base_url,
                  reasoning_effort=metadata['reasoning_effort'], max_output_tokens=65536,
                  instructions=metadata['instructions'], suffix=metadata['suffix'],
                  original_export_sha256=digest(export_path),
                  data_sha256=digest(ROOT / 'data/original.jsonl'),
                  code_sha256=code_hashes(), problem_ids=[r['id'] for r in missing],
                  max_new_attempts_per_problem=1)
    if args.dry_run:
        print(json.dumps(dict(missing=len(missing), problem_ids=config['problem_ids'],
                              max_output_tokens=config['max_output_tokens'],
                              api_key_present=bool(key))))
        return
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'repair.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    journal = Journal(output / 'attempts.jsonl', config)
    pending = [r for r in missing if r['id'] not in journal.ids]
    if pending and not key and args.prompt_key:
        key = getpass.getpass('API key (hidden; not saved): ')
    if pending and not key:
        write_json(output / 'status.json', {'state': 'blocked_missing_api_key', 'pending': len(pending)})
        raise SystemExit('API key missing. Configure models.local.json or use --prompt-key.')
    if pending:
        from reasoning.models import TeacherClient
        client = TeacherClient(api_key=key, base_url=config['base_url'], timeout=900, max_retries=0)
        for row in pending:
            write_json(output / 'status.json', {'state': 'running', 'problem_id': row['id'], 'pid': os.getpid()})
            start = time.monotonic()
            try:
                response = client.generate(
                    model=config['model'], instructions=config['instructions'],
                    input=row['problem'] + config['suffix'],
                    reasoning={'effort': config['reasoning_effort']},
                    max_output_tokens=config['max_output_tokens'])
            except Exception as exc:
                # Do not persist exception bodies: gateways may echo credentials.
                write_json(output / 'status.json', {'state': 'api_failed', 'error_type': type(exc).__name__})
                raise SystemExit(f'API call failed: {type(exc).__name__}') from None
            text = response.output_text
            completed = response.status == 'completed'
            journal.append({**row, 'completion': text, **score(text, row['answer']),
                            'truncated': not completed, 'response_status': response.status,
                            'response_id': response.id,
                            'incomplete_details': response.incomplete_details.model_dump() if response.incomplete_details else None,
                            'usage': response.usage.model_dump() if response.usage else None,
                            'seconds': time.monotonic() - start,
                            'created_at': datetime.now(timezone.utc).isoformat(),
                            'attempt': 2, 'repair_config': config})
    merged = list(original)
    for row in journal.rows:
        if row['truncated'] or row['parsed_answer'] is None:
            continue
        merged.append(dict(id=row['id'], problem=row['problem'], answer=row['answer'],
                           solution=row['completion'], subject=row['subject'], level=row['level'],
                           source_split='MATH500/test', teacher_model=config['model'],
                           trace_id=row['id'] + ':repair:2', label='positive',
                           label_basis='steering_source_group_teacher', correct=row['correct'],
                           parsed_answer=row['parsed_answer'], score_status=row['score_status'],
                           validation='final_answer_only', reasoning_verified=False,
                           provenance=str(output/'attempts.jsonl'),
                           independent_test_if_used_for_training=False))
    target = output / 'positive.merged.jsonl'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in merged))
    tmp.replace(target)
    missing_ids = sorted({problem_id(r['problem']) for r in rows} - {r['id'] for r in merged})
    summary = dict(state='completed' if not missing_ids else 'finished_with_missing_traces',
                   exported_problems=len(merged), correct=sum(r['correct'] for r in merged),
                   missing_export_ids=missing_ids, original_data_preserved=True,
                   merged_export=str(target))
    write_json(output / 'summary.json', summary)
    write_json(output / 'status.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
