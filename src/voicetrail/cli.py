'''CLI for the VoiceTrail session-level asynchronous pipeline.'''
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
from .config import Settings
from .data import load_dataset
from .scheduler import Limits, VoiceTrailPipeline

def main() -> None:
    parser = argparse.ArgumentParser(
        description='VoiceTrail: speech → memory → retrieval → answer')
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Unified dataset directory')
    parser.add_argument('--output', type=Path,
                        help='Private output directory (do not commit)')
    parser.add_argument('--users',
                        help='Optional comma-separated user IDs')
    parser.add_argument('--sessions',
                        help='Optional comma-separated session IDs')
    parser.add_argument('--question-limit', type=int)
    parser.add_argument('--asr-workers', type=int, default=1)
    parser.add_argument('--writer-workers', type=int, default=2)
    parser.add_argument('--index-workers', type=int, default=1)
    parser.add_argument('--reader-workers', type=int, default=1)
    parser.add_argument('--queue-capacity', type=int, default=8)
    parser.add_argument('--preflight', action='store_true',
                        help='Validate input files without loading models')
    args = parser.parse_args()

    def values(v):
        return set(x.strip() for x in v.split(',') if x.strip()) if v else None
    sessions, questions = load_dataset(
        args.dataset, users=values(args.users),
        sessions=values(args.sessions),
        question_limit=args.question_limit)
    info = {
        'sessions': len(sessions),
        'audio_turns': sum(len(s.turns) for s in sessions),
        'questions': len(questions),
        'users': len({s.user_id for s in sessions}),
    }
    if args.preflight:
        print(json.dumps(info, indent=2))
        return
    if args.output is None:
        parser.error('--output is required unless --preflight is specified')
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    settings = Settings.from_env()
    engine = __import__('voicetrail.adapters', fromlist=['LiveStages']).LiveStages(settings)
    limits = Limits(
        asr=args.asr_workers,
        writer=args.writer_workers,
        index=args.index_workers,
        reader=args.reader_workers,
        queue_capacity=args.queue_capacity)
    runner = VoiceTrailPipeline(
        asr=engine.asr, writer=engine.writer,
        indexer=engine.indexer, answer=engine.answer,
        out_dir=args.output, limits=limits)
    report = asyncio.run(runner.run(sessions, questions))
    print(json.dumps({k: report[k] for k in (
        'status','wall_seconds','sessions','audio_turns',
        'qa_completed','writer_jobs','errors')},indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
