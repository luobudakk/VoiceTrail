'''Check release contents for accidentally committed private files and credentials.'''
from __future__ import annotations
import argparse
import re
import subprocess
from pathlib import Path

TEXT_EXT = {'.py','.md','.toml','.txt','.yaml','.yml','.json','.cfg','.ini',
            '.gitignore','.gitattributes','.example'}
FORBIDDEN_EXT = {'.pem','.key','.p12','.pfx','.wav','.flac','.mp3','.mp4',
                 '.safetensors','.pt','.pth','.ckpt','.sqlite','.sqlite3','.db',
                 '.jsonl','.ipynb'}
FORBIDDEN_NAMES = {'id_rsa','id_ed25519','.env','credentials.json','secrets.json'}
FORBIDDEN_DIRS = {'data','datasets','results','runs','logs','outputs','checkpoints',
                  'weights','models','artifacts','private','credentials'}
PATTERNS = {
    'absolute_home_path': re.compile(r'/home/[A-Za-z][A-Za-z0-9_.-]+/'),
    'windows_user_profile': re.compile(r'[A-Z]:\\Users\\[A-Za-z0-9_. -]+\\',re.I),
    'private_ipv4': re.compile(r'(?<![0-9])(?:10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})(?![0-9])'),
    'secret_literal': re.compile(r'(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*[\"\x27][A-Za-z0-9_-]{20,}[\"\x27]',re.I),
    'bearer_value': re.compile(r'Bearer\s+[A-Za-z0-9_-]{25,}'),
    'old_benchmark_user_id': re.compile(r'\betm_z\d{3}\b'),
}

def scan(root: Path) -> list[str]:
    issues=[]
    for p in root.rglob('*'):
        rel=p.relative_to(root)
        if '.git' in rel.parts or '__pycache__' in rel.parts or '.pytest_cache' in rel.parts:
            continue
        if p.is_dir():
            if p.name.lower() in FORBIDDEN_DIRS:
                issues.append('forbidden directory: '+str(rel))
            continue
        if p.name.lower() in FORBIDDEN_NAMES or p.suffix.lower() in FORBIDDEN_EXT:
            issues.append('private-format file: '+str(rel))
        if p.stat().st_size>1_000_000 and p.suffix.lower() not in {'.md'}:
            issues.append('oversized release file: '+str(rel))
        if p.suffix.lower() not in TEXT_EXT and p.name not in {'.gitignore','.gitattributes'}:
            continue
        if p.name=='security_scan.py':
            continue # This script contains the detection patterns as literal source.
        text=p.read_text(encoding='utf8',errors='replace')
        for name,pat in PATTERNS.items():
            if pat.search(text):
                issues.append(name+': '+str(rel))
    return issues

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    issues=scan(args.root.resolve())
    if issues:
        print('SECURITY_SCAN_FAILED')
        for issue in issues:
            print(' -',issue)
        return 1
    print('SECURITY_SCAN_PASS')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
