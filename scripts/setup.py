#!/usr/bin/env python3
"""Prepare a fresh checkout without replacing local credentials or instance data."""
import getpass
import json
import os
from pathlib import Path
import secrets
import shutil


def write_private(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(data)


def prepare(root, ask_password=getpass.getpass):
    root = Path(root)
    config = root / 'butler_config.json'
    if not config.exists():
        password = ask_password('Choose a Butler admin password (at least 3 characters): ')
        if len(password) < 3:
            raise ValueError('Password must contain at least 3 characters')
        if password != ask_password('Repeat password: '):
            raise ValueError('Passwords do not match')
        models = root / 'models'
        models.mkdir(exist_ok=True)
        settings = dict(api_url='http://127.0.0.1:8090/v1', api_key=secrets.token_urlsafe(32),
                        models_dir=str(models.resolve()), auth_user='admin', auth_pass=password,
                        idle_enabled=False, idle_timeout_minutes=30)
        write_private(config, json.dumps(settings, indent=2, ensure_ascii=False)+'\n')
    if not (root / 'butler_instances.json').exists():
        write_private(root / 'butler_instances.json', '[]\n')
    if not (root / 'jwt_secret.key').exists():
        write_private(root / 'jwt_secret.key', secrets.token_urlsafe(48)+'\n')
    static = root / 'static'
    (static / 'image').mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / 'index.html', static / 'index.html')
    for source in (root / 'image').iterdir():
        if source.is_file():
            shutil.copy2(source, static / 'image' / source.name)
    print('Static assets prepared. Existing credentials and instance data preserved.')
    print('Login: admin (or your existing configured user). API key is in butler_config.json.')


if __name__ == '__main__':
    try:
        prepare(Path(__file__).resolve().parents[1])
    except (ValueError, OSError) as error:
        raise SystemExit(f'Setup failed: {error}')
