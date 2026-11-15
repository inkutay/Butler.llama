import importlib.util
import json
from pathlib import Path
import tempfile
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('butler_setup',root/'scripts/setup.py')
setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)
with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp);(folder/'image').mkdir();(folder/'image/a.svg').write_text('<svg/>');(folder/'index.html').write_text('<title>Butler</title>')
    answers=iter(['local-test-password','local-test-password'])
    setup.prepare(folder,ask_password=lambda _:next(answers))
    config=json.loads((folder/'butler_config.json').read_text())
    assert config['auth_pass']=='local-test-password' and len(config['api_key'])>=32
    assert len((folder/'jwt_secret.key').read_text().strip())>=48
    assert json.loads((folder/'butler_instances.json').read_text())==[]
    assert (folder/'static/image/a.svg').read_text()=='<svg/>'
    saved={name:(folder/name).read_bytes() for name in ('butler_config.json','butler_instances.json','jwt_secret.key')}
    def no_prompt(_):raise AssertionError('must preserve existing settings')
    (folder/'index.html').write_text('updated UI')
    setup.prepare(folder,ask_password=no_prompt)
    assert all((folder/name).read_bytes()==data for name,data in saved.items())
    assert (folder/'static/index.html').read_text()=='updated UI'
with tempfile.TemporaryDirectory() as tmp:
    try:setup.prepare(Path(tmp),ask_password=lambda _:'x')
    except ValueError:pass
    else:raise AssertionError('short password must fail')
    assert not (Path(tmp)/'butler_config.json').exists()
print('PASS: fresh setup, generated secrets, static assets, existing settings preserved, short password rejected')
