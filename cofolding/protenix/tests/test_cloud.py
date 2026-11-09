"""CPU transport/preparation tests: fake object store, explicit FAKE worker, NO inference."""
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cloud_job as cloud  # noqa: E402
import prepare_cloud_job as host  # noqa: E402
from resident.worker import Settings  # noqa: E402

IMAGE = 'test/protenix@sha256:' + 'a' * 64


class Store:
    """In-memory S3 double: create-only puts, readback, presigning."""

    def __init__(self):
        self.objects = {}
        self.signatures = []

    def put_object(self, Bucket, Key, Body, IfNoneMatch):
        assert IfNoneMatch == '*'
        if Key in self.objects:
            raise ValueError('collision')
        self.objects[Key] = Body.read() if hasattr(Body, 'read') else Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {'Body': io.BytesIO(self.objects[Key])}

    def generate_presigned_url(self, method, Params, ExpiresIn):
        self.signatures.append((method, Params, ExpiresIn))
        return 'https://example.invalid/' + Params['Key'] + '?secret=TEST-CAPABILITY'


class CloudPrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        (self.inputs / 'input.yaml').write_bytes(
            b'version: 1\nsequences:\n- protein:\n    id: A\n    sequence: AAAA\n'
            b'- ligand:\n    id: L\n    smiles: CCO\n')
        (self.inputs / 'unreferenced.secret.json').write_text('TEST-CAPABILITY must not be bundled')
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        self.checkpoint = self.assets / 'checkpoint'
        self.checkpoint.mkdir()
        (self.checkpoint / 'config.json').write_bytes(b'{}')
        self.store = Store()
        self.args = host.parser().parse_args([
            '--inputs', str(self.inputs), '--output', str(self.root / 'prepared'),
            '--bucket', 'tests', '--image', IMAGE, '--seeds', '12', '11', '--samples', '2'])

    def test_prepare_packages_inputs_and_writes_private_launch_files(self):
        job = host.prepare(self.args, client=self.store)
        self.assertRegex(job['job_id'], r'^protenix-\d{8}T\d{6}Z-[a-f0-9]{12}$')
        self.assertEqual(job['settings'], {**asdict(Settings()), 'samples': 2})
        self.assertEqual(set(job['files']), {'input.yaml'})
        with tarfile.open(self.root / 'prepared' / 'inputs.tar.gz') as tar:
            self.assertEqual(tar.getnames(), ['inputs/input.yaml'])
            self.assertNotIn('inputs/unreferenced.secret.json', tar.getnames())
            self.assertEqual(tar.extractfile('inputs/input.yaml').read(),
                             (self.inputs / 'input.yaml').read_bytes())
        for name in ('runtime-env.secret.json', 'deployment.secret.json'):
            self.assertEqual((self.root / 'prepared' / name).stat().st_mode & 0o777, 0o600)
        for name in ('job.json', 'launch.json'):
            self.assertNotIn('TEST-CAPABILITY', (self.root / 'prepared' / name).read_text())
        payload = json.loads((self.root / 'prepared' / 'deployment.secret.json').read_text())
        self.assertEqual(payload['command'], ['cofold-cloud-run'])

    def test_settings_are_validated_by_the_transport(self):
        cloud.validate_settings([11], asdict(Settings(samples=2)))
        for bad in ({'max_parallel_samples': 2}, {'output_format': 'pdb'},
                    {'steps': 0}, {'recycles': 0}, {'derived': 'full'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cloud.validate_settings([11], {**asdict(Settings()), **bad})
        self.assertNotIn('backend', asdict(Settings()))

    def test_worker_command_has_protenix_flags_and_no_kernel_backend(self):
        job = {'seeds': [11], 'settings': asdict(Settings(samples=2))}
        command = cloud.worker_command(job, Path('/in'), Path('/out'), Path('/models'),
                                       Path('/models'))
        for flag, value in (('--samples', '2'), ('--recycles', '10'), ('--steps', '200'),
                            ('--derived', 'tensors'), ('--output-format', 'mmcif')):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn('--backend', command)
        self.assertNotIn('--no-kernels', command)
        self.assertNotIn('--no-low-memory', command)

    def test_host_rejects_external_missing_absolute_and_symlink_dependencies(self):
        for ref in ('../outside.a3m', '/absolute.a3m', 'missing.a3m'):
            (self.inputs / 'input.yaml').write_text(
                'version: 1\nsequences:\n- protein:\n    id: A\n    sequence: AAAA\n'
                '    msa: ' + ref + '\n')
            with self.assertRaises(ValueError):
                host.inspect_inputs(self.inputs)
        (self.inputs / 'input.yaml').write_bytes(
            b'version: 1\nsequences:\n- protein:\n    id: A\n    sequence: AAAA\n')
        (self.inputs / 'link').symlink_to(self.checkpoint)
        with self.assertRaises(ValueError):
            host.inspect_inputs(self.inputs)

    def test_image_must_be_an_immutable_digest(self):
        self.args.image = 'mutable:tag'
        with self.assertRaises(ValueError):
            host.prepare(self.args, client=self.store)
        self.assertFalse((self.root / 'prepared').exists())


if __name__ == '__main__':
    unittest.main()
