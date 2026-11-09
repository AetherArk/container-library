"""CPU checks for release configuration; never publish or emulate registry results."""
from pathlib import Path
import re
import subprocess
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow(name):
    return yaml.load((ROOT / '.github/workflows' / name).read_text(), Loader=yaml.BaseLoader)


class ReleasePackagingTests(unittest.TestCase):
    def test_ci_covers_shared_launcher_and_both_test_directories(self):
        data = workflow('boltz2.yml')
        for event in ('push', 'pull_request'):
            for path in ('run_cofold.py', 'scripts/**', 'tests/**'):
                self.assertIn(path, data['on'][event]['paths'])
        self.assertIn('python scripts/test_local.py --suite cpu', [s.get('run') for s in data['jobs']['test']['steps']])

    def test_publish_workflows_supply_the_shared_common_build_context(self):
        # Both Dockerfiles COPY --from=common, so a workflow that omits the named
        # context fails at build time rather than at review time.
        for name in ('boltz2-publish.yml', 'esmfold2-publish.yml'):
            text = (Path(__file__).resolve().parents[1] / '.github/workflows' / name).read_text()
            self.assertIn('--build-context common=cofolding/common', text, name)
            self.assertIn('docker buildx build', text, name)

    def test_publication_is_manual_and_uses_scoped_workflow_token(self):
        data = workflow('boltz2-publish.yml')
        self.assertEqual(set(data['on']), {'workflow_dispatch'})
        self.assertEqual(data['jobs']['publish']['permissions'], {'contents': 'read', 'packages': 'write'})
        self.assertEqual(data['permissions'], {'contents': 'read'})
        self.assertEqual(data['concurrency']['cancel-in-progress'], 'false')
        steps = data['jobs']['publish']['steps']
        self.assertTrue(any(s.get('env', {}).get('GHCR_TOKEN') == '${{ secrets.GITHUB_TOKEN }}' for s in steps))

    def test_checks_precede_push_and_registry_digest_is_pulled(self):
        steps = workflow('boltz2-publish.yml')['jobs']['publish']['steps']
        names = [s.get('name') for s in steps]
        order = ['CPU tests', 'Build labeled release image', 'Installed-image checks before publication', 'Push and resolve registry digest', 'Pull and check the published digest']
        self.assertEqual(sorted(names.index(n) for n in order), [names.index(n) for n in order])
        text = '\n'.join(s.get('run', '') for s in steps)
        for value in ('CONTAINER_REVISION="$GITHUB_SHA"', 'CONTAINER_VERSION="$RELEASE_TAG"', 'docker pull "$PUBLISHED_IMAGE"'):
            self.assertIn(value, text)
        self.assertNotIn(':latest', text)
        for suite in ('image', 'image-cloud'):
            checks = [line.strip() for line in text.splitlines()
                      if line.strip().startswith('python scripts/test_local.py --suite ' + suite + ' ')]
            self.assertEqual(len(checks), 2)
            self.assertTrue(any('"$IMAGE:$RELEASE_TAG"' in command for command in checks))
            self.assertTrue(any('"$PUBLISHED_IMAGE"' in command for command in checks))
            for command in checks:
                self.assertIn('--cpus 2 --image', command)

    def test_actions_are_pinned_and_shell_scripts_parse(self):
        for file in ('boltz2.yml', 'boltz2-publish.yml'):
            data = workflow(file)
            for job in data['jobs'].values():
                for step in job['steps']:
                    if 'uses' in step:
                        self.assertRegex(step['uses'], r'@[0-9a-f]{40}$')
                    if 'run' in step:
                        result = subprocess.run(['bash', '-n'], input=step['run'], text=True, capture_output=True)
                        self.assertEqual(result.returncode, 0, (file, result.stderr))

    def test_labels_separate_container_and_upstream_identity(self):
        text = (ROOT / 'cofolding/boltz2/Dockerfile').read_text()
        for value in ('org.opencontainers.image.version="${CONTAINER_VERSION}"', 'org.opencontainers.image.revision="${CONTAINER_REVISION}"', 'ai.aetherark.boltz.revision="cb04aeccdd480fd4db707f0bbafde538397fa2ac"'):
            self.assertIn(value, text)
        tag = workflow('boltz2-publish.yml')['on']['workflow_dispatch']['inputs']['image_tag']['default']
        self.assertTrue(re.fullmatch(r'2\.2\.1-worker-[A-Za-z0-9][A-Za-z0-9_.-]*', tag))
        self.assertEqual(tag, '2.2.1-worker-v3-benchmark-rc1')
        self.assertIn('ARG CONTAINER_VERSION=' + tag, text)
