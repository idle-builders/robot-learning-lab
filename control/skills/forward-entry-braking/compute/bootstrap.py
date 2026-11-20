"""Execute the unchanged isolated bootstrap with its counted task path change."""
import subprocess
from common import project

source = project('control/skills/forward-stop-learning/compute/bootstrap.sh',
    '1dc85d20542f3b193125b4f20eb2f0bfabda6ffe4804b5573b9a760a13055272', [
        ('control/skills/forward-stop-learning/compute', 'control/skills/forward-entry-braking/compute', 1),
    ])
subprocess.run(['bash', '-c', source], check=True)
