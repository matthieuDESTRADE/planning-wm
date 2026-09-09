import os
import sys
import subprocess

env = os.environ.copy()


for max_nodes in [500,1000,2000]:
    for sample_scale in [0.5,1.0,2.0,5.0,10.0]:
        for sample_size in [1,32,64,128]:
            for batch_size in [128]:
                subprocess.run([sys.executable,
                    "eval.py",
                    "policy=pusht/lewm", 
                    f"solver1.max_nodes={max_nodes}",
                    f"solver1.sample_scale={sample_scale}",
                    f"solver1.sample_size={sample_size}",
                    f"solver1.batch_size={batch_size}",
                    "--config-name", 'pusht.yaml',
                ], env=env, check=True)