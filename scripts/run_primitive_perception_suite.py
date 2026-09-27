#!/usr/bin/env python3
import subprocess,sys
for test in ('tests.unit.test_primitive_perception','tests.unit.test_perception','tests.unit.test_grasp_planning','tests.unit.test_async_runtime','tests.unit.test_placement'):
 print(f'== {test} =='); subprocess.run([sys.executable,'-m',test],check=True)
print('PRIMITIVE PERCEPTION SUITE PASS')
