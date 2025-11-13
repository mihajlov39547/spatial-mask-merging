import sys
# Clean modules
for mod in list(sys.modules.keys()):
    if mod.startswith('smm'):
        del sys.modules[mod]

# Fresh import
import smm

public_keys = [k for k in smm.__dict__.keys() if not k.startswith('_')]
print('Public namespace:', public_keys)
print('Expected:', smm.__all__)
unexpected = set(public_keys) - set(smm.__all__)
print('Unexpected exports:', unexpected if unexpected else 'None')
