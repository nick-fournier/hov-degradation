# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['hov_degradation_launcher.py'],
             pathex=['C:\\gitclones\\connected-corridors\\hov-degradation'],
             binaries=[],
             datas=[('hov_degradation/static/hyperparameters_I210_2020-12-06_to_2020-12-12.json', 'hov_degradation/static'), ('hov_degradation/static/scores_I210_2020-12-06_to_2020-12-12.json', 'hov_degradation/static'), ('hov_degradation/static/trained_classification_I210_2020-12-06_to_2020-12-12.pkl', 'hov_degradation/static'), ('hov_degradation/static/trained_unsupervised_I210_2020-12-06_to_2020-12-12.pkl', 'hov_degradation/static')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='hov_degradation_launcher',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
