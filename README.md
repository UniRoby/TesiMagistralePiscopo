# Tesi Magistrale Piscopo

Codice sperimentale per localizzare manipolazioni sintetiche locali in CT 3D usando una pipeline patch-wise ispirata a `Localization of Synthetic Manipulations in Western Blot Images` e valutata su M3Dsynth.


## Struttura

- `src/tesi_m3d/`: dataset, patch 3D, modello, loss, inference, post-processing, evaluation.
- `configs/`: configurazioni train x -> test y,z; leave-generator-out & 
- `scripts/`: entrypoint CLI sottili.
- `tests/`: test sintetici senza dataset reale.


## Setup

Ambiente gia presente in `.venv`. 

## Test patch extraction volumetrica

Questo test non richiede M3Dsynth scaricato. Usa un volume sintetico `64x64x64` e una mask con cubo manipolato `32x32x32`.

```bash
cd PATH PROGETTO
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_patch3d.py
```

Criteri attesi:

- `PatchGrid(64^3, patch=32^3, stride=16)` produce `27` patch.
- Patch partono sugli assi da `0, 16, 32`.
- Patch senza voxels manipolati hanno label `0`.
- Patch boundary con `0 < overlap < 0.05` hanno label `None`.
- Patch con `overlap >= 0.05` hanno label `1`.
- Heatmap ricostruita ha shape `64^3` e score piu alto nel cubo.


## Test completi

```bash
cd path progetto
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
```


## Smoke training sintetico

Questo comando non usa M3Dsynth/LIDC. Genera volumi NumPy finti, allena il modello per 2 epoche CPU, salva checkpoint e heatmap in `outputs/synthetic_smoke/`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/smoke_synthetic_training.py
```

Uso: validare pipeline end-to-end prima della macchina laboratorio. Non produce risultati scientifici.


## Dataset M3Dsynth: metadata e TIFF

Il training usa due percorsi diversi:

- `metadata_dir`: CSV ufficiali dal repo clonato M3Dsynth, default `../M3Dsynth/data`. TODO cambia percorso 
- `data_root`: dataset TIFF scaricato, default `data/M3Dsynth`, con sottocartelle `cycle/`, `pix2pix/`, `diffusion/`, `real/`.


## Esperimenti cross-generator

Protocollo principale di tesi: allenare su un solo generatore e valutare sugli altri due. Questo misura generalizzazione severa tra famiglie di manipolazioni. (Attualmente un solo dataset manipolato scaricato)

Configurazioni principali:

- `configs/train_pix2pix_test_cycle_diffusion.yaml`
- `configs/train_cycle_test_pix2pix_diffusion.yaml`
- `configs/train_diffusion_test_pix2pix_cycle.yaml`

Esempio:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m tesi_m3d.train --config configs/train_pix2pix_test_cycle_diffusion.yaml
```

Le configurazioni `configs/leave_out_*.yaml` restano come baseline secondaria: allenano su due generatori e valutano sul generatore escluso. I dati M3Dsynth/LIDC vanno scaricati in `data/` o cartella esterna ignorata da Git.

## Nota preprocessing

Default v1: percentile normalization. HU clipping fisso `[-1000, 400]` sara aggiunto solo dopo verifica del range reale dei TIFF M3Dsynth (`dtype`, `min`, `max`).
