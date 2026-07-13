# Workflow Laboratorio GPU

Guida autonoma per rieseguire progetto tesi su macchina laboratorio con disco 1TB e GPU NVIDIA.

## 1. Obiettivo

Protocollo principale: train su un generatore M3Dsynth, test sugli altri due.

Run principali:

- train `pix2pix`, test `cycle + diffusion`
- train `cycle`, test `pix2pix + diffusion`
- train `diffusion`, test `pix2pix + cycle`

Le run `leave_out_*` restano baseline secondaria: train su due generatori, test su uno.

## 2. Clone repository

Scegli una cartella lavoro con spazio sufficiente, esempio `/data/tesi`.

```bash
mkdir -p /data/tesi
cd /data/tesi

git clone <URL_REPO_TESI> TesiMagistralePiscopo
git clone https://github.com/grip-unina/M3Dsynth.git M3Dsynth
```

Struttura attesa:

```text
/data/tesi/
  TesiMagistralePiscopo/
  M3Dsynth/
```

## 3. Ambiente Python

Consigliato Python 3.10 o 3.11.

```bash
cd /data/tesi/TesiMagistralePiscopo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,train]'
```

Se server usa CUDA specifica, installa PyTorch seguendo comando ufficiale da https://pytorch.org/get-started/locally/.
Poi rilancia:

```bash
python -m pip install -e '.[dev,train]'
```

## 4. Check GPU

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
```

Atteso: `cuda_available True` e nome GPU NVIDIA.

## 5. Test senza dataset reale

Prima di scaricare dati o lanciare training vero:

```bash
cd /data/tesi/TesiMagistralePiscopo
source .venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_synthetic_training.py
```

Atteso:

- test passano
- loss numerica
- checkpoint `outputs/synthetic_smoke/synthetic_patch3d.pt`
- heatmap `outputs/synthetic_smoke/synthetic_heatmap.npy`

## 6. Dataset: path importanti

Il progetto usa due path diversi:

```yaml
data_root: data/M3Dsynth
metadata_dir: ../M3Dsynth/data
```

Significato:

- `metadata_dir`: CSV ufficiali del repo M3Dsynth (`data.csv`, `sets.csv`, `centers.csv`).
- `data_root`: TIFF reali/manipolati scaricati per training.

Struttura attesa dopo download:

```text
TesiMagistralePiscopo/data/M3Dsynth/
  pix2pix/scan/<img_id>/slide0000.tiff ...
  pix2pix/label/<img_id>/slide0000.tiff ...
  cycle/scan/<img_id>/...
  cycle/label/<img_id>/...
  diffusion/scan/<img_id>/...
  diffusion/label/<img_id>/...
  real/scan/<img_id>/...
```

## 7. Download dati ufficiali

Serve LIDC-IDRI locale in DICOM per creare `real/`.

```bash
cd /data/tesi/M3Dsynth
bash ./get_M3Dsynth.sh /PATH/TO/LIDC-IDRI /data/tesi/TesiMagistralePiscopo/data/M3Dsynth
```

Note:

- `cycle.tgz`, `pix2pix.tgz`, `diffusion.tgz` vengono scaricati dal sito GRIP.
- `real/` viene creato convertendo LIDC-IDRI.
- Se `wget` manca, installalo o sostituisci download con `curl -L -o file.tgz URL`.

## 8. Verifica metadata

```bash
cd /data/tesi/TesiMagistralePiscopo
python - <<'PY'
from pathlib import Path
from tesi_m3d.dataset import read_records
records = read_records(Path('../M3Dsynth/data'))
print('records', len(records))
print(records[0])
PY
```

Atteso: numero record > 0.

## 9. Training cross-generator

Run principale 1:

```bash
cd /data/tesi/TesiMagistralePiscopo
source .venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m tesi_m3d.train \
  --config configs/train_pix2pix_test_cycle_diffusion.yaml \
  --device cuda
```

Run principale 2:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m tesi_m3d.train \
  --config configs/train_cycle_test_pix2pix_diffusion.yaml \
  --device cuda
```

Run principale 3:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m tesi_m3d.train \
  --config configs/train_diffusion_test_pix2pix_cycle.yaml \
  --device cuda
```

Output atteso:

```text
outputs/train_pix2pix_test_cycle_diffusion/patch3d_classifier.pt
outputs/train_cycle_test_pix2pix_diffusion/patch3d_classifier.pt
outputs/train_diffusion_test_pix2pix_cycle/patch3d_classifier.pt
```

## 10. Baseline secondaria leave-one-out

```bash
PYTHONDONTWRITEBYTECODE=1 python -m tesi_m3d.train --config configs/leave_out_cycle.yaml --device cuda
PYTHONDONTWRITEBYTECODE=1 python -m tesi_m3d.train --config configs/leave_out_pix2pix.yaml --device cuda
PYTHONDONTWRITEBYTECODE=1 python -m tesi_m3d.train --config configs/leave_out_diffusion.yaml --device cuda
```

## 11. Errori comuni

### `metadata directory not found`

Controlla config:

```yaml
metadata_dir: ../M3Dsynth/data
```

Repo M3Dsynth deve essere sibling di repo tesi.

### `no TIFF slices found`

`data_root` punta a cartella sbagliata o download incompleto.
Controlla:

```bash
find data/M3Dsynth -maxdepth 3 -type d | head
```

### `cuda_available False`

PyTorch CPU-only o driver CUDA assente. Verifica `nvidia-smi` e reinstalla PyTorch CUDA.

### Out of memory GPU

Riduci in config:

```yaml
training:
  batch_size: 4
model:
  base_channels: 8
```

### Training troppo lento

Aumenta solo se RAM/CPU reggono:

```yaml
training:
  num_workers: 4
```

## 12. Note scientifiche

- Training sintetico NumPy non produce risultato tesi.
- Serve solo per validare pipeline.
- Risultati tesi arrivano da M3Dsynth reale.
- Split deve evitare leakage su `orig_id`.
- Protocollo principale resta single-generator training -> two-generator testing.
