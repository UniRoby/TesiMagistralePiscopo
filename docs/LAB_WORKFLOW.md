# Workflow PC laboratorio Windows

Guida autonoma per configurare la macchina, validare i dati e avviare gli
esperimenti della tesi.

## 1. Ambiente disponibile

- Windows, esecuzione locale tramite Conda.
- CPU Intel Core i5-14400.
- RAM 32 GB.
- GPU NVIDIA RTX 5060 Ti con 16 GB VRAM.
- Dataset DICOM LIDC-IDRI già scaricato.
- Dataset manipolato pix2pix già scaricato.
- Diffusion in download; cycle non ancora disponibile.

Percorsi:

```text
C:\Tesi Magistrale Piscopo\Reale\lidc_idri       DICOM LIDC-IDRI
C:\Tesi Magistrale Piscopo\Reale\metadata.csv   manifest download IDC
C:\Tesi Magistrale Piscopo\pix2pix\Scan          TIFF manipolati
C:\Tesi Magistrale Piscopo\pix2pix\label         mask TIFF
```

## 2. Clone e aggiornamento repository

Aprire Anaconda Prompt:

```bat
cd /d "C:\Tesi Magistrale Piscopo"
git clone URL_DELLA_REPOSITORY TesiMagistralePiscopo
cd TesiMagistralePiscopo
```

Se la repository è già presente:

```bat
cd /d "C:\Tesi Magistrale Piscopo\TesiMagistralePiscopo"
git pull
```

Il clone separato di `grip-unina/M3Dsynth` non è più necessario per leggere i
CSV: sono versionati in `metadata\m3dsynth`. Può essere mantenuto come codice di
riferimento del paper.

## 3. Environment Conda

Creare l'environment una volta sola:

```bat
conda create -n tesi-m3d python=3.11 -y
conda activate tesi-m3d
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train,conversion]"
```

Nelle sessioni successive bastano:

```bat
cd /d "C:\Tesi Magistrale Piscopo\TesiMagistralePiscopo"
conda activate tesi-m3d
```

## 4. Verifica NVIDIA e PyTorch

```bat
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'non disponibile')"
```

Risultato necessario: `cuda True` e nome RTX 5060 Ti. Se CUDA è `False`, usare
il selettore ufficiale PyTorch per installare una build Windows compatibile con
il driver NVIDIA, poi ripetere il controllo.

## 5. Test del codice senza dataset

```bat
python -m pytest -q -p no:cacheprovider
python scripts\smoke_synthetic_training.py
```

Atteso:

- tutti i test passano;
- loss numerica, non `nan`;
- `outputs\synthetic_smoke\synthetic_patch3d.pt`;
- heatmap con shape `(64, 64, 64)`.

Questa fase dimostra solo che la pipeline funziona end-to-end.

## 6. Ruolo dei CSV

### Manifest IDC `Reale\metadata.csv`

Il file contiene 1.308 serie appartenenti a 1.010 pazienti: PatientID, Study e
Series Instance UID, dimensione, URL, percorso di download e stato. Tutte le
righe hanno `completion_status=success`.

Serve per:

- confermare che il download DICOM sia terminato;
- associare le serie richieste da M3Dsynth alle cartelle create dal downloader;
- diagnosticare serie mancanti tramite PatientID e SeriesInstanceUID.

Non decide gli split e non viene letto dal training. La verifica svolta nella
repo ha trovato una corrispondenza unica per tutte le 744 serie di `LIDC.csv`:
il file è sufficiente e non deve essere pulito. Il suo campo
`S5cmdManifestPath` può conservare il vecchio path senza `Reale`, perché lo
script ricostruisce il percorso relativo a partire da PatientID.

### CSV ufficiali `metadata\m3dsynth`

- `data.csv`: 11.828 record; collega `img_id`, generatore, `orig_id`, serie e coordinate.
- `sets.csv`: assegna ciascun `orig_id` a train, valid o test evitando leakage.
- `centers.csv`: centri di crop usati dalle baseline ufficiali.
- `LIDC.csv`: 744 serie DICOM reali richieste e relativo identificatore `sdir_id`.

La pipeline custom usa direttamente `data.csv` e `sets.csv`; la conversione usa
anche `LIDC.csv`. `centers.csv` resta disponibile per confronti con il protocollo
ufficiale.

## 7. Perché convertire i reali in TIFF

Il loader di training riceve stack TIFF sia per pix2pix sia per i negativi
reali. La conversione è quindi necessaria per:

- fornire esempi reali con lo stesso contenitore dei manipolati;
- mantenere ordinamento delle slice e conversione `uint16` di M3Dsynth;
- applicare la stessa normalizzazione percentile a entrambe le classi.

Non eseguire `get_M3Dsynth.sh`: oltre a non essere nativo Windows, scarica
cycle, pix2pix e diffusion prima di convertire LIDC. I dati manipolati già
presenti non devono essere riscaricati.

## 8. Smoke conversion di una serie

Prima controllare che `pydicom` sia installato:

```bat
python -c "import pydicom; print(pydicom.__version__)"
```

Convertire una sola serie:

```bat
python scripts\convert_lidc_to_tiff.py ^
  --dicom-root "C:\Tesi Magistrale Piscopo\Reale\lidc_idri" ^
  --download-metadata "C:\Tesi Magistrale Piscopo\Reale\metadata.csv" ^
  --output-root "C:\Tesi Magistrale Piscopo" ^
  --metadata-dir metadata\m3dsynth ^
  --limit 1 --workers 1
```

Controllare che sia apparsa una directory simile a:

```text
C:\Tesi Magistrale Piscopo\real\scan\LIDC-IDRI-0003__3000611\
  slide0000.tiff
  slide0001.tiff
  ...
  .complete
```

Aprire alcune slice con un visualizzatore TIFF e verificare che siano immagini
CT leggibili, non completamente nere e senza errori di orientamento evidenti.

## 9. Conversione completa dei reali

```bat
python scripts\convert_lidc_to_tiff.py ^
  --dicom-root "C:\Tesi Magistrale Piscopo\Reale\lidc_idri" ^
  --download-metadata "C:\Tesi Magistrale Piscopo\Reale\metadata.csv" ^
  --output-root "C:\Tesi Magistrale Piscopo" ^
  --metadata-dir metadata\m3dsynth ^
  --workers 2
```

Con 32 GB RAM partire da 2 processi. Aumentare a 4 solo dopo aver controllato
RAM e temperature. Le serie già concluse, riconosciute dal file `.complete`,
vengono saltate: il comando può essere rilanciato dopo un'interruzione.

## 10. Verifica struttura pix2pix

La root dataset è `C:\Tesi Magistrale Piscopo`, non la cartella del progetto.
La struttura minima deve essere:

```text
C:\Tesi Magistrale Piscopo\
  pix2pix\
    Scan\<img_id>\slide0000.tiff ...
    label\<img_id>\slide0000.tiff ...
  real\
    scan\<orig_id>__<sdir_id>\slide0000.tiff ...
```

Windows non distingue `Scan` da `scan`; il loader richiede logicamente `scan`.
Il nome `label` deve contenere le mask con gli stessi `img_id` di pix2pix.

## 11. Primo training disponibile: pix2pix

```bat
python -m tesi_m3d.train ^
  --config configs\train_pix2pix_test_cycle_diffusion.yaml ^
  --data-root "C:\Tesi Magistrale Piscopo" ^
  --device cuda
```

Questa run:

- legge split e record da `metadata\m3dsynth`;
- usa pix2pix positivo e CT reali negativi del solo split train;
- estrae patch `32x32x32` con stride train `32`;
- esclude patch boundary con overlap positivo inferiore al 5%;
- bilancia classi con `WeightedRandomSampler`;
- allena il classificatore 3D e salva il checkpoint.

Output:

```text
outputs\train_pix2pix_test_cycle_diffusion\patch3d_classifier.pt
```

Al momento il comando di training non esegue ancora la valutazione finale sui
due generatori indicati nel nome. Il checkpoint pix2pix serve a validare
architettura, accesso ai dati, patch extraction e uso GPU. La valutazione
cross-generator sarà eseguita quando cycle e diffusion saranno presenti.

## 12. Protocollo finale quando arrivano gli altri dataset

Run principali:

```bat
python -m tesi_m3d.train --config configs\train_pix2pix_test_cycle_diffusion.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda
python -m tesi_m3d.train --config configs\train_cycle_test_pix2pix_diffusion.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda
python -m tesi_m3d.train --config configs\train_diffusion_test_pix2pix_cycle.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda
```

Le configurazioni `leave_out_*.yaml` restano baseline secondaria: allenamento su
due generatori e test sul terzo.

## 13. Errori comuni

### `metadata directory not found`

Eseguire il comando dalla root `TesiMagistralePiscopo` e verificare:

```bat
dir metadata\m3dsynth
```

Devono esserci `data.csv`, `sets.csv`, `centers.csv`, `LIDC.csv`.

### `DICOM series not found`

Controllare che `--dicom-root` termini con `Reale\lidc_idri` e che
`--download-metadata` indichi il manifest associato a quel download.

### `no TIFF slices found`

Controllare `--data-root`. Deve essere la cartella che contiene direttamente
`pix2pix` e `real`, quindi `C:\Tesi Magistrale Piscopo`.

### `cuda False`

La build PyTorch installata non usa CUDA oppure il driver è incompatibile.
Verificare prima `nvidia-smi`, poi reinstallare PyTorch con il comando generato
dal selettore ufficiale per Windows/CUDA.

### Memoria GPU esaurita

Ridurre in config:

```yaml
training:
  batch_size: 4
model:
  base_channels: 8
```

### RAM o preparazione patch troppo lente

Lasciare inizialmente `num_workers: 0` su Windows. Il codice v1 indicizza le
patch leggendo le mask prima del training; per il primo test si deve attendere
questa fase anche se la GPU non è ancora occupata.
