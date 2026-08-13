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
- Diffusion e cycle non ancora disponibili.

Percorsi:

```text
C:\Tesi Magistrale Piscopo\Reale\lidc_idri       DICOM LIDC-IDRI
C:\Tesi Magistrale Piscopo\Reale\metadata.csv   manifest download IDC
C:\Tesi Magistrale Piscopo\pix2pix\Scan          TIFF manipolati
C:\Tesi Magistrale Piscopo\pix2pix\label         mask TIFF
```

## 3. Environment Conda

Creare l'environment una volta sola:

```powershell
conda create -n tesi-m3d python=3.11 -y
conda activate tesi-m3d
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train,conversion]"
```

Nelle sessioni successive bastano:

```powershell
cd "C:\Tesi Magistrale Piscopo\TesiMagistralePiscopo"
conda activate tesi-m3d
.\.venv\Scripts\Activate.ps1
```

## 4. Verifica NVIDIA e PyTorch

```powershell
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'non disponibile')"
```

Risultato necessario: `cuda True` e nome RTX 5060 Ti.

## 5. Test del codice senza dataset

```powershell
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

### CSV ufficiali `metadata\m3dsynth`

- `data.csv`: collega `img_id`, generatore, `orig_id`, serie e coordinate.
- `sets.csv`: assegna ciascun `orig_id` a train, valid o test evitando leakage.
- `centers.csv`: centri di crop usati dalle baseline ufficiali.
- `LIDC.csv`: 744 serie DICOM reali richieste e relativo identificatore `sdir_id`.

La pipeline usa direttamente `data.csv` e `sets.csv`; la conversione usa
anche `LIDC.csv`. `centers.csv` resta disponibile per confronti con il protocollo
ufficiale.

## 7. Perché convertire i reali in TIFF

Il loader di training riceve stack TIFF sia per pix2pix sia per i negativi
reali. La conversione è quindi necessaria per:

- fornire esempi reali con lo stesso contenitore dei manipolati;
- mantenere ordinamento delle slice e conversione `uint16` di M3Dsynth;
- applicare la stessa normalizzazione percentile a entrambe le classi.

## Attiva .venv

```powershell
.\.venv\Scripts\Activate.ps1
```

## 8. Smoke conversion di una serie

Prima controllare che `pydicom` sia installato:

```powershell
python -c "import pydicom; print(pydicom.__version__)"
```

Convertire una sola serie:

```powershell
python scripts\convert_lidc_to_tiff.py `
  --dicom-root "C:\Tesi Magistrale Piscopo\Reale\lidc_idri" `
  --download-metadata "C:\Tesi Magistrale Piscopo\Reale\metadata.csv" `
  --output-root "C:\Tesi Magistrale Piscopo" `
  --metadata-dir metadata\m3dsynth `
  --limit 1 --workers 1
```

`--output-root` deve essere la dataset root, cioè la cartella che contiene già
`pix2pix`. Il loader legge `pix2pix\scan` e
`real\scan` dalla stessa radice. 
La prima riga stampata indica sempre la
root risolta:

```text
Dataset root: C:\Tesi Magistrale Piscopo
```

Controllare che sia apparsa una directory simile a:

```text
C:\Tesi Magistrale Piscopo\real\scan\LIDC-IDRI-0003__3000611\
  slide0000.tiff
  slide0001.tiff
  ...
  .complete
```

## 9. Conversione completa dei reali

```powershell
python scripts\convert_lidc_to_tiff.py `
  --dicom-root "C:\Tesi Magistrale Piscopo\Reale\lidc_idri" `
  --download-metadata "C:\Tesi Magistrale Piscopo\Reale\metadata.csv" `
  --output-root "C:\Tesi Magistrale Piscopo" `
  --metadata-dir metadata\m3dsynth `
  --workers 2
```

 Le serie già concluse, riconosciute dal file `.complete`,
vengono saltate: il comando può essere rilanciato dopo un'interruzione.

## 10. Verifica dei TIFF reali convertiti

`scripts\check_real_tiff.py` controlla che le serie convertite siano CT
leggibili, non nere e con l'orientamento giusto, senza aprire i file a mano.
Va eseguito dopo lo smoke test dello step 8 e di nuovo dopo la conversione completa
dello step 9:

```powershell
python scripts\check_real_tiff.py --scan-root "C:\Tesi Magistrale Piscopo\real\scan" --preview-dir outputs\real_check
```

L'output completo viene scritto su `outputs/check_real_tiff.log`. Il comando stampa anche il
percorso del file alla fine per facilitarne la consultazione.

Cosa verifica, per ogni serie:

- ogni `slide*.tiff` si apre, è `uint16` e ha la stessa shape delle altre;
- la numerazione è contigua e coincide con il conteggio scritto in `.complete`;
- nessuna slice è costante, e il volume contiene tessuti molli e osso nelle
  proporzioni attese per una CT del torace;
- l'orientamento è corretto: aria fuori dal paziente ai bordi, corpo al centro
  dell'inquadratura, colonna vertebrale nella metà posteriore (nessun flip
  verticale) e aria polmonare massima a metà stack (ordinamento z corretto).

Le soglie sono espresse in HU. L'offset `uint16` non è ricavabile dal TIFF, per
cui viene stimato dal picco dell'aria polmonare: è quello il modo corretto,
perché `scan_to_uint16` trasla il volume di `-min(scan)` e quel minimo è il
padding fuori campo, non l'aria. L'offset risulta quindi diverso da serie a
serie (2048 per `LIDC-IDRI-0003`, 3072 per i TIFF pix2pix). La cosa non è un
problema: `normalize_percentile` scarta i voxel a zero e riscala tra due
percentili dei soli voxel non nulli, quindi una traslazione uniforme si
annulla. 

Output atteso:

```text
=== LIDC-IDRI-0003__3000611 ===
  shape (z,y,x)      : (140, 512, 512)
  spine row fraction : 0.59 (>0.5 = posterior, correct)
  lung air peak      : slice 72/140 (18.3% of volume)
  OK all checks passed

1/1 series passed
```

Lo script esce con codice `1` se una serie non passa e stampa una riga `FAIL`
per ogni controllo fallito. Con `--preview-dir` scrive anche, per ogni serie,
un montaggio di 9 slice assiali più una ricostruzione coronale e una sagittale
in PNG già finestrate sul polmone: le due ricostruzioni sono il modo più rapido
per accorgersi di slice duplicate o fuori ordine, che vi appaiono come scalini.
Usare `--limit N` per controllare solo le prime N serie.

## 11. Allineamento automatico scan/mask z-dimension

Il dataset pix2pix contiene disallineamenti nella dimensione z (profondità):
- **Caso pix2pix**: ogni serie ha label con **+1 slice** (padding accidentale)
- **Caso generale**: scan e mask possono avere z-dimensioni diverse (fino a 85+ slice)

**Soluzione implementata**: il loader (`load_scan_and_mask` in `dataset.py`) allinea
automaticamente:
- Se mask è **più lunga**: trimma le slice extra dalla fine (convenzione pix2pix)
- Se mask è **più corta**: aggiunge padding con zeros alla fine (background)
- Se y/x non matchano: **errore critico** (non tollerato)

Il dataset originale rimane intatto sul disco. Durante il training puoi vedere
messaggi di allineamento — questo è **normale e atteso** per i dati reali.

## 12. Verifica struttura pix2pix

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

Il loader richiede logicamente `scan`.
Il nome `label` deve contenere le mask con gli stessi `img_id` di pix2pix.

## 13. Ottimizzazione data loading

Il dataset contiene ~4.2 milioni di patch su 3262 serie TIFF (239 GB totali).
Senza ottimizzazione, ogni batch ricarica i TIFF dal disco (88 sec/batch = infattibile).

**Ottimizzazioni implementate**:
- **LRU cache dei TIFF** (`@lru_cache maxsize=1000`): la stessa serie non viene
  ricaricata dal disco se già in RAM. Cache capace di ~75 GB.
- **Normalized paths**: sempre `resolved()` per consistency nel cache key.

Speedup atteso: **~50-100×** (da 88 sec/batch a 0.5-1 sec/batch), rendendo il
training completabile in **7-15 ore per epoch** (ragionevole).

## 14. Training: pix2pix

### Baseline rapida (consigliata per una sessione di laboratorio)

Per ottenere un checkpoint utilizzabile per detection e heatmap senza eseguire
il protocollo completo da 50 epoche, usare la configurazione limitata a 256
record di training, 64 di validation e 4.096 patch per epoca:

```powershell
python -m tesi_m3d.train `
  --config configs\train_pix2pix_baseline.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda
```

Prima eseguire il controllo della dimensione effettiva del sampler:

```powershell
python -m tesi_m3d.train `
  --config configs\train_pix2pix_baseline.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda --dry-run
```

Il dry-run deve riportare fino a 256 record selezionati, circa 128 batch per
epoca e positivi non nulli. Record reali che condividono la stessa scansione
vengono raggruppati e possono ridurre leggermente il numero esatto di batch. La
configurazione parte con `num_workers: 0`; per misurare
due worker, copiare la config, impostare temporaneamente
`max_patches_per_epoch: 640` e `num_workers: 2`, poi confrontare i 20 batch con
la variante a zero worker usando `nvidia-smi` e Gestione attività. Conservare
la variante più veloce solo se la RAM rimane stabile.

Per monitorare il confronto, in un secondo terminale eseguire:

Eseguire una configurazione alla volta, senza `--resume`:

```powershell
python -m tesi_m3d.train --config configs\train_pix2pix_workers0_benchmark.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda
python -m tesi_m3d.train --config configs\train_pix2pix_workers2_benchmark.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda
```

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1
```

In Gestione attivita > Prestazioni > Memoria, la RAM non deve crescere
continuamente. Conservare `num_workers: 2` solo se termina i 20 batch in meno
tempo, la GPU resta piu occupata e RAM/VRAM rimangono sotto circa l'85%; in
caso contrario mantenere `num_workers: 0`.

Ogni checkpoint contiene anche optimizer, AMP scaler ed epoca. Per proseguire
una run interrotta, aumentare `training.epochs` oltre l'epoca salvata e usare:

```powershell
python -m tesi_m3d.train `
  --config configs\train_pix2pix_baseline.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda --resume outputs\train_pix2pix_baseline\checkpoint_epoch006.pt
```

I checkpoint creati prima di questa modifica contengono solo i pesi: sono
validi per inferenza e possono riprendere il modello, ma l'optimizer verrà
inizializzato di nuovo.

Il checkpoint migliore (`best.pt`) può produrre direttamente una heatmap e una
decisione volume-level basata sul massimo della heatmap:

```powershell
python -m tesi_m3d.inference `
  --checkpoint outputs\train_pix2pix_baseline\best.pt `
  --volume-dir "C:\Tesi Magistrale Piscopo\pix2pix\scan\<img_id>" `
  --device cuda --out outputs\heatmap.npy
```

La validation della baseline conserva tutte le patch positive di ciascun
volume e campiona solo le negative rimanenti. Al termine del training viene
creato `calibration.json` accanto a `best.pt`: contiene una soglia per la
decisione volume-level e una separata per la maschera di localizzazione. La
CLI le carica automaticamente; `--threshold <valore>` sostituisce solo la
soglia di detection. Per scegliere esplicitamente il file della maschera usare
`--mask-out outputs\heatmap_mask.npy`.

```powershell
python -m tesi_m3d.train `
  --config configs\train_pix2pix_test_cycle_diffusion.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda
```

Questa run:

- legge split e record da `metadata\m3dsynth`;
- usa pix2pix positivo e CT reali negativi del solo split train;
- estrae patch `32x32x32` con stride train `32`;
- esclude patch boundary con overlap positivo inferiore al 5%;
- bilancia classi con `WeightedRandomSampler`;
- allena il classificatore 3D e salva il checkpoint.

**Progresso e memoria**: Il comando stampa barre di progresso (tqdm). Il primo epoch
carica le serie in RAM (~75 GB peak). I successivi riutilizzano i dati cachati (molto
più veloce). Tempo atteso: ~7-15 ore per epoch, ~50 epoch = 350 ore = ~15 giorni.

Output:

```text
outputs\train_pix2pix_test_cycle_diffusion\patch3d_classifier.pt
```

Al momento il comando di training non esegue ancora la valutazione finale sui
due generatori indicati nel nome. Il checkpoint pix2pix serve a validare
architettura, accesso ai dati, patch extraction e uso GPU. La valutazione
cross-generator sarà eseguita quando cycle e diffusion saranno presenti.

## 14. Errori comuni

### `metadata directory not found`

Eseguire il comando dalla root `TesiMagistralePiscopo` e verificare:

```powershell
ls metadata\m3dsynth
```

Devono esserci `data.csv`, `sets.csv`, `centers.csv`, `LIDC.csv`.

### `DICOM series not found`

Controllare che `--dicom-root` termini con `Reale\lidc_idri` e che
`--download-metadata` indichi il manifest associato a quel download.

### `does not look like the dataset root: no 'pix2pix' directory here`

`--output-root` punta alla cartella sbagliata, tipicamente quella del progetto.
Indicare la dataset root, cioè `C:\Tesi Magistrale Piscopo`, oppure omettere il
parametro e lasciare che venga rilevata. Il flag `--allow-any-root` disattiva il
controllo, ma le serie convertite fuori dalla dataset root non vengono trovate
dal training.

### `no TIFF slices found`

Controllare `--data-root`. Deve essere la cartella che contiene direttamente
`pix2pix` e `real`, quindi `C:\Tesi Magistrale Piscopo`. Se la conversione è
stata lanciata con un `--output-root` sbagliato, `real\scan` si trova altrove:
spostarla accanto a `pix2pix` oppure rilanciare la conversione, che salta le
serie già complete.

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
### Baseline con campionamento e score volume migliorati

La configurazione `train_pix2pix_baseline.yaml` seleziona il 67% dei volumi di
ogni epoca tra quelli manipolati, senza duplicare patch nello stesso batch. Il
dry-run riporta anche `positive_volumes_per_epoch`,
`positive_patches_per_epoch` e `positive_patch_fraction`.

La calibrazione confronta `max` con le medie top-k configurate e salva tutti i
risultati in `calibration.json > classification > candidates`; lo score con
AUC validation migliore viene usato dall'inferenza. In
`outputs\train_pix2pix_baseline\validation_report` vengono inoltre salvati due
esempi TP, FP, TN e FN con pannelli CT, heatmap, mask reale e predizione.

Prima di una nuova run cancellare o rinominare esclusivamente la vecchia
cartella `outputs\train_pix2pix_baseline`, quindi eseguire prima il dry-run e
poi il training senza `--resume`.
