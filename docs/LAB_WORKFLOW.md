# Workflow PC laboratorio Windows

Guida  per configurare la macchina, validare i dati e avviare gli
esperimenti.

Questo branch sostituisce la CNN compatta con **ResNet-18 3D inizializzata con
MedicalNet (23 dataset)**. La testa binaria viene addestrata sul task di
manipolazione; tutto l’encoder viene aggiornato durante il fine-tuning.
Non esiste un’opzione per avviare ResNet con pesi casuali.

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
C:\Tesi Magistrale Piscopo\Reale\metadata.csv    manifest download IDC
C:\Tesi Magistrale Piscopo\pix2pix\Scan          TIFF manipolati
C:\Tesi Magistrale Piscopo\pix2pix\label         mask TIFF
```

## Attiva .venv

```powershell
.\.venv\Scripts\Activate.ps1
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

## 5. Pesi MedicalNet e test senza dataset

Dalla root del repository, nell'ambiente Python attivo, installare anche le
nuove dipendenze se l'ambiente esisteva già:

```powershell
python -m pip install -e ".[dev,train,conversion]"
```

Scaricare una sola volta il checkpoint ufficiale (circa 132 MB):

```powershell
New-Item -ItemType Directory -Force checkpoints\medicalnet | Out-Null
Invoke-WebRequest `
  -Uri "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/758fe28/resnet_18_23dataset.pth" `
  -OutFile checkpoints\medicalnet\resnet_18_23dataset.pth
Get-FileHash checkpoints\medicalnet\resnet_18_23dataset.pth -Algorithm SHA256
```

SHA256 atteso:
`61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4`.
Il codice verifica automaticamente questo hash e carica **tutti** i pesi
dell'encoder con controllo rigoroso delle chiavi. File assente, corrotto o di
un'altra versione: errore prima dell'indicizzazione del dataset. Non viene
avviato un training casuale e non vengono ignorati pesi incompatibili.
Il percorso si può cambiare in `model.medicalnet_weights` nel file YAML;
i percorsi relativi sono risolti dalla directory di esecuzione.

Fonti: [MedicalNet](https://github.com/Tencent/MedicalNet),
[checkpoint ufficiale](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/blob/main/resnet_18_23dataset.pth).
I blocchi ResNet sono forniti da MONAI; shortcut A, stride iniziale 2 e stadi
finali con dilatazione 2/4 riproducono l'encoder MedicalNet originale.

```powershell
python -m pytest -q -p no:cacheprovider
python scripts\smoke_synthetic_training.py --device cuda
```

I test unitari non richiedono dataset esterni o download: usano fixture controllate per
verificare caricamento, errori e ripristino. Lo smoke usa invece i pesi
MedicalNet reali, esegue due passi di fine-tuning e produce:

- loss e gradienti finiti;
- `outputs\synthetic_smoke_resnet18_medicalnet\synthetic_patch3d.pt`;
- `outputs\synthetic_smoke_resnet18_medicalnet\synthetic_heatmap.npy`, shape `(64, 64, 64)`.

Per lo smoke su CPU omettere `--device cuda`. Questa verifica controlla la
pipeline, non misura la qualità scientifica della localizzazione.

## 6. Ruolo dei CSV

### CSV ufficiali `metadata\m3dsynth`

- `data.csv`: collega `img_id`, generatore, `orig_id`, serie e coordinate.
- `sets.csv`: assegna ciascun `orig_id` a train, valid o test evitando leakage.
- `centers.csv`: centri di crop usati dalle baseline ufficiali.
- `LIDC.csv`: 744 serie DICOM reali richieste e relativo identificatore `sdir_id`.

La pipeline usa direttamente `data.csv` e `sets.csv`; la conversione usa
anche `LIDC.csv`. `centers.csv` resta disponibile per confronti con il protocollo
ufficiale.

## 7. Perché convertire i reali in TIFF?

Il loader di training riceve stack TIFF sia per pix2pix sia per i negativi
reali. La conversione serve per:

- fornire esempi reali con lo stesso contenitore dei manipolati;
- mantenere ordinamento delle slice e conversione `uint16` di M3Dsynth;
- applicare la stessa normalizzazione percentile a entrambe le classi.


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

Verificare che sia apparsa una directory simile a:

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

L'output completo viene scritto su `outputs/check_real_tiff.log`. 

Cosa verifica, per ogni serie:

- ogni `slide*.tiff` si apre, è `uint16` e ha la stessa shape delle altre;
- la numerazione è contigua e coincide con il conteggio scritto in `.complete`;
- nessuna slice è costante, e il volume contiene tessuti molli e osso nelle
  proporzioni attese per una CT del torace;
- l'orientamento è corretto: aria fuori dal paziente ai bordi, corpo al centro
  dell'inquadratura, colonna vertebrale nella metà posteriore (nessun flip
  verticale) e aria polmonare massima a metà stack (ordinamento z corretto).

Le soglie sono espresse in HU. L'offset `uint16` non è ricavabile dal TIFF, per
cui viene stimato dal picco dell'aria polmonare,
perché `scan_to_uint16` trasla il volume di `-min(scan)` e quel minimo è il
padding fuori field, non l'aria. L'offset risulta quindi diverso da serie a
serie (2048 per `LIDC-IDRI-0003`, 3072 per i TIFF pix2pix). Questo viene gestito in quanto: `normalize_percentile` scarta i voxel a zero e riscala tra due
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

Con `--preview-dir`, il controllo dei TIFF reali scrive anche, per ogni serie,
un montaggio di 9 slice assiali più una ricostruzione coronale e una sagittale
in PNG già finestrate sul polmone: le due ricostruzioni aiutano a vedere slice
duplicate o fuori ordine. Usare `--limit N` per controllare solo le prime N
serie.

### Audit intensità reale vs pix2pix

Prima di attribuire al modello le differenze tra classi, confrontare coppie con
lo stesso `img_id`, quindi la stessa anatomia sorgente:

```powershell
python -m tesi_m3d.audit_intensity `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --metadata-dir metadata\m3dsynth `
  --split train --max-pairs 32 `
  --out-dir outputs\intensity_audit_train
```

`volumes.csv` contiene intensità, percentili, offset stimato e statistiche dopo
la stessa normalizzazione p1-p99 del training. `pairs.csv` mostra le differenze
caso per caso. `summary.json` confronta le distribuzioni normalizzate tramite
divergenza Jensen-Shannon, distanza KS e differenza media standardizzata.
`no_large_shift_detected` permette di scartare una grossa scorciatoia di
intensità, ma non dimostra che i due domini siano identici. L'offset grezzo può
essere diverso senza essere un problema: il controllo decisivo è quello dopo
normalizzazione.

### Audit locale dentro e fuori dalla mask

Per cercare differenze piccole che l'audit globale può diluire, iniziare con
una sola coppia e poi usare le stesse 32 coppie (`--seed 21`) dell'audit sopra:

```powershell
python -m tesi_m3d.audit_local_artifacts `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --metadata-dir metadata\m3dsynth `
  --split train --max-pairs 1 `
  --out-dir outputs\local_artifact_audit_smoke

python -m tesi_m3d.audit_local_artifacts `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --metadata-dir metadata\m3dsynth `
  --split train --max-pairs 32 `
  --out-dir outputs\local_artifact_audit_train
```

`regions.csv` contiene, per coppia, le differenze assolute e di gradiente
sul piano assiale nella mask, nei quattro voxel vicini per asse e lontano da
essa. Fuori dalla mask si ignorano i voxel zero in uno dei due TIFF.
`summary.json` riassume le medie delle coppie; `previews/` mostra la slice con
più voxel annotati e, se diversa, quella con la maggiore differenza totale
fuori dalla mask, usando la stessa finestra per reale e pix2pix. Il pannello
delle differenze è riscalato per essere leggibile: usare i valori del CSV per
confrontare quantitativamente coppie diverse. Differenze fuori dalla mask
non dimostrano da sole che la rete le stia usando: vanno confrontate con gli
score delle patch.

## 11. Allineamento automatico scan/mask z-dimension

Il dataset pix2pix contiene disallineamenti nella dimensione z (profondità):
- **Caso pix2pix**: ogni serie ha label con **+1 slice** (padding accidentale)
- **Caso generale**: scan e mask possono avere z-dimensioni diverse (fino a 85+ slice)

**Soluzione implementata**: il loader (`load_scan_and_mask` in `dataset.py`) allinea
automaticamente:
- Se mask è **più lunga**: trimma le slice extra dalla fine (convenzione pix2pix)
- Se mask è **più corta**: aggiunge padding con zeros alla fine (background)
- Se y/x non matchano: **errore critico** (non tollerato)

## 12. Verifica struttura pix2pix

La root dataset è `C:\Tesi Magistrale Piscopo`.
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


## 14. Esperimento ResNet-18 MedicalNet: pix2pix

La configurazione di partenza è
[`train_pix2pix_balanced_patches.yaml`](../configs/train_pix2pix_balanced_patches.yaml).
Le patch restano `32x32x32`, stride train 32 e inferenza 16, soglia positiva 5%,
8 positive e 24 negative nei batch manipolati. Sono mantenuti seed 21, massimo
256 record train, 64 validation, 4096 patch per epoca e 16 epoche. La loss è
focal con target binario; il learning rate resta `3e-4`.

Il cambio riguarda encoder e inizializzazione. Preprocessing, dati e sampling
restano quelli della balanced, per evitare di mescolare questa prova con
resampling isotropico, target soft o hard-negative mining.

```yaml
model:
  name: resnet18_medicalnet
  in_channels: 1
  num_classes: 1
  medicalnet_weights: checkpoints/medicalnet/resnet_18_23dataset.pth
  dropout: 0.2
```

Tutte le configurazioni del branch usano MedicalNet. Ogni `output_dir` termina
con `_resnet18_medicalnet`, così i risultati storici della CNN non vengono
sovrascritti. La run balanced scrive in:

```text
outputs\train_pix2pix_balanced_patches_resnet18_medicalnet\
```

### Dry-run e training

```powershell
python -m tesi_m3d.train `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda --dry-run

python -m tesi_m3d.train `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda
```

Il dry-run carica e verifica MedicalNet, costruisce l'indice e riporta il
campionamento, ma non esegue forward/backward: lo smoke del punto 5 verifica
questi ultimi. Avviare la prima run senza `--resume` e senza `--init-checkpoint`.
La testa finale è nuova; i pesi dell'encoder provengono obbligatoriamente da
MedicalNet. Non usare i checkpoint della vecchia CNN su questo branch:
le architetture sono incompatibili. Gli output storici restano consultabili,
e il vecchio branch consente di riprodurre quella pipeline.

Monitorare memoria e utilizzo GPU da un secondo terminale:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1
```

ResNet richiede più memoria e tempo della CNN compatta. Il batch iniziale
resta 32. Se la GPU esaurisce la memoria, ridurre `training.batch_size` a 8
mantenendo `patches.max_patches_per_volume: 32`; annotare questa variazione
nel confronto. Per calibrazione e inferenza si può ridurre separatamente
`evaluation.calibration_batch_size` o passare `--batch-size 8` alle CLI.

### Checkpoint e ripresa

`metrics.csv` contiene loss train/validation e AP/AUC patch campionate.
`best.pt` è scelto con la migliore `valid_ap`; `patch3d_classifier.pt` è
l'ultimo checkpoint, che può essere diverso. Usare **best.pt** nelle
valutazioni successive. La scelta del checkpoint conserva il criterio
precedente: AP campionata e AP densa misurano distribuzioni diverse.

Per riprendere una run interrotta:

```powershell
python -m tesi_m3d.train `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda `
  --resume outputs\train_pix2pix_balanced_patches_resnet18_medicalnet\checkpoint_epoch006.pt
```

La ripresa recupera encoder, testa, optimizer, epoca e AMP scaler. Il
checkpoint completo contiene tutti i pesi: ripresa, inferenza e valutazione
non richiedono il file MedicalNet originale. Un training nuovo lo richiede.
Per proseguire oltre l'ultima epoca configurata, aumentare `training.epochs`.

Con circa 33 milioni di parametri, un checkpoint con stato AdamW occupa circa
400 MB. Conservare 16 epoche più best e ultimo richiede circa **7 GB**, oltre
a cache e report. Se C: ha poco spazio, passare al training
`--output-dir "D:\Esperimenti Tesi\balanced_resnet18_medicalnet"` e usare
quella directory in tutti i successivi comandi di ripresa e valutazione.
In alternativa, `training.checkpoint_every_epoch: false` conserva soltanto
best e ultimo; per riprendere usare `patch3d_classifier.pt` quando disponibile.

## 15. Valutazione della localizzazione sulla validation

Al termine del training, `calibration.json` contiene soglie distinte per
detection e localizzazione, calibrate sulla validation, e i report visuali
sono in `validation_report`. Valutare anche gli score sulla griglia densa:

```powershell
python -m tesi_m3d.evaluate_patch_level `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --checkpoint outputs\train_pix2pix_balanced_patches_resnet18_medicalnet\best.pt `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --split valid --device cuda --batch-size 8
```

Confrontare con i report storici in `outputs\train_pix2pix_balanced_patches`:

- top-1/top-3/top-5 hit e distanza della top-1 dalla mask;
- AP, precision e recall patch-level sulla griglia densa;
- correlazione score/copertura e posizione delle patch più votate nelle immagini;
- Dice/F1 e IoU della heatmap in `calibration.json`;
- detection volume-level, distinta dalla localizzazione.

Le immagini in `patch_level_report\images` mostrano centro della mask e
prime cinque patch. Non considerare sufficiente un aumento della sola AUC.
Il registro storico della CNN è
[`REGISTRO_DIAGNOSI_LOCALIZZAZIONE.md`](REGISTRO_DIAGNOSI_LOCALIZZAZIONE.md):
i suoi numeri non descrivono ResNet.

Per le metriche voxel dell'intero volume e della ROI polmonare:

```powershell
python -m tesi_m3d.evaluate_paper_protocol `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --checkpoint outputs\train_pix2pix_balanced_patches_resnet18_medicalnet\best.pt `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --split valid --mods pix2pix --device cuda
```

Questo comando valuta tutte le scansioni manipolate dello split richiesto;
la calibrazione e il report patch usano invece il sottoinsieme di validation
configurato. Confrontare risultati ottenuti sugli stessi insiemi. La massima
balanced accuracy per volume è una misura con soglia ottimizzata sulla sua
mask, non una soglia operativa da usare su nuove scansioni.

## 16. Inferenza su una CT

```powershell
python -m tesi_m3d.inference `
  --checkpoint outputs\train_pix2pix_balanced_patches_resnet18_medicalnet\best.pt `
  --volume-dir "C:\Tesi Magistrale Piscopo\pix2pix\scan\<img_id>" `
  --device cuda --batch-size 8 `
  --out outputs\train_pix2pix_balanced_patches_resnet18_medicalnet\heatmap.npy
```

Sostituire `<img_id>` con un volume esistente. La CLI usa le soglie del
`calibration.json` accanto al checkpoint e salva heatmap e mask binaria.
`--threshold` sostituisce soltanto la soglia di detection.
Il modello produce sempre uno score per patch: il pretraining non lo
trasforma in un segmentatore voxel-wise o in un modello MIL.

## 17. Test cross-generator

Dopo aver fissato configurazione e checkpoint sulla validation, valutare il
**medesimo best.pt** su cycle/diffusion quando i TIFF sono disponibili:

```powershell
python -m tesi_m3d.evaluate_paper_protocol `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --checkpoint outputs\train_pix2pix_balanced_patches_resnet18_medicalnet\best.pt `
  --data-root "D:\Dataset Tesi Piscopo" `
  --split test --mods cycle diffusion --device cuda
```

Se è disponibile soltanto Diffusion, usare `--mods diffusion`. Per il report
patch-level con i reali conservati sull'altro disco:

```powershell
python -m tesi_m3d.evaluate_patch_level `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --checkpoint outputs\train_pix2pix_balanced_patches_resnet18_medicalnet\best.pt `
  --data-root "D:\Dataset Tesi Piscopo" `
  --real-scan-root "C:\Tesi Magistrale Piscopo\real\scan" `
  --split test --mods diffusion --device cuda --batch-size 8 `
  --max-report-images 20 `
  --out-dir outputs\train_pix2pix_balanced_patches_resnet18_medicalnet\patch_level_report_test_diffusion
```

AP, AUC e hit rate non dipendono dalla soglia. Il report patch attuale
riottimizza precision/recall/F1 sullo split valutato: sul test questi valori
sono descrittivi, non prestazioni ottenute con una soglia fissata in
validation. Non scegliere un nuovo checkpoint usando i risultati di test.

Conservare YAML, `metrics.csv`, `calibration.json` e report densi della run.
La prima domanda è se migliora la localizzazione su pix2pix; la seconda è
quanto di quel miglioramento si mantiene sui generatori esclusi dal training.
