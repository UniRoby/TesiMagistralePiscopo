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

Il numero di record disponibili e il numero di patch elaborate per epoca sono
due limiti distinti. Il loader non tiene tutte le CT in RAM:

- costruisce una volta l'indice delle patch leggendo le mask, poi lo salva
  nella cache su disco;
- calcola p1/p99 una sola volta per scansione e li salva in una piccola cache
  JSON su disco, riusata dopo l'evizione dalla RAM e ai successivi avvii;
- legge e normalizza una CT quando serve; la cache RAM conserva due volumi
  per worker (`volume_cache_size: 2`);
- raggruppa le patch della stessa scansione in batch, riusando il volume
  caricato invece di decodificare tutti i TIFF per ogni singola patch;
- usa i worker persistenti indicati nel YAML, prefetch di un batch per worker, memoria pinned
  e trasferimenti non bloccanti verso CUDA.

La cache RAM è locale al worker e limitata: una scansione può essere riletta
nelle epoche successive. Ogni miss richiede ancora la lettura del volume
completo; non va confuso con un caricamento limitato alle sole 32 patch.

La cache dei percentili è attiva automaticamente nel training, nella
validation, nella calibrazione finale e nel benchmark. Usa
`<cache-dir>/percentiles/*.json`: per il full training il percorso predefinito è
`outputs/train_pix2pix_full_train_resnet18_medicalnet/cache/percentiles`.
`--cache-dir`, il campo YAML `cache_dir` e `TESI_M3D_CACHE_DIR` mantengono la
precedenza già prevista per il training. Nel benchmark la cache è invece
in `<out-dir>/index_cache/percentiles`, condivisa fra epoche e scenari.
Per condividerla fra run di training usare lo stesso `--cache-dir`.

Ogni voce contiene solo i due limiti di normalizzazione: non salva copie
delle CT e non scrive nelle directory del dataset. La chiave comprende il
percorso fisico della scansione, forma e dtype, parametri dei percentili,
dimensione e timestamp di tutte le sezioni lette. Modificare una sezione o i
parametri fa ricalcolare i limiti; i record reali che puntano alla stessa CT
condividono la voce. I file sono pubblicati con sostituzione atomica per
supportare più worker, anche su Windows. Una voce corrotta viene ricostruita;
un errore di scrittura genera un warning e il caricamento prosegue senza
salvare la voce. Non modificare i TIFF mentre il caricamento è in corso.

Il calcolo resta esattamente quello precedente: percentili sui voxel
strettamente positivi, fallback sull'intero volume se sono tutti zero,
riscalatura float32 e nessun clipping. Questa modifica può essere usata con
i checkpoint esistenti e non richiede un nuovo training per compatibilità.
Le CLI di analisi/inferenza autonome che non ricevono una directory cache
continuano a calcolare gli stessi percentili senza salvarli.

HexMIL include tutti i record train dei generatori selezionati, ma riduce il
lavoro spaziale: SliceMIL legge le sezioni selezionate vicino a `coord_z`,
mentre HexMIL legge una finestra di K sezioni. I percentili sono calcolati
sull'intera scansione la prima volta e salvati in `percentile.npy`.
Il suo sampler bilanciato usa reinserimento: tutti i record sono disponibili,
ma una singola epoca può ripeterne alcuni e saltarne altri.

Per confrontare i worker sul PC reale, il benchmark esistente misura soltanto
il caricamento. Usare prima il sottoinsieme bounded, senza eseguire ResNet:

```powershell
python -m tesi_m3d.benchmark_loading `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --workers 0 2 --cache-sizes 2 --orders grouped --epochs 2 `
  --out-dir outputs\loading_benchmark_resnet18_medicalnet
```

Il benchmark costruisce comunque modello e indice all'avvio, quindi richiede
il file MedicalNet; nelle epoche misurate non esegue forward/backward.
La GPU resta inutilizzata: `nvidia-smi` può mostrare 0 MiB. Durante la lettura
viene mostrata una barra dei batch per epoca, seguita da tempo, patch/s e
numero di caricamenti dei volumi. Con `workers=0` la lettura è sequenziale.
Confrontare `patches_per_second` e `volume_reads` in `epochs.csv`. Aumentare i
worker soltanto se migliora il throughput e la RAM resta sufficiente. Il tempo
di training completo comprende anche GPU, validation e calibrazione.

Per misurare il guadagno della cache dei percentili sul PC del laboratorio,
usare una cartella nuova, gli stessi parametri del benchmark precedente e
solo quattro worker:

```powershell
python -m tesi_m3d.benchmark_loading `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --workers 4 --cache-sizes 2 --orders grouped --epochs 2 `
  --out-dir outputs\benchmark_percentile_cache
```

La prima epoca popola la cache, la seconda la riusa. Un secondo avvio con lo
stesso `--out-dir` parte con i percentili già salvati (e sovrascrive il report
del benchmark precedente). I tempi dipendono anche dalla cache filesystem
di Windows: confrontare il secondo passaggio con quello del benchmark
precedente, senza interpretare tutta la differenza fra epoca 1 e 2 come
effetto dei soli percentili. `volume_reads` può restare invariato: conta i
caricamenti completi, anche quando i limiti sono già disponibili su disco.
Per forzare il ricalcolo basta rimuovere la sola sottocartella `percentiles`;
`--rebuild-index` ricostruisce invece l'indice delle patch e non cancella i
percentili. Le vecchie voci invalidate possono essere eliminate nello stesso
modo. Misurare prima il loader: il risparmio sul tempo totale di training non
è garantito se il collo di bottiglia prevalente resta la decodifica TIFF.


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

### Training con tutti i record train

Per usare tutto lo split train pix2pix + real, scegliere
[`train_pix2pix_full_train.yaml`](../configs/train_pix2pix_full_train.yaml).
Questa configurazione rimuove il limite di 256 record e quello di 4096 patch
per epoca. Visita tutte le scansioni fisiche indicizzate ogni epoca,
campionando 32 patch per scansione; le patch scelte cambiano con l'epoca.
Non elabora in ogni epoca tutte le milioni di patch della griglia.

```yaml
data:
  max_train_records: null
  max_valid_records: 64
patches:
  max_patches_per_volume: 32
  positive_volume_fraction: null
training:
  max_patches_per_epoch: null
```

I metadata contengono 1537 record pix2pix e 1787 reali di training. Con i reali
salvati dal convertitore come `orig_id__sdir_id`, i record reali corrispondono
a 489 scansioni: l'atteso è quindi circa **2026 scansioni, 2026 batch e 64832
patch per epoca**, con batch 32. Il numero effettivo riportato dal dry-run fa
fede per il layout TIFF presente sul PC. I record che puntano allo stesso
volume reale sono raggruppati, non richiedono batch duplicati.

```powershell
python -m tesi_m3d.train `
  --config configs\train_pix2pix_full_train.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda --dry-run

python -m tesi_m3d.train `
  --config configs\train_pix2pix_full_train.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda
```

Nel dry-run controllare che `batches_per_epoch` coincida con `n_volumes`
e che `patches_per_epoch` sia `n_volumes * 32`. La prima indicizzazione è più
costosa del sottoinsieme e viene riutilizzata dai successivi avvii.
Se mancano TIFF richiesti dal training, il loader segnala un errore.

Modello, loss, learning rate, numero di epoche e sampling delle patch restano
quelli della balanced. Il sottoinsieme di validation resta fissato a 64 record
per mantenere confrontabile la selezione dei checkpoint con la run precedente;
non diventa una validation densa. Cycle/diffusion e i pazienti di validation e
test restano esclusi dall'allenamento.

La run parte dai pesi MedicalNet, senza `--resume`, e salva in
`outputs\train_pix2pix_full_train_resnet18_medicalnet`. Per valutarla, usare i
comandi dei punti 15–17 sostituendo il file YAML con
`configs\train_pix2pix_full_train.yaml` e la directory del checkpoint con
quella della nuova run. Conservare i risultati della run bounded separati.

Ogni epoca esegue circa 16 volte più batch del precedente limite di 128:
è un rapporto di lavoro, non una stima dei tempi sul PC del laboratorio.

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

## 18. Recupero della localizzazione: segmentazione MedicalNet

Questo percorso affianca il classificatore: apprende la mask della
**manipolazione**, non il contorno clinico del tumore. Encoder ResNet-18
MedicalNet obbligatorio, decoder con skip e GroupNorm, output voxel della
stessa forma dell'input. Le 17 BatchNorm dell'encoder mantengono fisse le
statistiche anche in training; i parametri affini vengono affinati dopo il
warmup. Non usare i checkpoint del classificatore per inizializzare il decoder.

Prerequisiti: ambiente e pesi MedicalNet della sezione 14. Non sono richieste
nuove dipendenze. Dalla root del repository, impostare i percorsi del PC:

```powershell
$ctRoot = "C:\Tesi Magistrale Piscopo"
$realCtRoot = Join-Path $ctRoot "real\scan"
$baselineCheckpoint = "outputs\train_pix2pix_full_train_resnet18_medicalnet\best.pt"
```

Se pix2pix/cycle/diffusion sono su D: e i reali su C:, impostare invece
`$ctRoot = "D:\Dataset Tesi Piscopo"` e
`$realCtRoot = "C:\Tesi Magistrale Piscopo\real\scan"`.
Usare gli stessi percorsi in preparazione e training: un cambio richiede
un nuovo audit. La calibrazione/inferenza del modello usa il preprocessing
del checkpoint e non accetta override delle trasformazioni o della geometria.

### 18.1 Audit e manifest fissi

```powershell
python -m tesi_m3d.prepare_segmentation `
  --config configs\segmentation_pilot.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot `
  --out-dir outputs\segmentation_protocol --stage pilot
```

Salva `manifest.json`, `audit.json` e `paired_artifacts/`. L'audit iniziale
copre il train pilota e la validation comune; prima del full verrà esteso a
tutto il train. Controlla ogni header CT, sequenze TIFF senza buchi, mask non
vuote e compatibilità spaziale. Il taglio delle sezioni finali è consentito
solo se non elimina voxel positivi. Errori di forma, sequenza o annotazione
bloccano il protocollo: non corregge orientamenti/traslazioni e non modifica
gli originali. Le coordinate metadata sono riportate, non usate in inferenza.

Esaminare `records` nell'audit e le immagini dei tre piani in
`paired_artifacts/previews/`: l'audit appaia 16 injection e 16 removal del
train con i relativi reali. `bbox_fill_fraction` vicino a 1 suggerisce
un'annotazione a riquadro e richiede interpretazione visiva; non dimostra
un contorno anatomico. Differenze fuori dalla mask sono diagnostiche, mai
ground truth sostitutiva. Le verifiche automatiche non certificano da sole
la correttezza anatomica delle annotazioni.

I manifest contengono 6 casi overfit, 64 train pilota (24 inj, 24 rem, 16 real),
8 validation pilota (2 inj, 2 rem, 4 real), 16 validation full (4 inj, 4 rem,
8 real), tutte le 2026 scansioni fisiche train e le 58 validation comuni.
I sottoinsiemi sono annidati, deduplicati e separati per paziente; non
modificarli dopo aver confrontato i risultati.

### 18.2 Verificare l'apprendimento su sei scansioni

```powershell
python -m tesi_m3d.train_segmentation `
  --config configs\segmentation_overfit.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda
```

Massimo 300 aggiornamenti, crop fissi senza augmentation. In questa sola
fase una cache da sei scansioni e zero worker evita riletture ripetute.
Il risultato in `outputs/segmentation_overfit/gate.json` deve avere
`passed: true`: Dice dei crop positivi almeno 0,90, frazione di voxel
falsamente positivi sui crop negativi inferiore a 0,01 e gradienti finiti.
Sono misure sul train che verificano l'apprendimento, non generalizzazione.
Se fallisce, ispezionare le immagini prodotte e fermarsi: i piloti verranno
bloccati. Exit code 2 indica criterio scientifico non superato; leggere
il report, senza aumentare il budget del full.

Se il gate fallisce, rivalutare `last.pt` senza altri aggiornamenti:

```powershell
python -m tesi_m3d.train_segmentation `
  --diagnose-checkpoint outputs\segmentation_overfit\last.pt `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --output-dir outputs\segmentation_overfit\diagnostics
```

Usa manifest, preprocessing, seed e crop fissi del checkpoint. Non aggiorna
i pesi, non richiede nuovamente il file MedicalNet e non riscrive `gate.json`.
`diagnostics/report.json` confronta il gate in float32 con bfloat16 su CUDA,
la precisione del training. Non calibra soglie: entrambe restano a 0,5.
Ogni directory `fp32/` e `bfloat16/` contiene `crops.csv` (tutti i 48 crop),
`cases.csv` e figure del primo crop, del positivo con Dice minimo e del
negativo con più falsi positivi per scansione. I nomi includono `real` o
`pix2pix`: i reali conservano identificativi `inj_*`/`rem_*` ma hanno
correttamente target vuoto. Le metriche non definite sono celle vuote;
il Dice dei crop negativi non contribuisce alla media sui positivi.
Su CPU si produce soltanto float32. Copiare questi piccoli report e le figure
per l'analisi; i checkpoint possono rimanere sul PC del laboratorio.

Le nuove run salvano anche `overfit_metrics.csv` ogni 25 aggiornamenti e al
limite configurato. La prima run non conserva questa storia: la sola curva
della loss non dice quando Dice e falsi positivi siano stati migliori.
La diagnosi non sostituisce il gate e non sblocca i piloti.

La prima diagnosi ha confermato risultati quasi identici in float32 e
bfloat16 (Dice 0,70994 e 0,70982). Sui positivi il recall medio è 94,22%
ma la precision media è 61,04%; prevale l'eccesso di predizioni positive.
La loss continuava a scendere al limite dei 300 aggiornamenti. È stata
eseguita **una replica controllata**, per verificare se il decoder converge
meglio nello stesso budget, cambiando soltanto il suo LR da 3e-4 a 1e-3.
Il comando seguente documenta la prova, che è già risultata negativa:

```powershell
python -m tesi_m3d.train_segmentation `
  --config configs\segmentation_overfit_decoder_lr.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda
```

Avviare da MedicalNet e dallo stesso seed, **senza `--resume`**. Tutti gli
altri parametri, i crop, il manifest e il gate restano quelli originali;
non è un'estensione della run a 600 aggiornamenti. Gli output separati sono
in `outputs/segmentation_overfit_decoder_lr/`: confrontare `gate.json`,
`overfit_metrics.csv`, `cases.csv`, `crops.csv` e le immagini dei peggiori
errori. Un LR maggiore è un'ipotesi sperimentale, non una causa già dimostrata
né una correzione garantita. Se fallisce, fermarsi e documentare l'esito;
non avviare una serie di tentativi arbitrari o il full. Le configurazioni
pilota continuano a richiedere il gate originale e non vengono sbloccate
automaticamente da questa prova. Se passa, occorre riesaminare il risultato
prima di collegarne il report ai piloti.

**Esito della replica: non procedere ai piloti.** A 300 aggiornamenti il
Dice medio è 0,51008 e gli FP nei negativi sono 0,17503%. Il recall medio
dei positivi scende dal 94,22% della prima prova al 41,02%: gli FP minori
si accompagnano alla perdita di gran parte del target. Il miglior Dice
nei controlli salvati è 0,73410 a 275 aggiornamenti, con FP 1,19321%:
non supera il gate. Nelle due run già concluse `best.pt` veniva sovrascritto
a ogni epoca e coincide con l'ultimo checkpoint; non contiene il modello a 275
aggiornamenti. Non tentare di recuperarlo dal solo CSV né aumentare LR,
epoche o soglie per sbloccare il protocollo. La prossima fase richiede
debug del training (componenti della loss, gradienti e apprendimento di
crop fissi isolati), senza promuovere nessuna delle due run a modello
validato. Vedere la sezione 10.3 del registro per i conteggi e i limiti.

### 18.2.1 Debug guidato dopo i due overfit falliti

Per eseguire i quattro controlli senza modificare a mano il trainer:

```powershell
python -m tesi_m3d.debug_segmentation `
  --config configs\segmentation_overfit.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --output-dir outputs\segmentation_debug
```

Usare il config originale, con LR decoder 3e-4. Il comando legge il manifest
già sottoposto ad audit, non riprende i checkpoint delle run fallite e non
modifica i loro output. Si arresta automaticamente al primo controllo
fallito; exit code 2 indica che il criterio diagnostico non è superato.
Un errore tecnico è scritto in `report.json` se avviene durante le fasi e
viene anche mostrato nel terminale. Una directory di output non vuota viene
rifiutata: per una nuova esecuzione scegliere un altro `--output-dir`.

1. **Aggiornamenti effettivi.** Nel primo aggiornamento sul positivo
   verifica norme dei gradienti e delle variazioni dei pesi >0 per encoder
   e decoder, valori finiti e statistiche BatchNorm immutate. I dettagli
   sono in `positive/update_check.json`; gli stessi controlli vengono
   ripetuti all'inizio della fase con il negativo.
2. **Un positivo fisso.** Memorizza il crop 0 della prima scansione
   manipolata del manifest (attualmente `inj_2092`), senza augmentation,
   per massimo 300 aggiornamenti. Procede solo con Dice >=0,90 a soglia 0,5.
3. **Positivo e negativo fissi.** Riparte da MedicalNet e dallo stesso seed
   21, aggiungendo il crop 4 della stessa scansione, verificato privo di
   foreground. Massimo altri 300 aggiornamenti; richiede Dice positivo
   >=0,90 e FP voxel sul negativo <1%, alla stessa soglia 0,5.
4. **Sei scansioni.** Solo se entrambi i test isolati passano, riparte ancora
   da MedicalNet e seed 21 con il trainer originale, gli otto crop per
   scansione e massimo 300 aggiornamenti. Non avvia piloti o full.

Ogni fase usa la stessa loss 0,5 BCE + 0,5 Dice, AdamW e LR del config.
Il batch effettivo dei test isolati è rispettivamente 1 e 2 crop, perché
servono a isolare l'apprendimento; questi risultati non sono un confronto
scientifico con il training su sei scansioni. Le fasi sono indipendenti:
non proseguono dai pesi adattati nella fase precedente. Il caricamento
MedicalNet e il controllo SHA256 rimangono obbligatori; non c'è fallback
a encoder casuale. Il massimo complessivo è 900 aggiornamenti distribuiti
su tre inizializzazioni, con arresto anticipato ai gate.

Leggere prima **`outputs/segmentation_debug/report.json`**:

- `stopped_at: positive_update_check` o `positive_negative_update_check`:
  gradienti, aggiornamenti o BatchNorm non superano il controllo iniziale;
- `stopped_at: positive`: non memorizza nemmeno il positivo isolato;
- `stopped_at: positive_negative`: l'aggiunta del negativo impedisce di
  rispettare entrambi i criteri;
- `stopped_at: six_scans`: i test isolati passano, il gruppo completo no;
- `passed: true`, `stopped_at: null`: tutti i controlli sono superati,
  da riesaminare prima di collegare il gate delle sei scansioni ai piloti.

In `positive/` e `positive_negative/`, `updates.csv` contiene loss, BCE,
Dice, norme dei gradienti prima del clipping e variazioni effettive dei pesi
di encoder/decoder. `checks.csv` registra le metriche dopo l'aggiornamento,
in float32, al passo 0, 1 e ogni 25 passi (oltre al limite finale).
`selection.json` nella root identifica crop, conteggi target e SHA256
degli input/target. `initial_images/` e `images/` mostrano i crop prima e
dopo il test. Le metriche non definite, come FP senza negativo, sono `null`
nel JSON e celle vuote nei CSV; non sono successi aggiuntivi.

In `six_scans/`, `update_metrics.csv` registra le componenti della loss e
gli aggiornamenti per scansione; `case_history.csv` registra Dice e FP per
caso ogni 25 aggiornamenti. Nelle **nuove** run overfit `best.pt` conserva
il controllo migliore, dando precedenza a un gate superato, poi al Dice
maggiore e infine agli FP minori. `best_gate.json` ne indica il passo;
`last.pt` e `gate.json` descrivono lo stato finale. Queste modifiche non
recuperano i checkpoint intermedi delle vecchie run. Per il resume
dell'overfit usare soltanto `last.pt`: il best può trovarsi a metà epoca.

I checkpoint dei test isolati hanno task `segmentation_debug` e non sono
modelli validati per inferenza o ammissione ai piloti. Non sostituire i
vecchi gate con il report generale del debug. Per condividere gli esiti,
sincronizzare JSON, CSV e PNG dell'intera cartella, senza i file `.pt`.
La strumentazione aggiunge lavoro: i tempi non sono un benchmark di training.

**Esito della prima esecuzione sincronizzata.** Aggiornamenti dei pesi e
BatchNorm superano i controlli; positivo e coppia passano entrambi a 50
aggiornamenti (Dice 0,91058 e 0,90422; nella coppia solo 7 voxel FP sul
negativo). Il workflow arriva alle sei scansioni e si ferma: Dice finale
0,71146, FP 1,75724%. Il best è al passo 250, Dice 0,73293 e FP 0,82042%:
anche questo non passa. Il Soft Dice degli isolati rimane circa 0,30,
quindi il superamento binario non equivale a convergenza della loss.
Non rilanciare il debug identico né i piloti. L'analisi della composizione
dei batch e dei suoi effetti sulla ponderazione della loss è nella sezione
10.5 del registro; il confronto implementato successivamente è nella sezione 18.2.2.

### 18.2.2 Confronto appaiato: batch per volume e batch misti

Il debug ha superato positivo e coppia ma fallito sulle sei scansioni.
Per isolare l'effetto della composizione degli aggiornamenti, eseguire dalla
root del repository, nell'ambiente Python attivo:

```powershell
python -m tesi_m3d.compare_segmentation_batches `
  --config configs\segmentation_overfit.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --output-dir outputs\segmentation_batch_comparison
```

Il comando esegue **due nuove run appaiate**, senza riprendere le vecchie:

- `grouped/`: otto crop della stessa scansione per aggiornamento, nello
  stesso ordine deterministico usato dall'overfit originale;
- `mixed/`: ogni aggiornamento contiene otto crop di almeno tre scansioni,
  con due o tre crop positivi e almeno un negativo di ciascuno dei due reali.

I 48 crop vengono preparati una sola volta con il sampler originale fisso,
senza augmentation, e tenuti in RAM (circa 96 MiB per immagini e target 64³).
Non vengono più riletti CT, mask o percentili durante le due run. Ogni crop
compare esattamente una volta per epoca in entrambe le varianti: con il
config originale, 50 esposizioni ciascuno e **300 aggiornamenti per run,
600 complessivi su due inizializzazioni indipendenti**. Non ci sono arresti
anticipati al gate: si conserva il primo successo, ma si completa lo stesso
budget per mantenere il confronto appaiato. Nessun pilota/full viene avviato.

Encoder MedicalNet obbligatorio, stesso seed 21, LR encoder 3e-5 e decoder
3e-4, AdamW, AMP e soglia 0,5 del config originale. Un hash dello stato
iniziale completo verifica che i pesi e i buffer delle due run coincidano.
`crop_pool.csv` identifica tutti i crop tramite caso, indice, conteggio
foreground e SHA256 di immagine e target. Il confronto usa lo stesso pool
per training e valutazione, incluse le figure e i controlli per caso.

**Loss conservata, non solo formula conservata.** La BCE rimane mediata su
otto crop. Il denominatore del contributo Dice viene fissato a **quattro
positivi per aggiornamento** in entrambe le varianti; sui batch di soli
reali il contributo resta zero. Così, a pesi fermi, la somma delle loss su
un'epoca è in entrambi i casi:

`0,5 × somma(BCE dei 48 crop)/8 + 0,5 × somma(Dice loss dei 16 positivi)/4`.

Normalizzare invece sui due/tre positivi presenti in ciascun batch misto
porterebbe il peso Dice complessivo da quattro a sei batch, +50%. Il test
di confronto controlla esplicitamente l'uguaglianza delle loss e dei
gradienti a pesi fermi. Con aggiornamenti AdamW/clipping intermedi i
percorsi saranno diversi: è l'effetto del batching che si vuole misurare.
Nei CSV `soft_dice_loss` è la componente ponderata con questo denominatore;
nei batch misti non equivale a `1 - Soft Dice medio dei positivi`.

Alla fine sincronizzare l'intera cartella, **senza i file `.pt`**. Leggere
prima `comparison.json`: contiene risultati finali e migliori delle due
run, hash iniziali, budget, esposizioni e differenze mixed meno grouped.
`delta_last_dice` positivo indica Dice finale maggiore; `delta_last_fp_fraction`
negativo indica meno FP. `passed` riguarda il gate dell'ultimo stato mixed,
non il solo fatto che sia migliore di grouped. Un best che passa prima del
termine è riportato separatamente, senza sostituire le metriche finali.

Ogni variante salva `gate.json`, `best_gate.json`, `report.json`, metriche
per aggiornamento e per caso, immagini finali, `best.pt` e `last.pt`.
Il best viene conservato ai controlli; il last viene scritto al termine,
evitando di riscrivere un checkpoint completo a ciascuna delle 50 epoche.
`update_metrics.csv` contiene gli indici degli otto crop e la composizione
di ogni aggiornamento; `exposures.csv` verifica le 50 visite per crop.
Tempi e picco di memoria sono in `report.json`; i tempi includono
inizializzazione, valutazioni e checkpoint, con dati già in RAM.

Exit code 2 indica confronto completato con gate finale mixed non superato.
Se `comparison.json` contiene `error`, il confronto è incompleto: non usare
le differenze come risultato. Una cartella non vuota viene rifiutata;
per ripetere scegliere un nuovo `--output-dir`. Non usare `--resume`.
I gate e i checkpoint precedenti non vengono modificati. I piloti restano
collegati al gate originale finché il nuovo risultato non viene riesaminato.

**Esito sincronizzato e riesaminato.** Il confronto è completo e appaiato:
stessi hash iniziali/pool, 300 aggiornamenti e 50 visite per crop. Al passo
300 grouped ha Dice 0,71491 e FP negativi 1,23613%; mixed Dice 0,73945 e
FP 0,69098%. I best sono entrambi al passo 250: Dice 0,74808/0,76074,
FP 0,70795%/0,62721%. Nessuna variante supera il gate Dice 0,90.
Non rilanciare identico il confronto e non avviare la sezione 18.3:
piloti/full restano bloccati. Il dettaglio degli errori residui e il
controllo successivo sui due crop positivi peggiori sono nella sezione 10.7
del registro; il comando implementato è nella sezione 18.2.3 seguente.

### 18.2.3 Debug dei due crop peggiori

Nel medesimo ambiente Windows del confronto, dalla root del repository:

```powershell
python -m tesi_m3d.debug_segmentation_hard_crops `
  --config configs\segmentation_overfit.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --reference-dir outputs\segmentation_batch_comparison `
  --output-dir outputs\segmentation_hard_crops
```

Usare gli stessi `$ctRoot` e `$realCtRoot` già sottoposti all'audit, definiti
nella sezione 18.1. La directory reference deve contenere `comparison.json`,
`manifest.json`, `effective_config.json` e `crop_pool.csv` del confronto
completo; non servono i suoi `.pt`. Il comando verifica manifest, config,
hash del ledger e hash individuali di immagine/target rigenerati. Conserva
gli indici originali nella coorte: cambiare l'indice cambierebbe il sampler.
Ogni prova ricrea MedicalNet con seed 21 e confronta l'hash di pesi/buffer
iniziali con quello del confronto precedente, prima di allenare.
Se non coincide, fermarsi e conservare il messaggio: stesso seed non
garantisce lo stesso stato fra ambienti/versioni diversi. Non eliminare il
controllo e non riscrivere gli hash del riferimento per aggirarlo.

Vengono eseguite quattro prove fresche, senza riprendere pesi allenati:

| Caso | Prova singola | Prova con negativo |
|---|---|---|
| `inj_2092` | positivo crop 3, 1.300 voxel target | stesso positivo + crop 4 |
| `rem_133` | positivo crop 2, 3.042 voxel target | stesso positivo + crop 4 |

Crop 4 è il primo negativo fisso dello stesso caso, non il negativo scelto
perché più facile o più difficile. La sua mask deve essere completamente
vuota. Gli input selezionati sono preparati una volta e tenuti in RAM;
nessuna rilettura per aggiornamento. Nessuna augmentation, clipping o
variazione di LR. Restano loss 0,5 BCE + 0,5 Soft Dice e soglia 0,5, con
valutazione FP32. Il batch effettivo contiene uno o due crop, con microbatch
al massimo quello del config; non è necessario accumulare otto crop.

Con il config originale ogni prova completa **300 aggiornamenti**, anche
se supera prima i criteri: **1.200 aggiornamenti complessivi** e 300 visite
a ciascun crop presente nella singola prova. Un fallimento scientifico
non salta le altre prove; un errore tecnico (gradienti non finiti,
preflight encoder/decoder/BatchNorm fallito o hash errati) interrompe.
Si conservano best, last e primo controllo che supera Dice almeno 0,90
e, nella coppia, FP sul negativo sotto 1%. I controlli sono ai passi
0, 1, ogni 25 e al termine. Il criterio del singolo non verifica negativi.

Sincronizzare `outputs/segmentation_hard_crops/` **senza i `.pt`**:

- `report.json`: stato delle quattro prove, best/last, preflight, versioni,
  eventuale `error` e `active_stage`; `passed` richiede tutte le metriche finali.
- `selection.json`: identità, hash, target voxel, bounding box e facce del
  crop toccate dalla mask, in coordinate locali z/y/x.
- Per ogni caso e prova: `checks.csv`, `updates.csv`, `update_check.json`,
  `best_metrics.json`, `result.json`, immagini iniziali/finali nei tre piani,
  `best.pt` e `last.pt`. Le figure descrivono il last, non il best intermedio.
- `initial_images/spatial_errors.csv` e `images/spatial_errors.csv`:
  TP/FP/FN nell'intero crop e nelle fasce disgiunte a distanza 0–3, 4–7,
  8–15 e almeno 16 voxel dalla faccia più vicina. Per confrontare bordi e
  interno usare anche `background_fp_fraction` (FP/voxel negativi della
  fascia), perché le fasce hanno dimensioni diverse. Metriche non definite
  sono celle vuote, senza assegnare Dice perfetto ai negativi vuoti.

Exit code 2 con quattro prove complete e senza `error` indica criteri
finali non tutti superati; non è un crash. Una cartella di output non vuota
viene rifiutata: non usare `--resume` e non cancellare i vecchi esperimenti.

**Interpretazione.** Successo isolato dimostra rappresentabilità di quel
crop in queste condizioni; successo nella coppia verifica anche il vincolo
del negativo selezionato. Un successo dopo il passo 50 usa più esposizioni
per crop del confronto precedente. Cambiano inoltre il numero di crop
nella media BCE e il denominatore Dice (un positivo anziché quattro): è
una diagnosi di memorizzazione, non una prova causale del solo batching.
Non concludere automaticamente che il problema sia dimenticanza, padding
o capacità del decoder. Gli errori per fascia e le figure aiutano a
decidere quale ipotesi controllare dopo. Nessun risultato di questo comando
riscrive `gate.json` o abilita piloti/full.

### 18.3 Baseline comune e due piloti controllati

Prima ricalcolare la baseline con il nuovo valutatore sugli otto casi pilota:

```powershell
python -m tesi_m3d.evaluate_segmentation `
  --checkpoint $baselineCheckpoint --cohort pilot_valid `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda --batch-size 2 `
  --out-dir outputs\segmentation_protocol\baseline_pilot
```

Poi eseguire entrambe le prove, ciascuna da massimo otto epoche:

```powershell
python -m tesi_m3d.train_segmentation `
  --config configs\segmentation_pilot.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda

python -m tesi_m3d.train_segmentation `
  --config configs\segmentation_pilot_clip.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda
```

Cambia soltanto il clipping p1/p99. Seed, inizializzazione del decoder,
manifest e sampling sono identici. Otto crop 64³ per scansione: quattro
positivi con posizione variabile nel crop, due negativi polmonari e due
globali; sui reali quattro polmonari e quattro globali. I negativi hanno
intersezione nulla con la mask, con fallback globale se necessario. Anche
intersezioni positive inferiori al 5% restano supervisionate voxel per voxel.
La ROI euristica non taglia l'inferenza. Nessun resampling; spacing registrato
e risultati separati nelle fasce z <=1, 1–2 e >2 mm.

Ogni worker legge una scansione e prepara insieme tutti gli otto crop. Il
trainer usa microbatch 2 e accumulo 4, AMP bfloat16, BCE + Dice sui crop con
foreground; i crop vuoti ricevono comunque BCE. In caso di OOM cambiare in
**entrambi** i YAML `batch_size: 1` e `accumulation_steps: 8`: il peso della
loss sul batch effettivo resta uguale. I primi due epoch aggiornano solo il
decoder, poi l'encoder con LR 3e-5 e il decoder con LR 3e-4, AdamW 1e-4.

Ogni due epoche esegue inferenza sui volumi completi. `best.pt` massimizza
`valid_dense_ap`, calcolata su tutti i voxel con istogrammi streaming a
65.536 bin; è un'approssimazione dichiarata, confrontata con il calcolo
esatto nei test. Non è l'AP delle patch bilanciate. `last.pt` consente il
resume; si conservano solo questi due checkpoint, con scrittura atomica.

I risultati sono in `outputs/segmentation_pilot*/evaluation/metrics.json`,
`volumes.csv` e `images/`. Il gate richiede Dice micro calibrato >=0,10,
picco dentro la mask in almeno 2/4 positivi, AP voxel media almeno tripla
della baseline e quota di falsi positivi sui reali non superiore alla
baseline. L'AP media e quella pooled sono riportate separatamente.
Se entrambi falliscono, fermarsi e analizzare gli errori residui.

### 18.4 Full soltanto dopo il superamento del pilota

Estendere prima l'audit e ricalcolare la baseline sulle 58 scansioni comuni:

```powershell
python -m tesi_m3d.prepare_segmentation `
  --config configs\segmentation_full.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --stage full

python -m tesi_m3d.evaluate_segmentation `
  --checkpoint $baselineCheckpoint --cohort common_valid `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda --batch-size 2 `
  --out-dir outputs\segmentation_protocol\baseline_common

python -m tesi_m3d.train_segmentation `
  --config configs\segmentation_full.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda
```

Il full ricontrolla i report dei due piloti, richiede protocolli uguali e
sceglie automaticamente il clipping della variante ammessa con AP pooled
maggiore. Non copia i pesi del pilota: riparte da MedicalNet con seed 21.
La scelta effettiva è in `effective_config.json`. Visita tutte le scansioni
train a ogni epoca, massimo 16 epoche, validation densa ogni due epoche su
16 casi, stop dopo tre validation senza miglioramento. Il best viene poi
valutato sui 58 casi comuni, con soglie distinte per mask e detection.

Il gate finale richiede Dice medio dei positivi >=0,30, picco nella mask
in almeno 15/30 casi, AP voxel media >=3 volte la baseline e falsi positivi
sui reali non superiori alla baseline. Un fallimento resta scritto nel
report: la presenza del codice non dimostra un miglioramento scientifico.

Un avvio su una run esistente richiede `--resume outputs\segmentation_full\last.pt`
oppure un output nuovo con `--output-dir`; non sovrascrive silenziosamente
i risultati. Il resume ripristina optimizer e RNG, verifica manifest e
preprocessing e consente il cambio 2/4 -> 1/8 in caso di OOM.

### 18.5 Cross-generator e inferenza

Valutare i generatori separatamente con le soglie della validation congelate:

```powershell
python -m tesi_m3d.evaluate_segmentation `
  --checkpoint outputs\segmentation_full\best.pt `
  --calibration outputs\segmentation_full\calibration.json `
  --split test --mods cycle `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --out-dir outputs\segmentation_full\test_cycle

python -m tesi_m3d.evaluate_segmentation `
  --checkpoint outputs\segmentation_full\best.pt `
  --calibration outputs\segmentation_full\calibration.json `
  --split test --mods diffusion `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --out-dir outputs\segmentation_full\test_diffusion
```

La CLI rifiuta test senza calibrazione, generatori usati nel train o
calibrazioni appartenenti a un altro checkpoint. Non sceglie soglie o
checkpoint sul test. Le vecchie valutazioni già consultate rimangono
esplorative; risultati validi su pix2pix non dimostrano generalizzazione.
Non escludere silenziosamente scansioni mancanti o prive di annotazioni:
correggere i percorsi o dichiarare esplicitamente un protocollo diverso.

Per inferenza su una sola CT non servono metadata, mask o coordinate:

```powershell
python -m tesi_m3d.infer_segmentation `
  --checkpoint outputs\segmentation_full\best.pt `
  --volume-dir "$ctRoot\pix2pix\scan\inj_2176" `
  --device cuda --out-dir outputs\segmentation_prediction
```

Produce `probability.npy` float32, `mask.npy` e `prediction.json`. Sliding
window 64³, sovrapposizione 50%, fusione gaussiana di probabilità voxel,
accumulo su CPU; dimensioni e coordinate restano quelle originali.
Le figure di valutazione usano assi z/y/x, sezioni della mask e del picco,
scala della probabilità fissa 0..1 e colori distinti per FP/FN/TP.
