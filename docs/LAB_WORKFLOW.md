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

**Esito sincronizzato.** Le quattro prove sono complete, con gli hash
attesi. `rem_133/2` raggiunge Dice 0,98208 da solo e 0,96864 con il
negativo (primo gate ai passi 150/200); `inj_2092/3` resta a 0,89260
da solo e 0,81985 nella coppia, best 0,89965 al passo 250. Entrambi i
negativi finali hanno zero voxel sopra soglia 0,5. Il gate globale rimane
fallito; il best injection non va arrotondato a 0,90. Non rilanciare lo
stesso test e non avviare piloti/full. Il successo removal mostra che un
piccolo target su due bordi può essere rappresentato. Il basso Soft Dice
e il fondo probabilistico residuo visibile nelle sezioni motivano un
controllo senza training sui checkpoint best/last già conservati, prima
di cambiare loss o architettura. Analisi, limiti e misure nella sezione
10.9 del registro; il comando di sola valutazione è nella sezione 18.2.4.

### 18.2.4 Probabilità complete sui checkpoint best/last

Sul PC del laboratorio, con gli stessi percorsi già sottoposti ad audit:

```powershell
python -m tesi_m3d.diagnose_segmentation_probabilities `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --run-dir outputs\segmentation_hard_crops `
  --reference-dir outputs\segmentation_batch_comparison `
  --output-dir outputs\segmentation_probability_diagnostics
```

Servono gli **otto checkpoint allenati** `best.pt` e `last.pt` delle quattro
prove, rimasti sul PC del laboratorio, insieme ai relativi JSON. Il comando
controlla prima che esistano tutti; se mancano, non avvia una nuova run e
non li ricrea. Usa config e manifest salvati, senza un nuovo `--config`.
Richiede anche i JSON/ledger del riferimento grouped/mixed, come il debug.

Rigenera una sola volta gli stessi due crop positivi e i due negativi,
controllandone gli hash. Ogni checkpoint deve corrispondere a config,
manifest, input/target, inizializzazione dichiarata, aggiornamenti e
metriche dello stadio best/last atteso. Ripristina **encoder e decoder
completi** con caricamento strict: non serve il file MedicalNet originale
e non esiste fallback a pesi casuali. Non ricostruisce i pesi iniziali dal
seed. Usa solo l'immagine come input del modello; la mask serve a misurare
le probabilità dopo l'inferenza.

Le otto valutazioni sono in FP32, senza autocast, `eval()` e senza
gradienti. Nessun optimizer, aggiornamento, ricerca di soglie o modifica
del gate. È una misura dei crop completi 64³ di training, non un'inferenza
sliding-window sui volumi CT completi.

L'output è separato dalla run e contiene:

- `report.json`: completamento, versioni, otto risultati con SHA256 dei
  checkpoint, metriche ricalcolate, differenze rispetto a quelle salvate,
  massa e probabilità media dentro/fuori dalla mask per crop.
- In `<caso>/<positive|positive_negative>/<best|last>/`, `distributions.csv`:
  conteggi, massa, media, min/max e quantili p01/p05/p50/p95/p99, calcolati
  su **tutti** i voxel float32 con quantili lineari e somme float64.
  Separazione in foreground/background/totale, sia sull'intero crop sia
  nelle fasce rispetto al bordo già descritte. Le frazioni sopra 0,1/0,5/0,9
  sono statistiche descrittive, non nuove soglie selezionate.
- `spatial_errors.csv` e immagini di ogni positivo/negativo ai tre piani
  della mask e del picco, anche per i best prima non visualizzati.
- `probabilities.npz`: array completi `probability` float32 e `mask` uint8,
  forma `B×1×D×H×W`, con B=1 o 2. Nessuna quantizzazione a 8 bit delle
  probabilità; coordinate locali identiche ai crop originali.

Per leggere i risultati, confrontare prima `best` e `last` nella coppia
injection. `probability_mass_outside` è la somma delle probabilità su tutti
i voxel fuori mask, inclusi quelli sotto 0,5; non è il numero di FP binari.
`background_mass_fraction` ne misura la quota sulla massa predetta totale.
Il Soft Dice float64 viene ricalcolato per ogni crop positivo dalla massa
interna e totale, con lo stesso smoothing `1e-5`; quello dei negativi è
indefinito. Nel report si conserva anche la loss FP32 originale per
verificare la riproduzione. Compartimenti vuoti hanno conteggio/massa zero
e statistiche non definite vuote/null, mai un Dice perfetto artificiale.

`complete: true` indica otto misure concluse, anche se il gate del training
resta falso. `all_reproduced_within_1e_4` richiede che tutte le metriche
ricalcolate differiscano da quelle salvate al massimo di `1e-4` assoluto;
se falso, leggere le differenze per checkpoint prima di confrontare i
risultati. Non è un nuovo gate scientifico. Un errore interrompe con
`error` e `active_checkpoint` nel report se l'inferenza è già iniziata.
Le assenze di checkpoint e gli errori preliminari vengono segnalati prima.

Sincronizzare **l'intera `outputs/segmentation_probability_diagnostics/`**,
inclusi i `.npz`; non produce nuovi `.pt`. La run e i checkpoint originali
restano intatti. L'output deve essere nuovo e fuori dalle due directory
sorgenti: nessun `--resume`. Al termine non avvia piloti/full e non dichiara
automaticamente risolto il problema della loss.

**Esito sincronizzato.** Tutti gli otto checkpoint riproducono le metriche
salvate (massimo scarto `5,96e-8`). Il fondo diffuso è confermato sui crop
completi dai CSV: medie circa 0,127–0,146, con almeno il 99,68% del
background dei positivi sopra 0,1. Nella coppia injection, dal best al
last la loss combinata migliora 0,54751 → 0,53283, ma il Dice binario
peggiora 0,89965 → 0,81985 e gli FP passano 290 → 567. Non rilanciare
lo stesso audit o allungare il training: il confronto motivato è ora
una singola variante del termine Dice, descritta nella sezione 10.11
del registro e implementata nella sezione 18.2.5. Piloti/full restano bloccati.

Le mappe `.npz` non sono arrivate: erano escluse dal `.gitignore` globale.
È stata aggiunta un'eccezione per i soli `probabilities.npz` di questa
directory diagnostica. Dopo aver aggiornato il codice sul PC del
laboratorio, includere nella normale sincronizzazione anche quei file
già prodotti; non occorre rieseguire il comando. Gli aggregati consentono
l'analisi precedente; le mappe servono per ulteriori verifiche sui voxel.

### 18.2.5 Confronto Dice lineare/quadratico

Sul PC del laboratorio, nello stesso ambiente usato per gli overfit:

```powershell
python -m tesi_m3d.compare_segmentation_losses `
  --config configs\segmentation_overfit.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --reference-dir outputs\segmentation_batch_comparison `
  --output-dir outputs\segmentation_loss_comparison
```

È un confronto delle **quattro prove fisse sui due crop difficili**,
non ancora il gate sulle sei scansioni. Esegue due rami:

- `linear/`: nuova baseline con il Dice originale;
- `squared/`: stessa BCE e Dice con denominatore `sum(p²)+sum(y²)`.

Ogni ramo prova `inj_2092/3` e `rem_133/2`, ciascuno da solo e con il
negativo fisso crop 4 dello stesso volume. Con il config originale sono
**otto training indipendenti da 300 aggiornamenti**, 2.400 complessivi;
ciascun crop presente in una prova riceve 300 esposizioni. Si completa
il budget anche dopo un gate superato; un fallimento scientifico non
salta le prove successive. Un errore tecnico interrompe e viene salvato.

I quattro crop vengono rigenerati una volta, verificati tramite gli hash
del riferimento e riusati in RAM. Config, preprocessing, seed 21,
encoder/decoder, LR, optimizer, clipping dei gradienti e AMP sono identici
fra i rami. Ogni training riparte da MedicalNet e decoder fresco: l'hash
iniziale completo deve coincidere con quello del confronto precedente.
Non si riprendono i vecchi best/last. Servono i pesi ufficiali MedicalNet
e l'ambiente originale: un hash iniziale diverso blocca prima del training.
Non serve trasferire i `.pt` degli esperimenti precedenti.

L'unica differenza di training è la normalizzazione spaziale del Dice:

`linear: (2*sum(p*y)+eps)/(sum(p)+sum(y)+eps)`;

`squared: (2*sum(p*y)+eps)/(sum(p*p)+sum(y*y)+eps)`.

Restano `eps=1e-5`, pesi 0,5 BCE/0,5 Dice, media sui soli crop con
foreground per il termine Dice e BCE anche sui negativi vuoti. Stesso
batch effettivo di uno/due crop delle prove precedenti, stesso microbatch.
Non modifica la loss dei normali entrypoint: il default resta lineare.

**Valutatore comune.** Ogni 25 passi (più 0, 1 e termine) le due varianti
sono misurate in FP32 con soglia 0,5 e gli stessi criteri: Dice positivo
almeno 0,90 e, nelle coppie, FP sul negativo sotto 1%. Best scelto con
gate/Dice binario/FP, mai dalla loss di training. `checks.csv` riporta
sempre il Soft Dice lineare comune; in `updates.csv`, `soft_dice_loss`
è invece la componente di training del ramo indicato da `squared_pred`.
Non confrontare direttamente quei valori fra i rami per decretare il
successo. `training_loss` nei report/checkpoint rende esplicita la formula.

**Probabilità già incluse.** Dopo ogni training vengono ripristinati e
valutati automaticamente best e last, senza altri aggiornamenti:
quantili, masse dentro/fuori, TP/FP/FN, precision/recall, figure e mappe
float32 complete. Non è necessario lanciare un altro audit separato.
I vecchi checkpoint diagnostici senza `training_loss` restano leggibili;
per i nuovi si controlla anche la corrispondenza di questa informazione.

Sincronizzare **`outputs/segmentation_loss_comparison/` senza i `.pt`**,
includendo CSV, JSON, PNG e `.npz` (già ammessi dal `.gitignore` solo in
questa directory). `comparison.json` contiene:

- `complete`, eventuale `error`, `active_stage` e `active_checkpoint`;
- i due rami con quattro prove ciascuno, best/last, preflight, primo
  superamento osservato, esposizioni e valutazioni probabilistiche;
- `deltas_squared_minus_linear`, per prova e best/last: differenze di
  Dice, precision/recall, FP/FN, media del fondo e FP sui negativi;
- `all_reproduced_within_1e_4`, per la riproduzione delle 16 valutazioni.

Per ogni ramo/caso/prova si trovano i checkpoint diagnostici, i CSV di
training e controllo, `initial_images/` e `images/` del last; le directory
`best/` e `last/` contengono mappe, figure e statistiche del rispettivo
checkpoint, come nell'audit precedente.

`passed` richiede il superamento delle quattro prove **finali** del ramo
quadratico: non significa superiorità sulla baseline, né superamento del
gate sulle sei scansioni. Exit code 2 con `complete: true` e senza `error`
indica confronto concluso con criteri finali non tutti superati. Gli
eventuali best riusciti sono riportati separatamente. Prima di decidere
il passo successivo confrontare anche controllo removal, precision/recall,
FP e fondo: un punteggio migliore della nuova loss non basta.
Nessun vecchio gate viene riscritto e nessun pilota/full viene avviato.

**Risultato sincronizzato, 23 novembre 2026.** Confronto completo: il
quadratico arriva a Dice 1,00 in tutte e quattro le prove, sia best sia
last a 300 passi. La nuova baseline lineare last ottiene rispettivamente
0,89593 / 0,92527 / 0,99574 / 0,96541 (injection solo/coppia, removal
solo/coppia). Tutti i negativi best/last sono privi di FP in entrambi i
rami. Verificati direttamente i voxel delle 16 mappe sincronizzate,
gli hash delle mask, i budget e i CSV; dettagli e limiti nella sezione
10.13 del registro. Rimane probabilità diffusa del fondo intorno a 0,11:
il Dice binario perfetto non dimostra probabilità calibrate.

Il prossimo controllo è il gate su **sei scansioni e 48 crop fissi**
con il solo Dice quadratico, MedicalNet fresco e criteri/budget originali.
Il comando di questa sezione verifica soltanto i due crop difficili.
Il collegamento della variante al gate originale è ora disponibile nella
sezione 18.2.6. Il gate originale resta fallito; non avviare ancora piloti/full.
Una directory non vuota viene rifiutata; nessun `--resume`.

### 18.2.6 Gate sulle sei scansioni con Dice quadratico

Dopo aver sincronizzato il codice aggiornato sul PC del laboratorio,
dalla root del repository e nell'ambiente Python già usato:

```powershell
python -m tesi_m3d.train_segmentation `
  --config configs\segmentation_overfit_squared.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot `
  --device cuda --debug-updates
```

Usare i percorsi `$ctRoot` e `$realCtRoot` già verificati nell'audit della
sezione 18.1. Non occorre rigenerare il manifest o rieseguire il confronto
sui due crop. La nuova configurazione differisce dall'overfit originale
soltanto per `training.squared_pred: true` e i percorsi dell'output/gate.
L'opzione viene ora validata e inoltrata realmente alla loss; il default
degli esperimenti precedenti resta lineare. Non usare `--resume`: si parte
da encoder MedicalNet ufficiale e decoder fresco, senza pesi dei crop
già memorizzati. Nessun fallback a encoder casuale.

Stesse sei scansioni, otto crop fissi per scansione, niente augmentation,
batch fisico 2 e accumulo 4, LR e preprocessing originali. Massimo 300
aggiornamenti complessivi (50 esposizioni per crop a budget completo),
con stop anticipato se il gate passa a un controllo ogni 25 passi.
Il successo richiede ancora Dice medio dei crop positivi almeno 0,90,
FP sui crop negativi sotto 1% e gradienti finiti. Le soglie non cambiano.

Sincronizzare **`outputs/segmentation_overfit_squared/` senza i `.pt`**:

- `gate.json` e `best_gate.json`, con la loss dichiarata in `training_loss`;
- `effective_config.json`, `manifest.json`, `metrics.csv`, `overfit_metrics.csv`;
- `update_metrics.csv` (BCE, Dice quadratico, gradienti/aggiornamenti),
  `case_history.csv`, `crops.csv`, `cases.csv` e `images/`.

`--debug-updates` abilita i diagnostici durante il training; `crops.csv`,
`cases.csv` e le immagini finali riguardano il last. Best e last vengono
salvati separatamente; `best_gate.json` documenta la selezione intermedia.
L'output è `outputs/segmentation_overfit_squared`, senza sovrascrivere le
prove precedenti. Una run già presente viene rifiutata.

Exit code 2 con `gate.json` valido e `passed: false` indica gate scientifico
fallito. Attendere l'analisi prima dei piloti, anche se il gate passa.
Il trainer rifiuta di cambiare denominatore Dice durante un resume e di
usare per pilota/full un gate ottenuto con un denominatore differente.
Le configurazioni pilota/full esistenti restano lineari e non vengono
promosse automaticamente a questa variante.

**Esito del laboratorio.** Il gate quadratico è terminato a 300 passi:
Dice medio positivo **0,782871**, FP negativi **0,37899%**, `passed: false`.
Anche il best a 275 fallisce (Dice 0,785942). Precision micro nei crop
positivi 0,8752, recall 0,6760: la riduzione dei FP si accompagna a target
persi, soprattutto `inj_1419/1`. Dettagli e confronto con la baseline nella
sezione 10.15 del registro. Non avviare piloti/full e non prolungare questa
run. Il confronto successivo grouped/mixed con Dice quadratico e budget
invariato è ora disponibile nella sezione 18.2.7.

### 18.2.7 Batch per volume contro misti con Dice quadratico

Aggiornare prima il codice sul PC del laboratorio. Dalla root del
repository, nello stesso ambiente e con gli stessi percorsi dell'audit:

```powershell
python -m tesi_m3d.compare_segmentation_batches `
  --config configs\segmentation_overfit_squared.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --output-dir outputs\segmentation_batch_comparison_squared
```

Si riusa la configurazione quadratica, ma **l'output è quello indicato da
`--output-dir`**: i risultati del gate precedente non vengono sovrascritti.
Non occorrono un nuovo audit, un nuovo manifest o altre dipendenze.
Non aggiungere `--resume` o `--debug-updates`: il confronto non riprende
run precedenti e registra già i diagnostici di tutti gli aggiornamenti.

Il comando avvia **due nuove inizializzazioni MedicalNet**, con encoder
preaddestrato obbligatorio e decoder fresco. Il pool di 48 crop viene
preparato una sola volta, senza augmentation, verificato e identificato
tramite hash di immagini/mask e condiviso fra i rami:

- `grouped/`: otto crop della stessa scansione per aggiornamento;
- `mixed/`: otto crop di almeno tre scansioni, con due/tre positivi e
  almeno un negativo di ciascuno dei due reali.

In entrambi i rami la loss è **0,5 BCE + 0,5 Dice quadratico**. BCE su tutti
i crop, Dice solo sui positivi e normalizzatore foreground fisso a quattro:
cambiare la composizione non aumenta il peso complessivo del Dice. Seed,
preprocessing, LR, soglia 0,5, batch fisico 2 e accumulo 4 restano quelli
del gate quadratico. Le due inizializzazioni devono avere lo stesso hash;
un mismatch interrompe il mixed prima del primo aggiornamento.

**300 aggiornamenti per ramo, 600 complessivi e 50 esposizioni per crop**,
anche se un gate passa prima. Non equivale alle prove isolate da 300
esposizioni per crop. Il controllo rimane Dice medio positivo almeno 0,90,
FP negativi sotto 1% e gradienti finiti; non basta migliorare grouped.

Sincronizzare **`outputs/segmentation_batch_comparison_squared/` senza
i `.pt`**, mantenendo entrambe le sottocartelle e tutti i JSON, CSV e PNG.
In particolare:

- `comparison.json`: `complete`, ramo attivo, eventuale `error`, loss,
  risultati best/last, hash iniziali e differenze mixed meno grouped;
- `crop_pool.csv`, config e manifest: identità dei 48 crop;
- in ciascun ramo, gate best/last, `report.json`, `exposures.csv`, metriche
  per aggiornamento/caso, `crops.csv`, `cases.csv` e `images/` del last;
- `best/crops.csv`, `best/cases.csv` e `best/images/`, prodotti ricaricando
  il best con lo stesso valutatore FP32 e soglia 0,5. Le sue metriche devono
  riprodursi entro `1e-4`, altrimenti il confronto risulta incompleto.

`training_loss.squared_pred` deve essere `true` nei report/gate/checkpoint,
e `squared_pred` deve essere `True` nei CSV di aggiornamento. Il vecchio
config senza questa chiave continua a selezionare il Dice lineare.
Non confrontare i valori di loss lineare/quadratica per dichiarare successo.
Per capire se il mixed recupera anche le regioni perse, leggere precision,
recall, FP e FN nei CSV dei crop best/last, soprattutto `inj_1419/1`.

`complete: true` senza `error` indica che entrambi i budget e l'audit dei
best sono conclusi. `passed` riguarda il **last mixed**: un best riuscito
è documentato separatamente. Exit code 2 può quindi indicare un confronto
tecnicamente completato ma con gate scientifico fallito. Un `error` o
`complete: false` impone di risolvere il problema tecnico prima di usare
le differenze. Una cartella non vuota viene rifiutata; per ripetere usare
un nuovo output. Nessun vecchio gate viene modificato e nessun pilota/full
parte automaticamente. Questo rimane un test su crop di training.

**Esito sincronizzato.** Confronto completo e appaiato, 300 aggiornamenti
e 50 esposizioni per crop in entrambi i rami. Grouped last: Dice 0,801162,
FPR negativi 1,08588%. Mixed last: Dice 0,821193, FPR 0,49747%; best mixed
a 275: Dice 0,832099, FPR 1,23647%. Nessun gate superato. Il mixed finale
riduce i FP, ma la recall micro nei crop positivi scende da 0,8058 a 0,7741.
Non avviare la sezione 18.3 e non rilanciare identico il training. Dettagli
e limiti nella sezione 10.17 del registro. L'audit senza training delle
probabilità sui quattro best/last e tutti i 48 crop è ora nella sezione
18.2.8. Le soglie diagnostiche sul train non sostituiscono il gate a 0,5.

### 18.2.8 Probabilità e soglie sui quattro checkpoint e sui 48 crop

Aggiornare il codice sul PC del laboratorio, poi dalla root del repository
nell'ambiente Python già usato e con gli stessi percorsi dell'audit:

```powershell
python -m tesi_m3d.diagnose_segmentation_batch_probabilities `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --run-dir outputs\segmentation_batch_comparison_squared `
  --output-dir outputs\segmentation_batch_probability_diagnostics
```

È una **valutazione dei pesi già allenati**: richiede i quattro `.pt`
grouped/mixed best/last nel laboratorio, senza optimizer, backward o
riaddestramento. Non richiede un nuovo audit dati, manifest o file di
configurazione. Ricostruisce il pool una sola volta, controllando tutti
gli hash di immagini/mask del `crop_pool.csv` originale. Un checkpoint
mancante blocca prima di leggere le CT; non ripetere il training per
questo motivo se i pesi esistono ancora nel PC del laboratorio.

Carica lo stato completo dei checkpoint, senza download né necessità di
ritrovare separatamente il file MedicalNet iniziale. Verifica task, modello,
preprocessing, manifest, sampling, loss, aggiornamenti e selezione del best.
L'inferenza è FP32, senza autocast, sulla sola immagine CT. A soglia 0,5
deve riprodurre entro `1e-4` Dice/FPR aggregati, Dice/FPR dei singoli crop e
picchi di probabilità già salvati; controlla anche la stessa decisione
numerica del gate e riporta gli scarti nei conteggi. Un errore interrompe
l'audit senza modificare sorgenti o gate.

Per ciascuno dei quattro checkpoint salva:

- `crops.csv`: metriche a 0,5, AP per crop, medie/masse dentro e fuori mask,
  scarti dagli output originali e massimo Dice diagnostico per crop;
- `distributions.csv`: quantili esatti su tutti i voxel, anche per fasce
  di distanza dai bordi, riusando le definizioni dell'audit precedente;
- `thresholds.csv`: **1.001 soglie comuni** da 0 a 1, passo 0,001, con
  confronto `probabilità >= soglia`, Dice medio positivo, Dice micro,
  precision/recall nei crop positivi e FP sui negativi/reali;
- `diagnostic_crops.csv`: dettaglio per crop alle due soglie comuni che
  massimizzano il Dice medio, senza vincoli e con FPR negativo **<1%**;
- `report.json`: riproduzione a 0,5, AP aggregata su tutti i crop e sui
  positivi, media AP dei positivi, soglie diagnostiche e limiti dell'oracolo;
- `metrics.npz`: soglie, conteggi interi `48×1001×4` nell'ordine TP/FP/FN/TN,
  istogrammi `48×2×65536` nell'ordine foreground/background. Permettono di
  ricontrollare AP e profili senza rieseguire il modello;
- figure per il peggior positivo e negativo della valutazione originale,
  con gli stessi piani alla soglia 0,5 e a quella diagnostica ammissibile.

L'AP usa tutti i voxel, senza sottocampionamento, con quantizzazione a
**65.536 bin**: è approssimata, mentre i conteggi sulle 1.001 soglie sono
esatti rispetto alle probabilità float32 promosse a float64. AP non
definita sui crop reali/negativi = `null`, non zero. I crop sovrapposti
contano più volte i voxel fisici: non sono metriche di validation su volumi
originali. Non accumula tutte le probabilità predette in RAM e non le
esporta: conserva istogrammi/conteggi e solo le due mappe per le figure.
Quantili, ricerca delle soglie e compressione lavorano su CPU; l'uso della
GPU sarà intermittente. Nessuna nuova dipendenza.

**Interpretazione diagnostica.** Confrontare `fixed_threshold` e
`diagnostic_train_only` nei report. Una buona AP con Dice che cresce molto
a una soglia comune indica che serve approfondire la scala delle
probabilità; un recupero scarso anche sulla griglia richiede analizzare
le regioni confuse e le figure. La media dei massimi per singolo crop è
un oracolo ottimistico sulla stessa griglia, non una soglia utilizzabile
in inferenza. Una griglia con passo 0,001 non esamina tutti i possibili
valori di soglia. Nessuno di questi risultati riscrive il gate a 0,5,
genera una calibrazione o sblocca automaticamente i piloti.

Sincronizzare **`outputs/segmentation_batch_probability_diagnostics/`**,
con JSON, CSV, PNG e i quattro `metrics.npz`, già ammessi dal `.gitignore`
solo in questa directory. Non servono i `.pt` per analizzare gli output.
Il report principale deve avere `complete: true`, nessun errore e tutti
i quattro checkpoint riprodotti. Con `complete: false` leggere `error` e
`active_checkpoint`; non interpretare come concluso l'audit. Non esiste un
flag `passed`: una valutazione tecnicamente completata non è un successo
del modello. Una directory non vuota o interna alla run sorgente è rifiutata.

**Se compare `checkpoint/crop metrics were not reproduced within 1e-4`.**
È fallito il confronto con le metriche salvate, non un nuovo training.
Il valutatore è stato allineato alla run sorgente: `set_global_seed`
(che su CUDA abilita `cudnn.benchmark`), `torch.no_grad()` e sigmoid
sul device del modello prima del trasferimento CPU. La tolleranza resta
`1e-4`; questi allineamenti non garantiscono identità numerica fra kernel
CUDA o versioni diverse. Il nuovo errore indica metrica, scarto e crop;
il report del checkpoint conserva tutti i `reproduction_failures`.

Dopo aver aggiornato il codice, conservare il tentativo fallito e ripetere
solo la valutazione, con gli stessi `$ctRoot` e `$realCtRoot`:

```powershell
Rename-Item -Path outputs\segmentation_batch_probability_diagnostics `
  -NewName segmentation_batch_probability_diagnostics_failed_01 -ErrorAction Stop

python -m tesi_m3d.diagnose_segmentation_batch_probabilities `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --run-dir outputs\segmentation_batch_comparison_squared `
  --output-dir outputs\segmentation_batch_probability_diagnostics
```

Il nome `_failed_01` deve essere libero: non sovrascrivere tentativi precedenti.
Se fallisce ancora, sincronizzare la nuova directory anche incompleta,
in particolare `report.json`, `grouped/best/report.json` e
`grouped/best/crops.csv` (oppure il checkpoint indicato da `active_checkpoint`).
Non allargare la tolleranza e non riaddestrare per aggirare il controllo.

**Quando riprendere il training.** Dopo l'analisi di questi risultati si
può scegliere una sola modifica motivata e riprovare l'overfit. Il passaggio
ai piloti richiede ancora un gate valido sulle sei scansioni (Dice medio
positivo almeno 0,90, FPR negativo sotto 1%, gradienti finiti). Il training
completo richiede poi il successo dei piloti. Non aumentare nel frattempo
le epoche della run corrente.

### 18.2.9 Nuovo overfit: padding zero contro replicate nel decoder

**Risultati acquisiti.** Confronto completo, entrambi i gate falliti:
Dice 0,865861/FPR negativo 0,27144% per zero, 0,845678/0,31191% per
replicate. Audit best/last riprodotti e NPZ ricontrollati; non occorre
rilanciare questo comando sugli stessi dati per completare la consegna.
Mantenere zero. Il nuovo controllo zero differisce però dal precedente
mixed zero (Dice 0,821193) pur con stessi hash e schedule: prima di altre
varianti va verificata la ripetibilità del training. Il test deterministico
è ora implementato nella sezione 18.2.10. Dettagli dei risultati nella
sezione 10.22 del registro; i comandi seguenti documentano la prova acquisita.

L'audit dei quattro checkpoint è completo, ma nessuna soglia comune sulla
griglia recupera il gate. Anche gli oracoli separati per crop restano sotto
Dice 0,90. Nel mixed finale l'86,14% dei FP dei crop negativi cade negli
otto voxel esterni, che costituiscono il 57,81% dei voxel: questo motiva
una prova sul padding, senza dimostrarne già la responsabilità.

Aggiornare il codice sul laboratorio e, dalla root con l'ambiente attivo,
eseguire **un solo comando**, usando gli stessi percorsi già sottoposti
all'audit (`$ctRoot` e `$realCtRoot` della sezione 18.2):

```powershell
python -m tesi_m3d.compare_segmentation_padding `
  --config configs\segmentation_overfit_squared.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --reference-dir outputs\segmentation_batch_comparison_squared `
  --output-dir outputs\segmentation_padding_comparison
```

Non rilanciare prima `train_segmentation` o altri audit. Il comando:

1. Controlla config/manifest contro il confronto quadratico precedente e
   ricostruisce i medesimi 48 crop, verificando identità e hash immagini/mask.
2. Riparte due volte da MedicalNet ufficiale con decoder fresco, stesso
   seed e hash iniziale completo, coincidente anche con il riferimento.
   Non riprende i pesi allenati nei test precedenti.
3. Allena `zeros` e `replicate`, entrambi con batch misti, Dice quadratico,
   identici LR e ordine dei crop: 300 aggiornamenti per ramo, 50 esposizioni
   per crop. Un successo precoce non accorcia il budget del confronto.
4. Esegue automaticamente l'audit delle probabilità di best/last per
   entrambi i rami, inclusi AP, soglie diagnostiche, figure e conteggi
   per fascia dal bordo. Non occorre un successivo comando diagnostico.

Usare lo stesso ambiente PyTorch/MONAI della run sorgente: il laboratorio
ha registrato PyTorch `2.13.0+cu130` e MONAI `1.5.2`. Non aggiornare le
librerie prima del confronto. Se l'hash iniziale non coincide, il comando
si ferma prima del training del ramo e indica hash atteso/ottenuto:
conservare `comparison.json` e analizzare la differenza, senza rimuovere
il controllo. Nel runtime locale diverso, il vecchio e il nuovo builder
zero producono lo stesso stato iniziale, ma diverso da quello storico
Windows; non è stato attribuito a una singola libreria o piattaforma.

La sola differenza è `model.decoder_padding_mode`, nelle sei Conv3D 3×3
del decoder; encoder MedicalNet, shortcut, dilatazioni, BatchNorm e gli
altri strati restano identici. Il default assente rimane `zeros` per i
vecchi checkpoint. Il ripristino usa il valore salvato nel config; i
controlli pilota/full rifiutano un gate ottenuto con padding diverso.
`replicate` non aggiunge nuovo contesto anatomico esterno al crop: verifica
soltanto l'effetto del trattamento numerico del bordo nel decoder.

Gli output sono tutti in `outputs/segmentation_padding_comparison/`:

- `comparison.json`: stato completo/incompleto, ramo e checkpoint attivi,
  errori, hash iniziali/pool, metriche best/last e differenze replicate–zeros;
- `zeros/` e `replicate/`: checkpoint best/last, gate, cronologia loss e
  gradienti, esposizioni, CSV per crop/caso e figure del valutatore comune;
- per ciascun ramo, `probabilities/best/` e `probabilities/last/`: audit
  completo, `metrics.npz` e `boundary_metrics.csv` con TP/FP/FN, recall e
  FPR del background per fascia, separati fra positivi, negativi e reali.

Sincronizzare l'intera cartella, compresi i quattro `metrics.npz` già
ammessi dal `.gitignore`. I pesi `.pt` possono restare sul laboratorio.
La directory deve essere nuova; non sovrascrivere i risultati precedenti.
Non c'è resume. Se il comando termina con codice 2 ma `complete: true`
e senza `error`, il confronto tecnico è concluso e il gate del last
`replicate` è ancora fallito. Con `complete: false` analizzare l'errore,
senza interpretare come completo il confronto. I report distinguono
anche il best dal last: non promuovere un modello per il solo Dice maggiore.

Il criterio resta Dice medio dei positivi ≥0,90 e FPR negativi <1% alla
soglia 0,5, con gradienti finiti. Confrontare recall e fasce di bordo,
senza escludere voxel difficili dal conteggio. Le soglie sul train sono
solo diagnostiche. Nessun avvio automatico dei piloti o del full. Memoria
e tempi CUDA di `replicate` devono essere misurati nel laboratorio;
non cambiare altri parametri fra i due rami per aggirare un eventuale OOM.

### 18.2.10 Due overfit zero identici con determinismo stretto

Questo test verifica se il training controllato è ripetibile. Non cambia
loss, LR, sampling, inizializzazione o soglia per cercare un gate migliore.
Aggiornare il codice nel laboratorio e usare lo stesso ambiente delle run
precedenti, dalla root del repository. In un nuovo processo PowerShell,
impostare le variabili prima di avviare Python:

```powershell
$env:PYTHONHASHSEED = "21"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

python -m tesi_m3d.compare_segmentation_padding `
  --deterministic-repeats `
  --config configs\segmentation_overfit_squared.yaml `
  --data-root $ctRoot --real-scan-root $realCtRoot --device cuda `
  --reference-dir outputs\segmentation_batch_comparison_squared `
  --output-dir outputs\segmentation_determinism_comparison
```

Usare gli stessi `$ctRoot` e `$realCtRoot` già sottoposti all'audit.
Il runner esistente con il nuovo flag esegue `repeat_1` e `repeat_2`,
entrambi padding zero e batch misti, 300 aggiornamenti/50 esposizioni per
crop, stessi 48 crop verificati con hash e stesso stato iniziale MedicalNet
del riferimento. Non riprende i pesi allenati precedenti; il caricamento
MedicalNet completo con SHA resta obbligatorio. Nessun nuovo YAML.

**Operazioni deterministiche.** Abilita gli algoritmi deterministici con
`warn_only=False`, disabilita l'autotuning e abilita il determinismo cuDNN.
`set_global_seed` rispetta ora tale impostazione. Un'operazione non
supportata interrompe il test: non viene attivata una modalità permissiva.

Il backward CUDA della trilineare nativa è dichiarato non deterministico
da PyTorch. Per evitare che il test si fermi sistematicamente su questa
operazione, il percorso diagnostico usa **forward trilineare nativo e
solo backward trilineare su CPU**, esplicitamente registrato come
`cpu_first_order`. Il Jacobiano dipende dalla geometria, non dai valori
in ingresso; viene calcolato con la trilineare PyTorch CPU, stesse forme,
dtype e `align_corners=False`, riportando poi il gradiente sul device.
L'encoder e le convoluzioni rimangono sul device richiesto. Non si cambia
in nearest-neighbor e non si introduce un contorno diverso.

Anche `max_pool3d_with_indices_backward_cuda` dell'encoder è incompatibile
con la modalità stretta nella versione del laboratorio. Il percorso
diagnostico mantiene **forward e indici dei massimi nativi sul device**,
poi esegue il backward PyTorch di MaxPool3D su CPU usando proprio quegli
indici, senza ricalcolarli. Kernel, stride, padding, dilation e ceil mode
restano quelli MedicalNet; vengono sommate anche le contribuzioni di
finestre sovrapposte. Si usa il kernel CPU nativo anche per conservare
l'aritmetica bfloat16, diversa da un generico `scatter_add`. Il gradiente
torna sul device. La politica salvata è
`maxpool3d_backward: cpu_first_order_native_indices`; non si disabilita
il determinismo. Il training ordinario conserva il percorso nativo.

Se il primo tentativo si è interrotto con questo errore, aggiornare il
codice e conservare la cartella incompleta prima di rilanciare il comando
sopra, dall'inizio e in un nuovo processo Python:

```powershell
Rename-Item -Path outputs\segmentation_determinism_comparison `
  -NewName segmentation_determinism_comparison_failed_maxpool -ErrorAction Stop
```

Il nome di destinazione deve essere libero. Nessun peso allenato va
recuperato dal tentativo interrotto: l'errore avviene prima del primo step.

Questa scelta cambia il kernel e l'ordine numerico del backward rispetto
alle run storiche: il test misura la ripetibilità **di questo percorso
controllato nello stesso processo**, non l'identità numerica col vecchio
training né la causa unica della sua variabilità. I due run sono sequenziali,
con seed e modello reimpostati. Il controllo non dimostra ripetibilità
fra macchine, versioni o processi indipendenti. Il percorso serve a gradienti
di primo ordine, quelli usati da AdamW. Non modifica il forward in inferenza
o i checkpoint/config precedenti. Hash dello stato iniziale e gate invariati.

**Output e interpretazione.** Tutto in
`outputs/segmentation_determinism_comparison/`, directory nuova. Per ogni
update salva in `update_metrics.csv` anche `model_state_sha256` (parametri
e buffer dopo lo step) e `gradient_sha256` (gradienti dopo clipping).
Confronta esattamente loss, norme e hash per ciascun passo, monitoraggio,
metriche per crop e profili di soglia; nessuna tolleranza per dichiarare
uguali le due tracce. Restano i quattro audit best/last, figure, conteggi
per fascia e quattro `metrics.npz`, ammessi dal `.gitignore`.

Nel report principale `comparison.json`:

- `complete: true`: entrambi i budget e gli audit sono terminati;
- `reproducible: true`: tutte le tracce e metriche confrontate coincidono;
- `first_divergence`: file, riga/aggiornamento e valori che differiscono,
  oppure `null` se i confronti sono identici;
- `passed`: entrambe le run finali passano il gate numerico e sono
  riproducibili. Non è una promozione automatica a piloti/full;
- `runtime`: versioni, GPU, thread, flag deterministici/TF32, workspace e
  politica del backward. I checkpoint salvano anche `execution_policy`.

Codice di uscita 0: riproducibile e gate superato; 2: riproducibile ma
gate fallito; 3: confronto terminato con divergenze. Un errore tecnico
lascia `complete: false`, ramo/checkpoint attivo ed `error`; non dichiarare
riproducibilità per un'esecuzione interrotta. Le impostazioni globali
vengono ripristinate anche in caso di errore. Non esiste resume e non si
sovrascrivono gli esperimenti precedenti.

Sincronizzare l'intera nuova cartella anche se il gate fallisce o il test
si interrompe. Non rilanciare altri audit: sono inclusi. Questa prova sarà
più lenta per backward CPU e hash dei pesi/gradienti a ogni passo; non è
un benchmark della pipeline. Tempi/memoria CUDA richiedono la misura nel
laboratorio. Non aumentare il budget oltre 300 o avviare il full nel frattempo.

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
