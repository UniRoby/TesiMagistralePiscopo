# Localizzazione Pix2Pix con TruFor

Questa pipeline confronta TruFor adattato alle slice CT con la stessa rete dotata di contesto fra slice. L'obiettivo è segmentare i voxel manipolati, sia nelle aggiunte sia nelle rimozioni. La qualità sulle CT deve essere misurata: i pesi forensi originali sono stati addestrati su immagini fotografiche.

## Dati e normalizzazione

Gli split sono quelli di `metadata/m3dsynth/sets.csv`. Il training comprende tutte le CT Pix2Pix e le serie autentiche distinte. Le serie autentiche sono deduplicate usando `orig_id` e `sdir_id`; un paziente non può comparire in split diversi.

| Split | CT Pix2Pix | Serie autentiche distinte |
|---|---:|---:|
| Train | 1.537 | 489 |
| Valid | 416 | 103 |
| Test | 565 | 152 |

Questi sono i conteggi nei metadati della repository. `prepare` verifica che i file corrispondenti esistano sul PC del laboratorio. Un caso mancante blocca la preparazione e mostra il percorso: non viene rimosso dal dataset senza avviso.

Per ogni CT vengono calcolati i percentili 1 e 99 dei voxel non nulli. Un istogramma dei valori interi permette di leggere una slice alla volta e riprodurre l'interpolazione lineare di `numpy.percentile`. Tutte le slice della CT usano gli stessi limiti. I valori normalizzati vengono limitati all'intervallo [0,1]. Per una scansione costante si usa un denominatore minimo di `1e-6`.

I percentili e le informazioni sulle maschere sono salvati in `outputs/trufor/cache/percentiles`. La chiave comprende percorsi, dimensioni dei file, timestamp e parametri della normalizzazione. La cache sopravvive ai riavvii ed è condivisa dai tre run. Se cambia un input, occorre ripetere `prepare`; le serie invariate riutilizzano la cache. La prima preparazione deve leggere tutte le immagini e può richiedere tempo.

Le maschere rimangono quelle distribuite da M3Dsynth. Le eventuali slice finali eccedenti vengono tagliate soltanto se vuote, e il loro numero compare nel manifest. Maschere più corte delle immagini, buchi nella sequenza TIFF, dimensioni incompatibili o voxel positivi oltre il volume producono un errore. Non vengono ricavate maschere da `coord_z` o dalle altre coordinate della manipolazione.

## Architettura

La dipendenza esterna è [TruFor ufficiale](https://github.com/grip-unina/TruFor), revisione `ae54475df6f41a491d7615100feb19263dec13f7`. Si usano Noiseprint++, l'encoder a due rami SegFormer-B2 e il decoder di localizzazione. Le teste confidence e detection del checkpoint non entrano nel modello CT.

La slice in scala di grigi viene replicata su tre canali per conservare i pesi originali. Noiseprint++ riceve valori in [0,1], resta congelato e lavora in modalità valutazione. Il ramo immagine conserva la normalizzazione ImageNet prevista da TruFor. Encoder e decoder vengono adattati alle CT. Le statistiche BatchNorm rimangono congelate, perché il microbatch contiene una sola slice centrale.

La variante con contesto estrae le feature di `z-1`, `z` e `z+1` con gli stessi pesi. Per ogni scala, un gate convoluzionale 1×1 riceve feature centrali, feature vicine, differenza assoluta e distanza firmata in millimetri. Il gate produce un peso spaziale per il vicino. La correzione è la somma delle differenze pesate, divisa per il numero di vicini validi, e viene aggiunta alle feature centrali attraverso un coefficiente apprendibile inizializzato a zero.

Questa inizializzazione rende le due reti equivalenti in valutazione prima di allenare il contesto. Alle estremità della CT, i vicini mancanti non contribuiscono alla correzione. Non si esegue registrazione deformabile fra slice: il gate confronta feature nelle stesse coordinate in-plane. L'eventuale utilità del contesto anatomico è l'ipotesi da verificare con il confronto controllato.

Il decoder restituisce due logit per pixel. La softmax del canale manipolato fornisce `P(z,y,x)`. Le slice mantengono la risoluzione nativa; il padding al multiplo di 32 viene rimosso prima di salvare l'output. Il sistema non usa patch centrate sulla manipolazione in inferenza.

## Campionamento e confronto sperimentale

Ogni epoca presenta due slice positive e due negative per CT manipolata, e quattro slice negative per serie autentica. Se una CT manipolata non ha slice negative, i relativi slot vengono riempiti con slice autentiche. Il campionamento cambia fra epoche. I flip e le rotazioni di 90° sono sincronizzati su immagini e maschera. Non sono applicate augmentazioni di rumore, compressione o intensità.

Tutti i volumi partecipano a ogni epoca; questo non significa che tutte le slice siano visitate in una singola epoca. Il report di ogni epoca contiene presentazioni totali, slice distinte, volumi distinti e hash dell'ordine di campionamento. Con i metadati attuali sono previste 8.104 presentazioni per epoca.

La loss somma focal loss (`gamma=2`, peso positivo `0,75`) e soft Dice sul foreground. Nelle slice con maschera vuota, il termine Dice è escluso e la focal loss penalizza i falsi positivi. Si usa AdamW, con learning rate `1e-5` nell'encoder e `1e-4` nel decoder e nei gate, weight decay `0,01`, warm-up del 5% e decadimento cosine. Il clipping della norma dei gradienti è fissato a 1.

Il microbatch è 1 e l'accumulo copre 8 slice centrali. L'ultimo gruppo dell'epoca può essere più corto: la loss viene divisa per il numero effettivo di elementi. Su CUDA si usa BF16 quando supportato, altrimenti FP16 con scaler. Il checkpointing delle attivazioni riduce la memoria dell'encoder.

| Run | Inizializzazione | Budget iniziale |
|---|---|---:|
| `adapt_2d` | Pesi forensi TruFor ufficiali | 20 epoche |
| `continue_2d` | Miglior checkpoint di `adapt_2d` | 10 epoche |
| `context` | Lo stesso checkpoint di `adapt_2d` | 10 epoche |

Nelle due prosecuzioni optimizer e scheduler vengono inizializzati da zero. Slice centrali, augmentazioni, loss e numero di aggiornamenti sono uguali. Si confrontano `continue_2d` e `context`: confrontare soltanto `adapt_2d` e `context` confonderebbe l'effetto del contesto con quello del training aggiuntivo. Il costo computazionale differisce perché il contesto richiede tre passaggi nell'estrattore. Gli strati stocastici non hanno necessariamente la stessa sequenza di attivazioni fra le due architetture.

Un pannello fisso di validazione comprende 20 CT con aggiunta, 20 con rimozione e 10 autentiche. Ogni volume viene analizzato integralmente dopo ogni epoca. La selezione di `best.pt` usa il F1 medio delle CT manipolate a soglia 0,5. Il pannello riduce il costo della selezione dei checkpoint; calibrazione e risultati finali usano gli split completi.

## Installazione sul PC del laboratorio

Aprire PowerShell nella repository e attivare l'ambiente già funzionante. Occorrono Git, PyTorch almeno 2.6 e una build CUDA compatibile con la RTX 5060 Ti. Il comando seguente mostra la configurazione effettiva:

```powershell
Set-Location 'C:\Tesi Magistrale Piscopo\TesiMagistralePiscopo'
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Installare le dipendenze mantenendo la versione corrente di PyTorch. Il vincolo impedisce al resolver di aggiornarla per soddisfare una versione più recente di torchvision; se la risoluzione fallisce, controllare la coppia torch/torchvision prima di procedere.

```powershell
New-Item -ItemType Directory -Force outputs\trufor | Out-Null
python -c "import torch; print('torch==' + torch.__version__)" | Set-Content -Encoding ascii outputs\trufor\torch-constraint.txt
python -m pip install -e '.[trufor,dev]' -c outputs\trufor\torch-constraint.txt

New-Item -ItemType Directory -Force external | Out-Null
git clone https://github.com/grip-unina/TruFor.git external\TruFor
git -C external\TruFor checkout ae54475df6f41a491d7615100feb19263dec13f7
```

La clonazione serve una volta. Se la cartella esiste già, eseguire soltanto il checkout dopo aver verificato che sia il repository corretto. Il codice controlla revisione e assenza di modifiche alla libreria ufficiale.

Scaricare i pesi dalla sorgente degli autori. Il checksum MD5 è quello pubblicato nel loro README; il loader controlla anche lo SHA-256 del checkpoint estratto. La licenza del repository ufficiale si applica a codice e pesi.

```powershell
New-Item -ItemType Directory -Force checkpoints\TruFor | Out-Null
Invoke-WebRequest 'https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip' -OutFile checkpoints\TruFor\TruFor_weights.zip
if ((Get-FileHash checkpoints\TruFor\TruFor_weights.zip -Algorithm MD5).Hash.ToLower() -ne '7bee48f3476c75616c3c5721ab256ff8') { throw 'Checksum TruFor non valido' }
Expand-Archive checkpoints\TruFor\TruFor_weights.zip -DestinationPath checkpoints\TruFor -Force
```

Il checkpoint atteso è `checkpoints\TruFor\weights\trufor.pth.tar`. Le directory `external`, `checkpoints` e `outputs` sono escluse da Git.

## Preparazione e verifica della GPU

Impostare i due percorsi effettivi. `$ctRoot` è la cartella che contiene `pix2pix`, non `pix2pix` stessa. `$realCtRoot` contiene le serie autentiche. Gli esempi seguenti vanno adattati ai dischi presenti nel laboratorio:

```powershell
$ctRoot = 'D:\Dataset Tesi Piscopo'
$realCtRoot = 'C:\Tesi Magistrale Piscopo\real\scan'
Test-Path (Join-Path $ctRoot 'pix2pix\scan')
Test-Path (Join-Path $ctRoot 'pix2pix\label')
Test-Path $realCtRoot

python -m tesi_m3d.trufor prepare --data-root "$ctRoot" --real-scan-root "$realCtRoot" --output outputs\trufor\manifest.json
if ($LASTEXITCODE -ne 0) { throw 'Preparazione non completata: risolvere il primo errore mostrato prima del profiling.' }
if (-not (Test-Path outputs\trufor\manifest.json -PathType Leaf)) { throw 'Manifest non trovato.' }
python -m tesi_m3d.trufor profile --manifest outputs\trufor\manifest.json --output outputs\trufor\profile.json
```

Il manifest viene scritto soltanto dopo la verifica di tutte le serie. Attendere il messaggio `Preparation complete` prima del profiling. Se `prepare` si interrompe, ripetere lo stesso comando: la cache dei percentili delle serie già completate viene riutilizzata. Se il manifest è stato salvato con un altro nome o in un'altra cartella, indicare quel file con `--manifest`. I percorsi relativi vengono risolti dalla directory corrente di PowerShell.

Il profiling usa quattro slice di training, comprende forward, backward e optimizer, e verifica entrambe le varianti alla risoluzione nativa. Riporta memoria GPU allocata, tempo per slice e caricamento iniziale con i quattro worker. Non è una stima completa del tempo per epoca: non include il costo della validazione e conserva i quattro esempi in memoria durante la misura del modello.

Prima dei run completi eseguire il controllo di overfitting:

```powershell
python -m tesi_m3d.trufor profile --overfit-steps 100 --output outputs\trufor\overfit.json
```

Il controllo usa due slice positive da una CT e due da una serie autentica. Richiede che la loss si dimezzi, che il F1 aggregato sulle due slice positive raggiunga 0,8 e che la frazione di pixel positivi sulle due slice autentiche non superi 0,01. Queste soglie servono a controllare la capacità di memorizzare pochi esempi; non sono risultati di generalizzazione. Se fallisce, il comando termina con errore e conserva il report. Prima di avviare il training completo occorre esaminare il fallimento; aumentare gli step del controllo è possibile, ma non dimostra da solo l'utilità del metodo.

Le impostazioni del DataLoader sono `num_workers: 4`, `persistent_workers: true`, `prefetch_factor: 1` e `pin_memory: true` con CUDA. I worker leggono soltanto le slice necessarie e non mantengono copie complete delle CT. Se il disco non alimenta abbastanza velocemente la GPU, confrontare 0, 2 e 4 worker modificando il file di configurazione prima di iniziare i run. Non modificare la configurazione di un run già avviato.

## Training e ripresa

Avviare l'adattamento 2D:

```powershell
python -m tesi_m3d.trufor train --mode 2d --stage adapt --output outputs\trufor\adapt_2d --session-hours 4
```

Ogni invocazione dura circa quattro ore e si arresta al successivo punto sicuro. L'ultimo aggiornamento o volume di validazione può far superare leggermente il limite. Per riprendere, aggiungere `--resume` allo stesso comando:

```powershell
python -m tesi_m3d.trufor train --mode 2d --stage adapt --output outputs\trufor\adapt_2d --session-hours 4 --resume
```

Attendere `phase: complete` in `outputs\trufor\adapt_2d\run.json` prima di avviare le prosecuzioni. Il codice verifica che l'adattamento sia terminato.

```powershell
python -m tesi_m3d.trufor train --mode 2d --stage continue --init-from outputs\trufor\adapt_2d\best.pt --output outputs\trufor\continue_2d --session-hours 4

python -m tesi_m3d.trufor train --mode context --stage continue --init-from outputs\trufor\adapt_2d\best.pt --output outputs\trufor\context --session-hours 4
```

Eseguire una prosecuzione alla volta sulla GPU. Anche questi comandi si riprendono aggiungendo `--resume`. Nel weekend si può aumentare `--session-hours`: cambiano le ore della sessione, non il numero di epoche del confronto.

Un primo Ctrl+C richiede l'arresto ordinato; attendere il messaggio di salvataggio prima di chiudere PowerShell. `last.pt` contiene optimizer, scheduler, scaler, stato casuale e posizione del campionamento. I checkpoint periodici vengono scritti ogni 100 aggiornamenti o dopo circa 10 minuti, al primo aggiornamento utile. Una chiusura forzata perde il lavoro dall'ultimo checkpoint. La scrittura usa un file temporaneo e una sostituzione atomica.

Le trasformazioni dipendono dal token dell'esempio, non dallo stato casuale del worker. Dopo una ripresa, gli esempi preparati in anticipo ma non consumati vengono rigenerati. La validazione salva un risultato per volume; una ripresa ricomincia dal primo volume incompleto.

## Calibrazione e test completo

Per ogni modello, la soglia viene scelta sulla validazione completa fra 0,01 e 0,99, massimizzando il F1 medio per CT manipolata. A parità si sceglie la soglia maggiore. Il test richiede il relativo `calibration.json` e controlla che appartenga allo stesso checkpoint.

Esempio per la prosecuzione 2D:

```powershell
python -m tesi_m3d.trufor evaluate --checkpoint outputs\trufor\continue_2d\best.pt --split valid --output outputs\trufor\continue_2d\valid --session-hours 4

python -m tesi_m3d.trufor evaluate --checkpoint outputs\trufor\continue_2d\best.pt --split test --calibration outputs\trufor\continue_2d\valid\calibration.json --output outputs\trufor\continue_2d\test --session-hours 4
```

Ripetere sostituendo `continue_2d` con `context` e, se si vuole riportare anche l'adattamento iniziale, con `adapt_2d`. Se la valutazione si interrompe, ripetere esattamente il comando: non serve `--resume`. Prima del test attendere la creazione di `calibration.json`.

`report.json` contiene F1 e IoU per volume, medie macro, aggregazione micro e risultati separati per aggiunte e rimozioni. Riporta sia la soglia calibrata sia 0,5. Le serie autentiche hanno un riepilogo separato dei falsi positivi: non alzano il F1 medio delle CT manipolate attraverso maschere vuote correttamente predette.

Le anteprime PNG nella cartella `volumes` mostrano, da sinistra, CT, ground truth verde e predizione rossa a soglia 0,5. Per una CT manipolata, la slice illustrata è quella con più voxel positivi nella ground truth; per una CT autentica è quella con più pixel predetti positivi. Questa selezione riguarda soltanto la visualizzazione: la valutazione elabora tutte le slice. `preview_z` identifica la slice nel JSON.

Non si inverte la predizione scegliendo l'orientamento migliore mediante ground truth e non si escludono i bordi delle manipolazioni. Il [README di TruFor](https://github.com/grip-unina/TruFor/blob/ae54475df6f41a491d7615100feb19263dec13f7/TruFor_train_test/README.md) descrive convenzioni diverse. Prima di dichiarare un miglioramento rispetto alla tabella M3Dsynth, vanno verificati split, unità di valutazione, selezione delle immagini, gestione dei bordi e soglia usati dagli autori. Il confronto interno 2D/contesto usa invece lo stesso protocollo.

## Inferenza su una CT sospetta

Servono soltanto le immagini TIFF, lo spacing, il checkpoint e la soglia già calibrata. Il comando non riceve maschere, coordinate della manipolazione o immagini autentiche appaiate. Lo spacing è nell'ordine z, y, x e va ricavato dai metadati della serie; i valori seguenti sono un esempio da sostituire.

```powershell
python -m tesi_m3d.trufor predict --scan-dir 'D:\CT_sospetta' --spacing-zyx 2.5 0.7 0.7 --checkpoint outputs\trufor\context\best.pt --calibration outputs\trufor\context\valid\calibration.json --output outputs\trufor\predizione_ct
```

L'output contiene `probabilities.npy` float32 e `mask.npy` booleana, entrambi in ordine z,y,x. `prediction.json` registra spacing, normalizzazione, soglia e hash del checkpoint. La maschera è `P >= soglia`; non viene sostituita con un bounding box. Una CT 300×512×512 richiede circa 300 MiB per le probabilità e 75 MiB per la maschera. La scrittura usa array su disco senza trattenere tutto il volume in RAM.

La directory di predizione deve essere vuota. Se il processo viene interrotto, l'assenza di `prediction.json` segnala un output incompleto; usare una nuova directory per ripetere l'inferenza. Le coordinate sono quelle degli indici TIFF originali: non viene ricostruita una trasformazione spaziale DICOM completa.

## Verifiche del codice

```powershell
python -m pytest -q
$env:TRUFOR_ROOT = (Resolve-Path external\TruFor).Path
$env:TRUFOR_WEIGHTS = (Resolve-Path checkpoints\TruFor\weights\trufor.pth.tar).Path
python -m pytest tests\test_trufor.py -q
```

Il test con i pesi ufficiali controlla equivalenza con TruFor, identità iniziale del contesto e backward su CPU a 64×64. Gli altri test verificano cache, maschere, sampling, worker con avvio spawn, ripresa del training, metriche e inferenza senza ground truth. Questi controlli non sostituiscono il profiling a risoluzione nativa sulla GPU del laboratorio.

La prima versione non comprende cross-generator, consistenza fra crop o postprocessing delle componenti. Il seed iniziale è uno solo: prima di attribuire significato a differenze di pochi punti, sarà opportuno ripetere il confronto con più seed o quantificare l'incertezza per paziente.
