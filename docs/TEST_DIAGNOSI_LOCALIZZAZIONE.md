# Esecuzione delle verifiche aperte sulla localizzazione

Questi comandi preparano le misure dei punti 4–6, 8 e 9 del
[registro](REGISTRO_DIAGNOSI_LOCALIZZAZIONE.md). La disponibilità degli strumenti
non chiude le ipotesi: servono i risultati sui TIFF e sui checkpoint del laboratorio.
Eseguire dalla root del progetto, nell'ambiente `tesi-m3d`, con le dipendenze
`.[train,dev]` già installate. Gli esempi usano PowerShell.

## 1. Soglia e target: mantenere separate le due ablation

Usare i due training già documentati nel registro:
`train_pix2pix_overlap_1pct.yaml` cambia solo la soglia all'1% di overlap (`0.01`);
`train_pix2pix_soft_targets.yaml` cambia solo il target in `soft_score`, mantenendo
la soglia `0.05`. Il test `test_ablation_configs_change_one_factor_only` verifica
che, esclusi nome e directory di output, ogni configurazione differisca dalla
baseline soltanto nel parametro dichiarato. Avviare training nuovi, senza resume.

Dopo i training, confrontare baseline e ablation con **la configurazione comune
baseline al 5%**, evitando che cambino insieme score e definizione di positivo:

```powershell
python -m tesi_m3d.diagnose_localization `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --checkpoint outputs\train_pix2pix_balanced_patches\best.pt outputs\train_pix2pix_overlap_1pct\best.pt outputs\train_pix2pix_soft_targets\best.pt `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda `
  --out-dir outputs\diagnosi_ablation_5pct
```

È possibile iniziare con il solo checkpoint baseline. Tutti i checkpoint elencati
vengono valutati sulle medesime scansioni e sulle medesime patch campionate.
In `metrics.json`, per ogni volume, `train_overlap_audit` e `dense_overlap_audit`
contano positive, negative ed escluse alle soglie 1% e 5%, rispettivamente sulla
griglia di training e su quella densa. Queste sono statistiche delle label:
per misurare l'effetto sul modello occorrono i training separati.

## 2. Ranking, heatmap e allineamento sugli stessi casi

Il comando precedente produce una sottodirectory numerata per checkpoint con:

| File | Uso |
| --- | --- |
| `metrics.json` | AP, AUC e prevalenze campionate/dense; statistiche per volume, top-k hit, frazione della mask coperta dai top-k e rango della patch a maggior overlap |
| `patches.csv` | Patch con score più alto, maggior overlap e score più basso tra quelle che intersecano la mask; score, `soft_score`, coordinate e distanze |
| `NNNN_scores.npz` | Tutti gli score densi, overlap, coordinate degli inizi patch e geometria della griglia |
| `heatmaps.csv` | AUC/AP, precision, recall, Dice e IoU per volume, aggregazione, soglia e regione |
| `local_heatmaps.csv` | AUC/AP della heatmap ricostruita dentro le patch esaminate e intervallo degli score voxel |
| `NNNN_alignment.json` | Forme originali/allineate, voxel positivi scartati dal ritaglio, bbox mask, coordinate metadata e quota di mask fuori dalla ROI polmonare |
| `NNNN_average_*.png`, `NNNN_gaussian_*.png` | Stesse sezioni nei tre piani: CT/mask/ROI, heatmap/griglia/patch annotate, falsi positivi e negativi |

`NNNN` è l'indice del record in `manifest.json`, evitando collisioni fra generatori.
La ROI delle metriche polmonari è `lung | mask`; l'overlay mostra la ROI polmonare
originale, così sono visibili i positivi che l'unione avrebbe recuperato.
Le distanze sono in **voxel**, senza inferire millimetri o orientamento DICOM:
`distance_to_mask_voxels` misura la minima distanza fra la scatola della patch e
un voxel annotato, mentre `center_distance_voxels` confronta i centroidi.
La prima resta significativa anche con lesioni disconnesse.
Le proporzioni dei pannelli rispettano gli assi in voxel; non correggono
l'anisotropia fisica. L'allineamento usato è quello del loader esistente:
nessuna registrazione automatica e nessuna modifica degli originali.

Per concentrare il controllo sui casi critici, specificare gli ID effettivi della
validation configurata, ad esempio aggiungendo `--img-ids rem_14 inj_123` con ID
presenti nel proprio sottoinsieme. Gli ID estranei alla selezione sono rifiutati.
`--max-report-images 4` limita il numero di **volumi** con immagini, reali compresi;
le metriche restano calcolate su tutti i volumi selezionati. Per visualizzare ogni
caso selezionato, usare un limite almeno pari al loro numero.

Le immagini comprendono sezioni alle coordinate metadata, al centro della mask,
alla patch top-score e alla patch intersecante con score più basso.
Questi ultimi sono candidati errori di ranking; FP/FN voxel sono determinati dalla
soglia esplicita nei pannelli. La scala colore della heatmap è sempre `[0,1]`.
Le soglie predefinite sono `0.01 0.05 0.1 0.25 0.5 0.75 0.9 0.99` e la prima
viene usata per le immagini. Per controllare una soglia calibrata, leggere
`localization.threshold` dal `calibration.json` del run e passarla **per prima**
a `--thresholds`, seguita dalle altre soglie desiderate. Il comando non importa
implicitamente calibrazioni che potrebbero riferirsi a un'altra aggregazione.
Entrambe le aggregazioni sono ricostruite dagli stessi score patch.
Una griglia con voxel scoperti causa un errore esplicito della ricostruzione.

Le metriche AUC/AP locali usano score voxel della heatmap ricostruita, mai il
singolo score costante del classificatore. Sono riportate anche accanto alle
patch nelle immagini, con la dicitura `3D patch HM AUC=... AP=...`: si riferiscono
all'intera patch 3D e all'aggregazione mostrata, non alla sola sezione visibile.
Se manca una delle due classi sono
non definite (`null` in JSON, `nan` nei CSV); ciò vale anche per volumi reali.
Nelle immagini le metriche non definite sono indicate con `n/a`.
Un'AP non definita non viene sostituita con zero.

## 3. Validation campionata contro densa e classifica dei checkpoint

`sampled_validation.npz` salva record, coordinate e label del campionamento fisso;
`manifest.json` contiene configurazione, record originali, indici selezionati e
soglie. Il campionamento usa lo stesso `SequentialVolumeBatchSampler` del training,
conservando tutte le positive e applicando il limite alle negative.
I record duplicati di scansioni reali sono mantenuti per riprodurre il sampler;
la valutazione densa usa una sola copia per percorso scan. Se è configurato
`max_valid_volumes`, la griglia densa è limitata agli stessi volumi selezionati.
`--img-ids` restringe il campione: il risultato va dichiarato come sottoinsieme,
senza presentarlo come AP dell'intera validation.

`comparison.json` confronta AP/prevalenza campionata e densa dei checkpoint e
riporta entrambe le classifiche (`*_ap_rank`, con parità a pari rango).
Per la classifica delle epoche, passare più checkpoint dello **stesso training**:

```powershell
python -m tesi_m3d.diagnose_localization `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --checkpoint outputs\train_pix2pix_balanced_patches\checkpoint_epoch010.pt outputs\train_pix2pix_balanced_patches\checkpoint_epoch013.pt outputs\train_pix2pix_balanced_patches\checkpoint_epoch016.pt `
  --data-root "C:\Tesi Magistrale Piscopo" --device cuda `
  --max-report-images 0 `
  --out-dir outputs\diagnosi_ranking_epoche
```

Usare soltanto checkpoint effettivamente prodotti (l'early stopping può fermarsi
prima). La diagnostica esegue inferenza float32 senza autocast: l'AP campionata
ricalcolata può differire lievemente dalla `valid_ap` del training in AMP.
`dense_supervised` esclude gli overlap fra zero e soglia, come il report patch
esistente. `dense_all_boundary_as_negative` include tutte le patch, dichiarando
negative quelle sotto soglia: tenere distinte le due definizioni.
La prevalenza è il riferimento casuale dell'AP. Confrontare i modelli sulla stessa
definizione, senza interpretare come equivalenti AP ottenute con prevalenze diverse.

`dense_inference_seconds` misura l'inferenza patch densa (il ritorno degli score su
CPU sincronizza CUDA); `seconds` include anche letture, metriche e immagini.
Il comando usa solo lo split `valid`, non seleziona checkpoint e non riscrive
`best.pt`, `metrics.csv` o le calibrazioni. Il test va tenuto separato.

## 4. Costo delle letture: indice, cache per volume e worker

```powershell
python -m tesi_m3d.benchmark_loading `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --out-dir outputs\benchmark_letture `
  --workers 0 2 --cache-sizes 1 2 `
  --orders grouped shuffled --epochs 2 --rebuild-index
```

Il benchmark usa l'indice e il sampler di training esistenti e disabilita la
validation nella propria copia della configurazione. Ogni scenario legge epoche
complete con lo stesso budget di patch; `shuffled` rimescola gli stessi indici,
senza cambiare esempi o dimensioni dei batch. Il confronto 1/2 volumi residenti e
ordine raggruppato/casuale isola l'utilità della cache e del raggruppamento.
Il run predefinito comprende otto scenari, due epoche ciascuno: per una prima
prova ridurre la matrice con `--cache-sizes 2 --orders grouped`.

`benchmark.json` contiene i tempi di avvio e la configurazione effettiva;
`epochs.csv` riporta tempo per epoca e primo batch, patch/s, chiamate di caricamento
dello stack (`volume_reads`), hit cache e somma dei tempi degli item nei worker.
I contatori sono restituiti nei batch dai worker, inclusi quelli Windows `spawn`.
I worker persistenti rispettano `persistent_workers` della configurazione e
mantengono le proprie cache tra le epoche. Ogni scenario parte con cache volume
vuote. `--rebuild-index` forza la prima costruzione dell'indice nell'output del
benchmark, seguita da un secondo avvio che lo ricarica da disco.
I tempi di avvio includono anche costruzione di modello e optimizer.

`volume_reads` conta le chiamate al loader dopo un cache miss, **non** le letture
fisiche del disco: la cache del sistema operativo non viene svuotata.
`worker_item_seconds` somma tempi concorrenti e non equivale al tempo a parete.
Il benchmark misura solo il caricamento, senza forward/backward o GPU; per il
tempo completo di training restano utilizzabili le configurazioni
`train_pix2pix_workers0_benchmark.yaml` e `train_pix2pix_workers2_benchmark.yaml`.
L'indice dedicato richiede spazio su disco; il numero di volumi residenti è per
worker e quindi la RAM richiesta cresce con i worker.

## Verifica automatica e aggiornamento del registro

```powershell
python -m pytest -q -p no:cacheprovider
```

I test usano TIFF sintetici per verificare soglie di esclusione, ranking/distanze,
mask troncate e fuori ROI, confronto fisso tra checkpoint, report/overlay,
riuso dell'indice senza rilettura delle mask e contatori con worker `spawn`.
Non dimostrano la correttezza delle annotazioni dei pazienti reali.
Conservare gli output in directory distinte e aggiornare il registro soltanto
dopo aver esaminato misure e immagini, indicando split, casi, decisione e limiti.
