# Registro delle verifiche sulla localizzazione delle manipolazioni CT

Configurazione di riferimento:
[`train_pix2pix_balanced_patches.yaml`](../configs/train_pix2pix_balanced_patches.yaml).

## Glossario rapido

Le metriche vanno lette insieme: un valore alto può essere buono per il compito
che misura, senza dimostrare che la regione manipolata sia localizzata bene.

| Termine | Che cosa misura o indica | Lettura valore |
| --- | --- | --- |
| **CT / TC** | Tomografia computerizzata: il volume medico esaminato. | -|
| **DICOM / TIFF** | Formati dei dati: DICOM è il formato medico originale; TIFF contiene le sezioni usate dalla pipeline. | - |
| **Voxel / patch** | Voxel: elemento 3D del volume. Patch: piccolo blocco di voxel dato alla rete. |-|
| **Mask / heatmap** | Mask: regione manipolata annotata. Heatmap: score previsto per ogni voxel. | Una buona heatmap assegna score alti alla mask e bassi altrove. |
| **ROI / lung ROI** | *Region of interest*: area entro cui si calcolano le metriche; | - |
| **ROC-AUC** | *Area Under the Receiver Operating Characteristic Curve*: capacità degli score di ordinare positivi sopra negativi, senza scegliere una soglia. | Da 0 a 1: **1** ordinamento perfetto, **0,5** circa casuale; alto è buono per il ranking, ma non garantisce una buona localizzazione. |
| **Voxel AUC / lung AUC** | ROC-AUC sui voxel dell'intero volume / sui voxel della ROI polmonare. | Alto è meglio; la lung AUC riduce l'effetto dei negativi anatomici facili. |
| **AP** | *Average Precision*: riassume la curva precision–recall usando gli score continui. | Da 0 a 1: alto è meglio; confrontarlo con la quota di positivi, che rappresenta il riferimento casuale. |
| **Lung AP** | AP calcolata solo nella ROI polmonare. | Alto è meglio; utile quando i voxel manipolati sono rari. |
| **Precision / recall** | Precision: quota dei positivi predetti che è corretta. Recall: quota dei positivi reali trovata. | Da 0 a 1: alto è meglio, ma una può crescere a scapito dell'altra. |
| **F1 / Dice** | Sintesi di precision e recall dopo aver scelto una soglia; per due mask binarie coincidono.  Dice misura quanto la mask predetta dal modello si sovrappone alla mask reale della manipolazione. | Da 0 a 1: alto è meglio; vicino a **0** indica scarso accordo spaziale. |
| **IoU** | *Intersection over Union*: sovrapposizione fra mask predetta e reale divisa per la loro unione.  | Da 0 a 1: alto è meglio; **1** significa sovrapposizione perfetta. |
| **Balanced accuracy** | Media fra sensibilità sui positivi e specificità sui negativi. | Da 0 a 1: alto è meglio; **0,5** può derivare anche dal predire sempre una sola classe. |
| **KS / JS** | Distanza di Kolmogorov–Smirnov / divergenza di Jensen–Shannon fra due distribuzioni di intensità. | Vicino a **0** indica distribuzioni simili; non prova che le immagini siano identiche localmente. |
| **`soft_score`** | Frazione della patch coperta dalla mask manipolata; possibile target di training. | Da 0 a 1: più alto significa maggiore copertura, non maggiore certezza del modello. |
| **Train / validation / test** | Dati usati per apprendere / scegliere modello e soglie / valutare su casi tenuti separati. | Le metriche di validation non sono risultati finali di test. |


## Problema osservato

Le heatmap non evidenziano con precisione la regione manipolata, anche quando
alcune metriche di ranking sono alte. Sulla validation della configurazione di
riferimento, la voxel ROC-AUC media sui 30 volumi positivi è circa **0,901**,
mentre la voxel AP media è **0,0111** e il F1 di localizzazione, alla soglia
calibrata, è **0,00916**. Sono misure diverse: la AUC valuta l'ordinamento degli
score; AP e F1 sono più sensibili ai falsi positivi e allo sbilanciamento.
Fonte: [`calibration.json`](../outputs/train_pix2pix_balanced_patches/calibration.json).

## Stato sintetico

| Punto da attenzionare | Stato | Evidenza o prossima verifica |
| --- | --- | --- |
| Differenza globale di intensità fra CT reali e pix2pix | Ridimensionata nel campione train | Audit appaiato su 32 coppie; dettagli al punto 1 |
| Differenze locali o di texture prodotte da pix2pix | Aperto | Confrontare ROI manipolata e tessuto circostante nelle coppie appaiate |
| Metriche globali che nascondono errori nei polmoni | Confermato nella run MedicalNet full | AUC voxel 0,903 contro AUC polmonare 0,569; dettagli al punto 10 |
| Soglia del 5% e target proporzionale alla mask | Da sperimentare | Distinguere effetto della soglia da quello di `soft_score` |
| Ranking e copertura delle patch positive | Problema osservato, causa aperta | Analizzare score, overlap e distanza delle patch meglio classificate |
| Ricostruzione, soglia e visualizzazione della heatmap | Aperto | Confrontare score patch, metriche locali e overlay sulle stesse scansioni |
| Allineamento scan, mask e ROI polmonare | Da verificare su esempi critici | Controllo visivo su sezioni e coordinate dei falsi positivi/negativi |
| Presenza della validation loss | Chiarito: presente | Valori per epoca in `metrics.csv`; il best usa `valid_ap` |
| Validation densa e rappresentativa | Criticità osservata | Confrontare prevalenza e AP della validation campionata e della griglia densa |
| Rilettura dei volumi durante il training | Misurata; costo residuo presente | Quattro worker e cache percentili: circa 66 s contro 78 s nel benchmark bounded; dettagli al punto 10 |

## 1. Differenza globale di intensità fra reali e pix2pix

**Ipotesi.** La rete potrebbe separare i volumi in base a intensità, offset o
caratteristiche di conversione TIFF, invece di riconoscere la manipolazione.
Entrambi i domini sono letti come volumi monocanale: l'eventuale differenza
rilevante è nei valori dati al modello, non nel colore RGB del contenitore TIFF.

**Verifica eseguita.** L'[audit delle intensità](../outputs/intensity_audit_train/summary.json)
ha confrontato 32 coppie reale/pix2pix dello split train, appaiate per immagine
di origine. Tutte le coppie hanno la stessa forma e lo stesso offset stimato.
Sulle distribuzioni normalizzate, la distanza KS è **0,0000764** e la
divergenza Jensen–Shannon è **0,000000708**. Nelle
[`pairs.csv`](../outputs/intensity_audit_train/pairs.csv), la massima differenza
assoluta della media normalizzata per coppia è circa **0,000907**.

**Esito / risoluzione.** In questo campione **non emerge una grande differenza
globale di intensità o di offset** tra le immagini reali convertite in TIFF e
quelle pix2pix. L'ipotesi di un forte artefatto radiometrico globale viene
quindi ridimensionata e non è la prima spiegazione da perseguire per le
heatmap errate. L'audit è uno screening su 32 coppie train: non dimostra
l'assenza di differenze locali, di texture o in altri split. 

## 2. Differenze locali o di texture

**Questione.** Due volumi quasi identici nelle statistiche globali possono
differire molto nella piccola regione manipolata. Una rete potrebbe anche
riconoscere artefatti di generazione fuori dalla mask.

**Verifica da eseguire.** Nelle coppie appaiate, confrontare intensità e texture
dentro la mask, in una corona circostante e in zone lontane. Esaminare esempi
di score alto fuori dalla manipolazione. Il comando
[`audit_local_artifacts`](../src/tesi_m3d/audit_local_artifacts.py) è pronto
per produrre statistiche per regione e immagini di confronto; va eseguito
sui TIFF del PC di laboratorio. Il confronto è solo tra immagini e mask:
per stabilire se la rete usa un artefatto servirà poi confrontarlo con gli
score delle patch.

**Esito / risoluzione.** Da compilare dopo la verifica, indicando esempi,
misure, split e se l'ipotesi è confermata, esclusa o ancora incerta.

## 3. Metriche globali e metriche limitate ai polmoni

**Questione.** Molti voxel negativi sono anatomia facile o sfondo. Possono
alzare la ROC-AUC globale senza garantire una localizzazione utile.

**Verifica da eseguire.** Riportare voxel AUC e AP globali accanto a lung AUC
e lung AP, sugli stessi volumi e con intervalli o distribuzione per volume.
Controllare visivamente la ROI: nel codice la valutazione usa l'unione tra
maschera polmonare e mask manipolata, così i positivi non vengono esclusi.
La configurazione abilita `lung_metrics`, ma nel
[`calibration.json` disponibile](../outputs/train_pix2pix_balanced_patches/calibration.json)
non compaiono ancora `mean_lung_auc_on_positive_volumes` e
`mean_lung_ap_on_positive_volumes`: prima di trarre conclusioni serve produrre
o recuperare questi risultati per il checkpoint di riferimento.

**Esito / risoluzione.** Da compilare dopo aver confrontato i risultati
effettivi e verificato la copertura della ROI.

## 4. Soglia delle patch positive, ranking e target di training

**Osservazione.** Il
[`report patch della baseline bilanciata`](../outputs/train_pix2pix_balanced_patches/patch_level_report/metrics.json)
riporta AUC patch **0,873**, AP patch **0,0110**, top-1 hit rate **0** e
top-3 hit rate **0,10** sui 30 volumi positivi della validation. Un hit
richiede solo intersezione non nulla con la mask: questi risultati segnalano
che le patch più votate spesso non sono quelle manipolate.

**Verifica da eseguire.** Per i casi peggiori, mettere in relazione score,
quota di mask nella patch (`soft_score`), distanza dalla mask e posizione.
La soglia attuale del **5%** assegna label positiva alle patch con almeno
quella quota di mask; quelle con intersezione non nulla ma inferiore al 5%
sono escluse dall'indice di training. Verificare quante patch vengono escluse
e provare una soglia più permissiva come esperimento separato. Il codice
calcola già `soft_score` come proporzione di voxel manipolati e la
[`configurazione soft`](../configs/train_pix2pix_soft_targets.yaml) lo usa come
target, ma non è presente un output del relativo training. Confrontare tale
ablation con il target binario a parità degli altri parametri e della
validation; non attribuire a `soft_score` l'effetto di un eventuale cambio
simultaneo della soglia. Verificare anche la prevalenza molto bassa delle
patch positive nel report denso.

**Comandi per il PC del laboratorio.** Dalla root del progetto, con
l'ambiente `tesi-m3d` attivo, avviare due training nuovi, senza `--resume`:

```powershell
python -m tesi_m3d.train `
  --config configs\train_pix2pix_overlap_1pct.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda

python -m tesi_m3d.train `
  --config configs\train_pix2pix_soft_targets.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda
```

Il primo cambia solo `positive_overlap_fraction` da `0.05` a `0.01` e mantiene
il target binario; il secondo mantiene `0.05` e cambia solo il target in
`soft_score`. Ogni run scrive in una directory distinta. Dopo i training,
valutare **entrambi** i checkpoint con la definizione baseline del 5%,
per confrontare le stesse patch positive:

```powershell
python -m tesi_m3d.evaluate_patch_level `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --checkpoint outputs\train_pix2pix_overlap_1pct\best.pt `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --split valid --device cuda `
  --out-dir outputs\train_pix2pix_overlap_1pct\patch_level_report_common_5pct

python -m tesi_m3d.evaluate_patch_level `
  --config configs\train_pix2pix_balanced_patches.yaml `
  --checkpoint outputs\train_pix2pix_soft_targets\best.pt `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --split valid --device cuda `
  --out-dir outputs\train_pix2pix_soft_targets\patch_level_report_common_5pct
```

Per discutere i risultati, conservare `metrics.csv`, `calibration.json` e
`patch_level_report_common_5pct/metrics.json` di ciascun run, insieme alla
configurazione usata. Le `valid_ap` dei due training non sono direttamente
comparabili perché la prova all'1% cambia anche quali patch ricevono label
positiva nella validation campionata.

**Esito / risoluzione.** Da compilare dopo l'analisi e l'eventuale ablation;
indicare se il problema nasce già dagli score delle patch.

## 5. Ricostruzione e calibrazione della heatmap

**Questione.** L'aggregazione delle patch sovrapposte e la soglia possono
diffondere o cancellare un segnale corretto a livello patch. Una heatmap
visivamente fuorviante può anche derivare dalla scelta della sezione mostrata.

**Evidenza disponibile.** Nel
[`confronto delle aggregazioni`](../outputs/train_pix2pix_balanced_patches/aggregation_comparison.json),
`average` ha F1 **0,00917** e `gaussian` **0,00777** sulla validation; nessuna
delle due produce una buona localizzazione in questa prova.

**Verifica da eseguire.** Sovrapporre mask, griglia, score patch e heatmap sullo
stesso volume. Controllare zone senza copertura e scelta della soglia sulla
validation. Separare gli errori già presenti negli score patch da quelli
introdotti dalla ricostruzione. Annotare sulle immagini, per le patch
esaminate, score del modello, quota di mask coperta e distanza dalla mask.
Il classificatore dà un solo score a tutta la patch: una voxel ROC-AUC
calcolata su quello score costante non è informativa; è **non definita** se
nella patch c'è una sola classe di voxel. Le metriche voxel locali sulla
heatmap ricostruita si possono aggiungere dove entrambe le classi sono
presenti. Una patch lontana dalla mask con score alto segnala già
un errore di ranking del modello; un problema di visualizzazione è suggerito
se posizione o intensità mostrate non corrispondono agli score e alle
coordinate delle patch.

**Esito / risoluzione.** Da compilare dopo il confronto caso per caso.


## 6. Allineamento delle annotazioni e qualità dei casi

**Questione.** Disallineamenti fra stack, mask, orientamento o ROI alterano
sia le label delle patch sia la valutazione voxel. Il loader gestisce
differenze di profondità, ma il risultato va verificato sui casi critici.

**Verifica da eseguire.** Sovrapporre la mask alle sezioni originali e alla
heatmap per falsi positivi, falsi negativi e casi con differenze di profondità.
Registrare eventuali casi esclusi o corretti senza alterare i file originali.

**Esito / risoluzione.** Da compilare con identificativi dei casi e risultato
del controllo visivo.

## 7. Validation loss e scelta del checkpoint

**Verifica eseguita.** [`evaluate_patch_loader`](../src/tesi_m3d/train.py)
calcola la loss sulla validation a ogni epoca con la stessa funzione del
training e la salva come `valid_loss`. Nel
[`metrics.csv` del run bilanciato](../outputs/train_pix2pix_balanced_patches/metrics.csv)
vale **0,0425** all'epoca 1, **0,0205** all'epoca 10 e **0,0268** all'epoca
16. Il massimo `valid_ap` è **0,2275** all'epoca 13. Il codice sceglie
`best.pt` ed esegue early stopping in base a `valid_ap`, non a `valid_loss`.

**Esito / risoluzione.** **La validation loss non manca.** È disponibile per
monitorare l'andamento, ma non decide quale checkpoint conservare. Rimane da
valutare se l'AP usata per tale scelta sia rappresentativa del compito di
localizzazione; questo è il punto 8.

## 8. Rappresentatività della validation

**Questione.** La validation per epoca usa patch campionate, con un limite
nominale di 64 patch per volume: conserva tutte le positive e usa le negative
per riempire i posti rimanenti. Nel run
bilanciato comprende **3712 patch**, di cui **184 positive (4,96%)** per
epoca. Il
[`report su griglia densa`](../outputs/train_pix2pix_balanced_patches/patch_level_report/metrics.json)
considera **702519 patch**, di cui **1483 positive (0,211%)**. L'AP è circa
**0,227** nella validation campionata all'epoca migliore e **0,0110** nel
report denso del checkpoint: i due insiemi hanno prevalenze diverse, quindi
non sono AP direttamente intercambiabili. La differenza conferma però che
l'AP campionata non descrive la distribuzione delle patch incontrate in
inferenza densa.

**Verifica da eseguire.** Definire una validation densa o un sottoinsieme
fisso e rappresentativo, misurare costo e metriche su di esso, poi confrontare
la classifica dei checkpoint con quella prodotta da `valid_ap`. Mantenere
separata la scelta del modello dai risultati di test.

**Esito / risoluzione.** Da compilare dopo il confronto dei checkpoint; la
presenza della loss da sola non risolve la rappresentatività della validation.

## 9. Rilettura dei volumi e costo del training

**Verifica del codice.** L'indice delle patch viene costruito o caricato da
cache; durante il training il dataset usa una cache limitata di volumi
normalizzati (`volume_cache_size: 2`) e batch raggruppati per volume. Le
mask non vengono rilette a ogni richiesta di patch. Questo evita la rilettura
completa dello stack per ogni singola patch dello stesso volume.

**Verifica da eseguire.** Misurare tempo e letture effettive per epoca sul PC
di training, anche con due worker. La cache può comunque essere svuotata o
non condivisa tra worker, e i volumi possono essere riletti in epoche diverse.

**Esito / risoluzione.** Il problema prestazionale **è mitigato nel codice**;
non è ancora dimostrato che il costo residuo sia trascurabile sul run reale.


## Strumenti pronti per le verifiche aperte

La [guida operativa](TEST_DIAGNOSI_LOCALIZZAZIONE.md) raccoglie i comandi per il
PC del laboratorio e la descrizione degli output. I punti restano aperti fino
alla verifica sui dati reali:

- `python -m tesi_m3d.diagnose_localization`: audit delle esclusioni all'1% e
  al 5%, ranking/overlap/distanze e copertura della mask nei top-k; confronto
  `average`/`gaussian` e soglie sugli stessi score; metriche globali, polmonari
  e locali; overlay nei tre piani e controllo delle forme scan/mask e della ROI.
  Con più checkpoint confronta anche AP e classifica su validation campionata
  fissa e griglia densa, usando una configurazione comune.
- `python -m tesi_m3d.benchmark_loading`: confronto fra 0/2 worker, capacità
  della cache e batch raggruppati/casuali, tempi per epoca di caricamento,
  contatori restituiti dai worker e avvio con costruzione/riuso dell'indice.
  Non misura il tempo del forward/backward né le letture fisiche del disco.

I test automatici usano TIFF sintetici; non sostituiscono il controllo visivo
delle annotazioni né i tempi sul PC di training. Le due configurazioni di
ablation già presenti rimangono separate: soglia all'1% con target binario,
oppure soglia al 5% con target `soft_score`.

## 10. Evidenze MedicalNet full e protocollo di recupero

**Verifica eseguita.** La run
[`train_pix2pix_full_train_resnet18_medicalnet`](../outputs/train_pix2pix_full_train_resnet18_medicalnet/patch_level_report/metrics.json)
ha 58 scansioni validation, 30 manipolate: AP patch 0,01041, top-1 hit
1/30, top-5 hit 4/30. La
[`calibrazione`](../outputs/train_pix2pix_full_train_resnet18_medicalnet/calibration.json)
sceglie soglia voxel 0,46 con Dice micro 0,01118. AUC voxel media 0,90289,
AP voxel media 0,00803; nella ROI polmonare unita alla mask, AUC 0,56918
e AP 0,06288. La sola ROI polmonare copre mediamente il 74,89% della mask.

**Decisione.** Il problema è già nel ranking delle patch. Né la sostituzione
dell'encoder né l'aumento dei dati hanno risolto la localizzazione; non basta
modificare la soglia o ritoccare le immagini. Non usare la ROI polmonare per
tagliare le predizioni. Queste evidenze aggiornano i punti 3, 4, 5 e 8;
non dimostrano una causa unica e non chiudono l'audit delle annotazioni.

**Prestazioni.** Il
[`benchmark con percentili riusati`](../outputs/benchmark_percentile_cache/benchmark.json)
misura 65,93 e 65,74 secondi, rispetto a 79,39 e 77,16 secondi con quattro
worker nel benchmark precedente: circa -16% nel solo caricamento. La cache
filesystem di Windows contribuisce al confronto; non è una misura del
risparmio totale di training GPU e i TIFF vengono ancora decodificati.

**Intervento implementato, da validare sul dataset.** Il protocollo della
[sezione 18 della guida laboratorio](LAB_WORKFLOW.md#18-recupero-della-localizzazione-segmentazione-medicalnet)
aggiunge supervisione voxel con encoder MedicalNet e decoder spaziale,
BatchNorm dell'encoder con statistiche fisse, crop positivi anche con piccola
intersezione e negativi polmonari/globali. Seleziona il checkpoint con AP
densa su tutti i voxel; conserva separatamente l'AP media sui positivi.
Un confronto controllato con/senza clipping cambia solo quel preprocessing.
La cache percentili resta una modifica prestazionale distinta.

Preparazione e gate impediscono di saltare audit, verifica di apprendimento
e confronto pilota. La preparazione controlla sequenze/forme/mask e produce
audit di 32 coppie train, bilanciate inj/rem, con figure nei tre piani.
Le mask etichettano la manipolazione disponibile: un riquadro non equivale
al contorno preciso di una lesione. Le differenze fake/reale non diventano
etichette automatiche. La modalità legacy mantiene il proprio allineamento;
la segmentazione vieta il taglio di voxel annotati positivi.

**Criteri da verificare.** Overfit: Dice crop positivi >=0,90 e FP voxel su
crop negativi <1%. Pilota: Dice micro >=0,10, picco corretto >=2/4,
AP voxel media almeno tripla della baseline comune e FP sui reali non
peggiori. Full: Dice medio positivo >=0,30 e picco corretto >=15/30,
mantenendo gli stessi vincoli AP/FP. Il picco voxel e il vecchio hit di una
patch sono misure diverse; il confronto avviene col nuovo valutatore comune.

**Limite e stato.** Nessun criterio scientifico è dichiarato superato solo
perché implementato. Servono le esecuzioni sui TIFF e la revisione visiva
delle annotazioni. I report cross-generator usano checkpoint e soglie
congelati; risultati soltanto su pix2pix non chiudono la generalizzazione.

### 10.1 Prima prova overfit della segmentazione: gate non superato

**Verifica eseguita.** La run
[`segmentation_overfit`](../outputs/segmentation_overfit/gate.json), manifest
`0de4718f8115e9c7ca10bd75f32350e214bddb63a105a12b10406c6c86594200`,
ha completato 50 epoche, 300 aggiornamenti su sei scansioni train: due
injection, due removal e due reali. Crop fissi, nessuna augmentation,
normalizzazione senza clipping, encoder MedicalNet aggiornato fin dall'inizio.
Il gate restituisce Dice medio dei crop positivi **0,70994**, frazione di voxel
predetti positivi sui crop negativi **0,016996** e gradienti finiti.
Fallisce entrambi i criteri spaziali (>=0,90 e <0,01).

**Curva e immagini.** La [loss](../outputs/segmentation_overfit/metrics.csv)
passa da 0,55507 a 0,29618; la media delle epoche 41–45 è 0,31639 e quella
delle epoche 46–50 è 0,29142. Non emerge un plateau della loss in questa
finestra; manca però la storia intermedia del Dice e dei falsi positivi.
I primi crop dei quattro manipolati mostrano sovrapposizione con il target,
con predizioni spesso più ampie e bordi arrotondati. Non mostrano un
grossolano scambio degli assi nelle sezioni visualizzate. I reali
[`inj_1713`](../outputs/segmentation_overfit/images/04_inj_1713.png) e
[`rem_605`](../outputs/segmentation_overfit/images/05_rem_605.png)
mostrano falsi positivi: il prefisso del nome non indica il dominio,
che nel manifest è `real` e richiede target vuoto.

**Annotazioni e audit appaiato.** I quattro manipolati overfit hanno
`bbox_fill_fraction=1`; lo stesso vale per tutti i 78 record manipolati
controllati dall'[audit iniziale](../outputs/segmentation_protocol/audit.json).
Sono riquadri volumetrici pieni, non contorni anatomici di tumore.
Nell'[audit delle differenze locali](../outputs/segmentation_protocol/paired_artifacts/regions.csv),
per `inj_2092` e `inj_1419` cambia rispettivamente il 42,45% e il 46,45%
dei voxel dentro la mask, senza differenze nelle regioni esterne esaminate.
Per `rem_133` cambia il 93,80% del target e le differenze esterne sono
al massimo un'unità nei valori TIFF. Per `rem_15`, corona e tessuto lontano
hanno differenza assoluta di circa 1024, con differenze dei gradienti quasi
nulle: compatibile con un offset additivo, senza dimostrarne segno o causa.
Le differenze raw non vanno confuse con quelle dopo normalizzazione né
usate per sostituire la mask.

**Decisione e limite.** Piloti e full restano bloccati. Non cambiare soglia,
budget o criterio per trasformare questo risultato in un successo.
Il risultato dimostra apprendimento parziale sui crop del train; non misura
la localizzazione su volumi completi e non è confrontabile direttamente
con il Dice della baseline su validation. Prima di cambiare ottimizzazione
o decoder, rivalutare il checkpoint con la diagnostica della sezione 18.2
della guida: tutti i crop, precision/recall, errori per scansione e confronto
float32/bfloat16 alla stessa soglia. La nuova diagnostica non modifica pesi
o gate originale. Checkpoint e TIFF sono sul PC Windows; la rivalutazione
successiva e i report sincronizzati sono documentati nella sezione 10.2.

### 10.2 Diagnostica di tutti i crop e confronto delle precisioni

**Verifica eseguita.** La
[diagnostica del checkpoint a 300 aggiornamenti](../outputs/segmentation_overfit/diagnostics/report.json)
`665d87e234721605fbeb91401ecdd9f0268739f9e51e6f602fa331ad956d1755`
è stata eseguita sul PC del laboratorio e i report sono stati sincronizzati.
Alla soglia 0,5, float32 dà Dice medio 0,709938 e FP sui negativi 1,69957%;
bfloat16 dà Dice 0,709819 e FP 1,70363%. La differenza di Dice è 0,000119,
con massimo scarto per crop di circa 0,000594: in questo checkpoint il cambio
di precisione non spiega il fallimento del gate.

**Errori misurati.** Nei
[16 crop positivi](../outputs/segmentation_overfit/diagnostics/fp32/crops.csv)
il recall medio per crop è 94,22%, la precision media è 61,04%. I conteggi
aggregati sono TP=246.575, FP=136.112, FN=27.344: la precision micro è
64,43% e il recall micro 90,02%. Non confondere medie per crop e conteggi
globali. Solo un crop positivo su 16 raggiunge Dice >=0,90; 13/16 hanno
recall >=0,90. Il crop 3 di `inj_2092` ha 1.300 voxel target, tutti trovati,
ma 12.638 falsi positivi: Dice 0,17063 e precision 9,33%. La sua figura
mostra una piccola porzione di target al bordo del crop e una predizione
molto più ampia. I crop negativi non intersecano il target secondo il
sampler; 17/32 superano comunque l'1% di voxel predetti positivi.

| Scansione | Dominio | Dice medio crop positivi | FP voxel nei suoi crop negativi |
| --- | --- | ---: | ---: |
| inj_2092 | pix2pix | 0,54868 | 1,35% |
| inj_1419 | pix2pix | 0,78602 | 4,12% |
| rem_133 | pix2pix | 0,69577 | 1,69% |
| rem_15 | pix2pix | 0,80928 | 1,12% |
| inj_1713 | real | non definito | 0,51% |
| rem_605 | real | non definito | 2,15% |

I [conteggi per caso](../outputs/segmentation_overfit/diagnostics/fp32/cases.csv)
non indicano un errore isolato: tutte le scansioni manipolate hanno Dice
medio inferiore al criterio aggregato del gate e FP negativi >1%. Nei soli
16 crop reali la frazione FP aggregata è 1,32985%; nei 16 negativi estratti
da pix2pix è 2,06928%. Le figure mostrano anche attivazioni lontane dal
target e su reali. È confermata la sovraestensione delle predizioni;
la causa nel training, nei dati o nell'architettura resta aperta.

**Decisione.** Nessun cambiamento di soglia, clipping, loss, encoder o
budget. Per verificare l'ipotesi di convergenza troppo lenta nel budget
fissato, è preparata una sola replica con
[`segmentation_overfit_decoder_lr.yaml`](../configs/segmentation_overfit_decoder_lr.yaml):
LR decoder 1e-3 invece di 3e-4, nuova inizializzazione identica tramite
seed 21 e MedicalNet, stessi crop e massimo 300 aggiornamenti, output
separati. Un test verifica che nessun altro parametro sperimentale cambi.
L'esito non è ancora disponibile. I piloti restano bloccati dal gate
originale: non promuovere la replica senza esaminarne report e immagini.
Il risultato resta sul train e non dimostra generalizzazione.

## Come aggiornare questo registro

Per ogni punto chiuso, sostituire «Da compilare» con: **verifica eseguita**
(split, numero di casi e metodo), **risultato** (misure e figure), **decisione**
(problema confermato, assente nel perimetro verificato o ancora incerto) e
**limite**. Collegare gli output riproducibili; non chiamare «risolto» un punto
solo perché il codice per misurarlo è presente.
