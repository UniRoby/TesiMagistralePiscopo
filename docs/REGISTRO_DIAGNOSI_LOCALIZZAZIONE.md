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
L'esito successivo è riportato nella sezione 10.3. I piloti restano bloccati
dal gate originale: non promuovere la replica senza esaminarne report e immagini.
Il risultato resta sul train e non dimostra generalizzazione.

### 10.3 Replica con LR decoder 1e-3: criterio Dice ancora fallito

**Verifica eseguita.** Sincronizzati e analizzati
[`gate.json`](../outputs/segmentation_overfit_decoder_lr/gate.json),
[`overfit_metrics.csv`](../outputs/segmentation_overfit_decoder_lr/overfit_metrics.csv),
[`crops.csv`](../outputs/segmentation_overfit_decoder_lr/crops.csv) e immagini
della replica a LR decoder 1e-3. Manifest identico alla prima prova;
configurazione effettiva diversa solo per LR decoder e percorsi degli
output. I conteggi target dei 48 crop coincidono con la prima diagnostica.
Completati 300 aggiornamenti, gradienti finiti e gate `passed: false`.

| Misura alla soglia 0,5 | Prima run, LR 3e-4 | Replica, LR 1e-3 |
| --- | ---: | ---: |
| Dice medio sui 16 crop positivi | 0,70994 | 0,51008 |
| Recall medio sui crop positivi | 94,22% | 41,02% |
| Precision micro nei crop positivi | 64,43% | 87,94% |
| Recall micro nei crop positivi | 90,02% | 40,34% |
| FP voxel nei 32 crop negativi | 1,69957% | 0,17503% |
| FP voxel nei 16 crop reali | 1,32985% | 0,19712% |
| Loss di training dell'ultima epoca | 0,29618 | 0,17756 |

**Errori residui.** I positivi della replica hanno TP=110.499, FP=15.150 e
FN=163.420 su 273.919 voxel target complessivi dei crop. Nessuno dei 16
crop positivi raggiunge Dice >=0,90. Due crop di `inj_1419` (0 e 3)
producono zero voxel predetti positivi, pur contenendo rispettivamente
19.264 e 9.600 voxel target; i massimi score sono circa 0,108 e 0,123.
Il crop 3 di `rem_133` trova solo 1.710 dei 16.848 voxel annotati, senza
FP: Dice 0,18429. Le figure confermano target persi o predizioni troppo
piccole. Un crop reale di `inj_1713` conserva 8.100 FP (3,08990%).
La riduzione degli FP aggregati non equivale a localizzazione migliore:
in questo checkpoint si accompagna soprattutto a sensibilità minore.

**Cronologia disponibile.** I controlli fino a 175 aggiornamenti riportano
Dice e FP negativi nulli: non conservano mappe sufficienti per descrivere
tutte le predizioni intermedie. A 250 aggiornamenti: Dice 0,68887, FP
0,93126%; a 275: Dice 0,73410, FP 1,19321%; a 300: Dice 0,51008, FP
0,17503%. Nessun controllo salvato soddisfa entrambi i criteri del gate.
Il Dice finale peggiore, nonostante la loss media più bassa, conferma che
la sola loss non può decidere il superamento del controllo spaziale.
Non prova da solo un errore nel calcolo della loss o una causa specifica
delle oscillazioni.

**Limite dei checkpoint overfit di questa run.** Nel trainer usato `best.pt`
veniva riscritto a ogni epoca overfit, come `last.pt`; era un artefatto di debug,
non il miglior checkpoint selezionato con le metriche intermedie. Il CSV
non consente di ricostruire i pesi a 275 aggiornamenti. Anche quel controllo
non superava il gate, quindi recuperarlo non sbloccherebbe il protocollo.

**Decisione.** La singola replica non conferma che aumentare il LR decoder
risolva il controllo di apprendimento. Fermare le prove di LR e mantenere
bloccati piloti e full; nessuna promozione, aumento di budget o cambio di
soglia. Il prossimo intervento deve isolare il comportamento del training:
componenti BCE/Dice, gradienti e aggiornamenti effettivi, apprendimento di
crop fissi isolati prima del gruppo di sei scansioni. L'implementazione
successiva è descritta nella sezione 10.4; non è una causa diagnosticata.
Non è stata eseguita nuova inferenza GPU né training locale: l'analisi
si basa sui report e sulle immagini sincronizzati dal laboratorio.

### 10.4 Debug guidato dei quattro controlli

**Implementazione.** L'entrypoint
[`debug_segmentation`](../src/tesi_m3d/debug_segmentation.py) esegue il
controllo di gradienti/aggiornamenti e BatchNorm, la memorizzazione del primo
crop positivo del primo caso overfit, la coppia con il primo crop negativo
dello stesso caso e, soltanto dopo i due successi, la prova sulle sei
scansioni. Si usa il config originale a LR decoder 3e-4, soglia 0,5 e nessuna
augmentation. Ogni fase riparte da seed 21 e MedicalNet ufficiale; massimo
300 aggiornamenti per fase. I batch isolati sono 1 e 2 crop; non sono una
replica controllata del gruppo di sei scansioni né una metrica di validation.
Crop e target sono identificati nel report tramite indici, conteggi e hash.

Il codice della loss e dell'aggiornamento è condiviso con il trainer normale:
le componenti BCE/Dice non sono una seconda implementazione. Sono registrati
norme dei gradienti e variazioni dei pesi per encoder/decoder. I gate bloccano
le fasi dipendenti e non modificano i vecchi report né avviano piloti/full.
La sezione 18.2.1 della guida contiene un unico comando Windows e gli output
da sincronizzare, senza checkpoint pesanti.

**Correzione dei nuovi checkpoint overfit.** `best.pt` ora viene salvato al
controllo migliore: precedenza a gate passato, poi Dice maggiore e FP minori.
`best_gate.json` identifica metriche e passo. `last.pt` rimane l'ultimo stato;
`gate.json` è calcolato su quest'ultimo e non viene sostituito dal migliore.
Il resume overfit accetta solo `last.pt`, perché il best può essere a metà
epoca; impedisce inoltre aggiornamenti oltre il limite già raggiunto.
La selezione dei checkpoint pilota/full tramite AP densa resta invariata.
I checkpoint delle prove isolate sono marcati `segmentation_debug`, non
utilizzabili come segmentatori validati o come gate sulle sei scansioni.

**Verifica e limite.** Test sintetici controllano aggiornamenti effettivi,
componenti della loss, memorizzazione di positivo e coppia con un modello
minimo, reinizializzazione identica fra fasi, arresti prima delle fasi
dipendenti, integrità del gate originale e conservazione di un best
intermedio quando l'ultimo stato peggiora. Un aggiornamento CPU con pesi
MedicalNet ufficiali e crop 64 cubo ha inoltre verificato gradienti e
variazioni dei pesi non nulli per entrambi i gruppi, con BatchNorm immutate.
Questo non dimostra che MedicalNet memorizzi i crop CT: serve l'esecuzione
sul PC del laboratorio. Non è dichiarato risolto il problema di localizzazione.

### 10.5 Esito del debug: aggiornamenti e crop isolati superati, sei scansioni fallite

**Verifica eseguita.** Analizzati
[`report.json`](../outputs/segmentation_debug/report.json), selezione dei
crop, curve per aggiornamento e per caso, conteggi voxel e figure. Il
workflow si arresta a `six_scans`; i controlli precedenti passano.

| Controllo | Aggiornamenti | Dice binario positivo | FP voxel sui negativi | Esito |
| --- | ---: | ---: | ---: | --- |
| Positivo isolato | 50 | 0,91058 | non definito | superato |
| Positivo e negativo isolati | 50 | 0,90422 | 0,0026703% (7 voxel) | superato |
| Sei scansioni, ultimo stato | 300 | 0,71146 | 1,75724% | fallito |
| Sei scansioni, miglior controllo | 250 | 0,73293 | 0,82042% | fallito: Dice <0,90 |

Encoder e decoder hanno gradienti e variazioni dei pesi finiti e non nulli
in tutti i 300 aggiornamenti registrati sulle sei scansioni, oltre che nei
due test isolati. Le norme dei gradienti dell'encoder vanno da 0,000528 a
2,19867, quelle del decoder da 0,22425 a 2,88077. Il clipping globale a
norma 1 interviene in 29/300 aggiornamenti: non tronca sistematicamente
tutta la run. I due controlli iniziali confermano statistiche BatchNorm
immutate. Questo esclude un blocco completo dei due gruppi di parametri
nei passi osservati, non certifica ogni singolo parametro o l'intero dataset.
La loss registrata coincide con 0,5 BCE + 0,5 Dice entro circa 6e-8.

**Stesso crop, comportamento diverso.** Il positivo isolato è il crop 0
di `inj_2092`, con 13.468 voxel target; il negativo è il crop 4 della stessa
scansione, senza foreground. Le figure CT/GT nelle tre sezioni della mask
coincidono esattamente fra positivo, coppia e primo crop delle sei scansioni.
Nel gruppo di sei il medesimo positivo ha Dice 0,64155, con TP=13.452 e
FP=15.016: viene quasi tutto trovato ma la previsione si allarga. Non è
quindi un crop intrinsecamente impossibile da rappresentare con il modello.
Il risultato isolato non esclude errori di annotazione in altri casi e non
dimostra riconoscimento della manipolazione: una rete può memorizzare il
riquadro di un solo esempio.

**Dice binario e Soft Dice vanno distinti.** Nei test isolati la Dice loss
finale è ancora 0,70220 e 0,71306: il Soft Dice è rispettivamente 0,29780 e
0,28694, non 0,91 e 0,90. Non è una contraddizione aritmetica: il gate
soglia le probabilità a 0,5, mentre la loss usa tutte le probabilità, anche
quelle sotto soglia sull'ampio sfondo. Il criterio binario è superato,
ma la loss non dimostra convergenza o probabilità vicine a una mask 0/1.
Non sono disponibili mappe numeriche 3D complete per ricostruire qui la
distribuzione degli score; non attribuire una calibrazione specifica alle
sole immagini PNG.

**Errore sul gruppo completo.** All'ultimo stato i 16 crop positivi hanno
TP=249.009, FP=128.997, FN=24.910. Precision media per crop 61,21%, recall
medio 90,23% (micro: precision 65,87%, recall 90,91%). Nessun positivo su
16 raggiunge Dice >=0,90, quindi il gate non dipende da un unico outlier.
Il peggiore è ancora il crop 3 di `inj_2092`: target 1.300 voxel, TP=484,
FP=12.610 e Dice 0,06725. Nei 32 negativi si contano 147.408 FP;
14/32 superano individualmente l'1%. Le medie per caso sono:

| Scansione | Dice medio positivi | FP nei suoi crop negativi |
| --- | ---: | ---: |
| inj_2092 | 0,54717 | 0,63190% |
| inj_1419 | 0,78054 | 5,37214% |
| rem_133 | 0,71805 | 1,19238% |
| rem_15 | 0,80007 | 1,90649% |
| real inj_1713 | non definito | 0,56334% |
| real rem_605 | non definito | 1,91417% |

La cronologia mostra variazioni simultanee su più casi: a 150 aggiornamenti
Dice medio 0,45018 e FP 6,83966%; a 175 Dice 0,11750 e FP 0,05168%.
Non è una crescita stabile del controllo spaziale. Il best a 250 è ora
conservato secondo `best_gate.json`; le immagini finali sincronizzate
riguardano invece il passo 300. La scelta del best risulta coerente con la
cronologia, ma neppure il best supera il gate.

**Interpretazione e prossimo intervento motivato.** Il problema compare
nel passaggio alla composizione multi-crop/multi-scansione del training.
L'attuale aggiornamento usa otto crop di un solo volume: 200/300 passi
contengono quattro positivi e quattro negativi; 100/300 contengono solo
negativi reali e ricevono soltanto BCE. Nei passi finali dei manipolati
la Dice loss media è ancora circa 0,67195. Questa struttura, il peso
effettivo del foreground e l'interferenza tra aggiornamenti sono le prime
ipotesi da isolare. Le norme non misurano l'allineamento fra gradienti:
non dimostrano che i reali causino dimenticanza o che la BCE sia errata.

Il confronto successivo motivato è con batch composti da crop di più
scansioni, sugli stessi 48 crop fissi, stesse 50 esposizioni per crop e
300 aggiornamenti. Prima di implementarlo va mantenuta esplicita la
ponderazione della loss: con la normalizzazione attuale, portare foreground
in ogni batch aumenterebbe da solo il peso Dice complessivo (ora assente
nei passi reali). Chiamare questo confronto "solo shuffle" nasconderebbe
un cambiamento dell'obiettivo. Non si propone un altro LR, né nuove epoche,
soglie o un cambio di architettura sulla sola base di questi dati.

**Decisione e limite.** Piloti/full restano bloccati. Il debug ha ristretto
il problema e dimostrato rappresentabilità del crop selezionato, non
validità della localizzazione su nuove scansioni. Nessuna nuova esecuzione
GPU o modifica del training in questa analisi: i pesi non sono presenti
localmente, le conclusioni sono sui report e sulle figure sincronizzati.

### 10.6 Confronto appaiato della composizione dei batch: implementato

**Intervento.** L'entrypoint
[`compare_segmentation_batches`](../src/tesi_m3d/compare_segmentation_batches.py)
esegue grouped e mixed su un solo pool di 48 crop fissi della coorte
overfit originale. Ogni epoca visita tutti i crop una volta: 50 epoche e
300 aggiornamenti per variante, nessun arresto anticipato che alteri le
esposizioni. Entrambe ripartono da MedicalNet e seed 21, con hash completo
dei pesi/buffer iniziali confrontato a fine esecuzione. Non si riusa uno
dei checkpoint falliti e non si modifica il gate originale dei piloti.

**Composizione e loss.** Grouped ripete l'ordine originale dei volumi;
mixed include in ogni aggiornamento due/tre positivi e negativi di almeno
tre scansioni, inclusi entrambi i reali, senza duplicare o saltare crop.
La BCE rimane mediata su otto crop; il Dice usa sempre denominatore quattro
per conservare il peso di ogni crop positivo sull'epoca. Questo coincide
con la loss originale nei batch fake e dà zero Dice nei batch reali.
Usare la normale media sui positivi presenti nel batch misto ne aumenterebbe
il peso complessivo del 50%; il confronto evita questa confusione.
L'estensione del trainer è un parametro opzionale: gli altri entrypoint
mantengono la normalizzazione precedente.

**Tracciabilità e verifica.** Hash dei singoli input e target, piano degli
aggiornamenti, esposizioni per crop, componenti della loss, gradienti,
metriche per caso, best e last sono salvati separatamente per variante.
Test sintetici verificano determinismo e copertura del sampler, presenza
di scansioni diverse, identità dell'obiettivo e dei gradienti su un'epoca
a pesi fermi, equivalenza dell'accumulo 1×8/2×4, inizializzazioni identiche,
budget completato anche se il gate passa prima e protezione dei vecchi
output. Il comando e la lettura del risultato sono nella sezione 18.2.2
della guida Windows.

**Stato al momento dell'implementazione.** Verificata localmente; risultati
successivi sui CT e sulla GPU Windows nella sezione 10.7. Il cambio della
composizione può modificare la traiettoria di AdamW e del clipping anche a
obiettivo globale conservato: è l'ipotesi sperimentale, non una correzione
scientifica già dimostrata. Nessuna promozione a pilota/full automatica.

### 10.7 Risultato grouped/mixed: beneficio parziale, gate ancora fallito

**Evidenze sincronizzate.** Analizzati
[`comparison.json`](../outputs/segmentation_batch_comparison/comparison.json),
i CSV degli aggiornamenti, delle esposizioni, dei crop e dei casi, le curve
ai controlli ogni 25 passi e sei figure finali di entrambe le varianti
complessivamente. Manifest `0de4718f…`, pool `80c77851…`, stato iniziale
`9a053aa1…`: gli hash riportati coincidono fra le due run; l'hash del ledger
del pool è stato anche ricalcolato. Config identici, 48 crop (16 positivi,
32 negativi), target voxel coincidenti nei due `crops.csv`, 300 passi e
50 visite a ogni crop in ciascuna run. Ogni epoca copre esattamente i 48
indici una volta. Nei mixed ogni passo ha otto crop, due/tre positivi,
almeno due negativi reali e almeno tre scansioni. Gradienti e variazioni
dei pesi di encoder/decoder sono finiti e non nulli in tutti i 600 passi;
la loss registrata coincide con metà BCE più metà componente Dice.

| Misura sui crop fissi, soglia 0,5 | Grouped | Mixed |
|---|---:|---:|
| Dice medio dei positivi, passo 300 | 0,714908 | 0,739454 |
| FP sui voxel dei crop negativi, passo 300 | 1,23613% | 0,69098% |
| Dice medio del best registrato, passo 250 | 0,748079 | 0,760742 |
| FP negativi del best | 0,70795% | 0,62721% |
| Precision media sui 16 positivi, passo 300 | 59,77% | 61,73% |
| Recall media sui 16 positivi, passo 300 | 93,54% | 96,06% |
| Positivi con Dice almeno 0,90, passo 300 | 0/16 | 0/16 |
| Crop negativi con FP oltre 1%, passo 300 | 9/32 | 7/32 |
| Un gate superato ai controlli registrati | No | No |

Mixed guadagna 0,02455 di Dice finale e riduce del 44,1% relativo la quota
FP sui negativi. Il vantaggio di Dice fra i best scende a 0,01266; entrambi
sono al passo 250 e peggiorano dopo. La curva mixed anticipa l'emersione
del foreground (Dice 0,461 al passo 100 contro zero grouped), ma resta
oscillante. Il vantaggio finale interessa 10 positivi su 16; sei peggiorano.
Il caso `rem_133`, per esempio, migliora appena nel Dice medio ma peggiora
nei suoi crop negativi da 0,13762% a 0,86279% FP. Non è un beneficio uniforme.
Sui soli negativi dei due reali, FP aggregati 0,73044% → 0,43316%; sui
negativi delle scansioni fake, 1,74181% → 0,94881%. Queste sono misure su
crop di training, non falsi positivi dell'inferenza su scansioni complete.

**Errore residuo.** Sui positivi mixed: TP 261.865, FP 128.330, FN 12.054.
Il foreground viene largamente recuperato, ma la precisione rimane bassa.
`inj_2092` crop 3 (1.300 voxel target) passa da Dice 0,17286 a 0,33723:
mixed predice 4.773 voxel, con TP 1.024, FP 3.749 e FN 276.
`rem_133` crop 2 (3.042 voxel target) passa da 0,50694 a 0,60988 e
mantiene 3.869 FP contro 3.032 TP. Le figure dei due crop mostrano regioni
lisce più estese dei target cuboidali e troncate al bordo del crop; anche
nei peggiori negativi `inj_1419` e reale `rem_605` si vedono attivazioni ai
bordi. È un indizio da misurare, non prova di un errore del padding o del
sampler. Il primo crop di `inj_1419` è migliore (Dice 0,88902), ma ancora
con alone FP. I valori di Soft Dice ricostruiti dai passi finali restano
molto inferiori al Dice binario: circa 0,357 grouped e 0,370 mixed sui
positivi degli ultimi quattro giri, misurati durante il training a pesi
variabili, non rivalutando un singolo checkpoint.

**Decisione.** Gate fallito anche per il best: Dice richiesto almeno 0,90,
FP negativi sotto 1%. Il batching misto è un candidato per le prossime
diagnosi, non una soluzione sufficiente né una modifica da promuovere ora
al full. Nessuna nuova ricerca di LR, estensione delle epoche o modifica
della soglia per far passare il test. Il prossimo controllo motivato è
verificare la rappresentabilità dei due crop peggiori (`inj_2092/3` e
`rem_133/2`) isolatamente e con un negativo fisso dello stesso caso,
conservando input/target verificati tramite hash, inizializzazione,
preprocessing, LR e limite di 300 aggiornamenti per prova. Finora il debug
isolato riguardava solo `inj_2092/0`, molto più grande: il suo successo
non dimostra apprendibilità dei piccoli target parziali al bordo.
Nel controllo andrà quantificata anche la distribuzione di FP/FN rispetto
ai bordi, senza spostare o correggere automaticamente le annotazioni.
Un successo isolato restringerebbe il problema alla gestione congiunta di
questi esempi; un fallimento richiederebbe verifiche di geometria e
rappresentazione prima di un altro confronto del batching. Nessuno dei due
esiti autorizza da solo il passaggio a pilota/full.

**Limiti.** Una coppia di run e un solo seed non dimostrano robustezza né
generalizzazione. Tempi 54,06/51,13 s e picchi allocati 4.646/1.698 MiB
sono riportati, ma non attribuiti al sampler: grouped è sempre eseguito
prima, nello stesso processo, e inizializzazione/workspace possono avere
costi diversi. Il tempo esclude la preparazione comune dei CT e include
inizializzazione, valutazioni e checkpoint. I pesi `.pt` non sono presenti
localmente: verificati report, coerenza interna dei CSV e figure finali,
senza rieseguire inferenza o controllare le immagini del best al passo 250.

### 10.8 Controllo dei due crop peggiori: implementazione

**Intervento.** `debug_segmentation_hard_crops` rigenera esattamente
`inj_2092/3` e `rem_133/2` negli indici originali della coorte, confronta
immagini e mask con il ledger del confronto grouped/mixed e prova ciascuno
da solo e con il primo negativo fisso (crop 4) dello stesso caso.
Quattro inizializzazioni fresche, ciascuna confrontata con l'hash iniziale
del riferimento; nessun checkpoint allenato viene usato come partenza.
Si riusa `run_fixed_stage`, mantenendone l'arresto anticipato per il vecchio
debug; solo il nuovo comando completa l'intero budget, massimo 300 passi
per prova, conservando primo superamento osservato, best e last.
I fallimenti scientifici non impediscono le altre prove; quelli tecnici
interrompono e vengono registrati. I checkpoint restano di tipo diagnostico
e il gate del protocollo non viene modificato.

**Misure.** Le figure iniziali/finali sono accompagnate da TP/FP/FN,
precision, recall e tasso FP sul background per distanza dal bordo
(0–3, 4–7, 8–15, almeno 16 voxel), più l'intero crop. Le fasce sono
disgiunte e ne vengono verificati copertura e conteggi su fixture nota;
si registra anche quali facce tocca il target. Il conteggio distingue
geometria dei target da estensione delle predizioni, senza correggere o
traslare annotazioni. Un errore concentrato ai bordi, da solo, non dimostra
un difetto del padding.

**Verifica locale.** Test su TIFF sintetici verificano identità degli input,
indici originali (compreso `rem_133` al posto 2), negativi vuoti, quattro
prove complete, preservazione del gate e rifiuto di config, manifest,
ledger, CT, mask o stati iniziali differenti. Un controllo con i pesi
ufficiali MedicalNet e due crop sintetici 64³ ha completato un aggiornamento
CPU con gradienti e variazioni dei pesi positivi in encoder/decoder,
statistiche BatchNorm invariate e tabelle spaziali salvate. Non è un test
di qualità scientifica o di memoria CUDA.
La verifica pertinente comprende 57 test superati (segmentazione, debug,
confronto dei batch e nuovi crop); escluso il test dei worker persistenti,
non coinvolti da questo comando che accede direttamente ai due volumi.

**Limiti e decisione.** L'ambiente locale (PyTorch 2.15.0+cu132, MONAI
1.5.2, CPU) ricrea lo stato `46820f3d…`, diverso da `9a053aa1…` del confronto
Windows: verificata quindi anche la necessità di non assumere identità
dell'inizializzazione dal solo seed. Eseguire nell'ambiente originale del
laboratorio; un hash diverso blocca il confronto prima del training.
Il protocollo isolato usa fino a 300 esposizioni per crop contro le 50
delle sei scansioni e la normale loss sui soli uno/due crop selezionati:
un eventuale successo non isola causalmente l'effetto del batching e non
prova generalizzazione. Gli esiti successivi sui due target sono nella
sezione 10.9. Piloti/full restano bloccati. Comando, file da sincronizzare
e criteri di lettura nella sezione 18.2.3 della guida Windows.

### 10.9 Due crop peggiori: risoluzione sufficiente in un caso, fondo probabilistico residuo

**Verifica eseguita.** Analizzati
[`segmentation_hard_crops/report.json`](../outputs/segmentation_hard_crops/report.json),
selezione, tutte le curve e i 1.200 aggiornamenti, tabelle spaziali e sei
figure finali (quattro positivi e due negativi). Esecuzione completata su
CUDA, PyTorch 2.13.0+cu130 e MONAI 1.5.2, senza errori tecnici. Manifest e
config coincidono con il riferimento grouped/mixed; i quattro SHA256 di
immagini/target corrispondono alle righe originali 3, 4, 18, 20 del pool.
Tutti gli stati iniziali riportano `9a053aa1…`. Ogni prova completa 300
passi e 300 visite per crop; controlli ai passi 0, 1 e ogni 25. Verificati
best/last contro i JSON dedicati, identità della loss e concordanza delle
metriche finali con TP/FP/FN dei CSV. I conteggi delle fasce sommano al
volume intero. Gradienti e aggiornamenti encoder/decoder finiti e non nulli
in tutti i 1.200 passi; preflight BatchNorm invariato nelle quattro prove.

| Caso/prova | Dice finale | Miglior Dice registrato | Primo gate osservato | FP sul negativo finale |
|---|---:|---:|---:|---:|
| `inj_2092/3` solo | 0,892596 | 0,892596 (300) | Mai | Non applicabile |
| `inj_2092/3` + negativo | 0,819848 | 0,899654 (250) | Mai | 0% |
| `rem_133/2` solo | 0,982082 | 0,982082 (300) | 150 | Non applicabile |
| `rem_133/2` + negativo | 0,968636 | 0,968636 (300) | 200 | 0% |

Il gate globale rimane falso: due prove injection non passano. Il best
injection nella coppia dista appena 0,000346 dal criterio 0,90; è un
quasi-superamento, non prova di incapacità del modello. Non arrotondarlo
a un successo. Il last peggiora dopo il passo 250, mentre la loss continua
a diminuire. Anche la removal singola oscilla (0,958 al passo 225,
0,865 al 250, 0,982 al 300): loss e Dice binario non sono intercambiabili.
Nei controlli al passo 50 injection ha ancora Dice zero in entrambe le
prove, removal circa 0,763/0,788. I successi removal richiedono più delle
50 esposizioni usate nel confronto delle sei scansioni.

**Dimensioni, bordi e sovrastima.** Injection ha target 13×10×10 voxel,
1.300/262.144 voxel (0,496% del crop), bbox locale z/y/x
`[24,54,0]–[36,63,9]`, tocca y alto e x basso. Removal ha target
13×9×26, 3.042 voxel (1,160%), bbox `[8,0,0]–[20,8,25]`, tocca y basso e
x basso. Entrambi sono parziali rispetto alla mask completa del volume.
Quindi anche il caso riuscito tocca due facce e ha una dimensione di soli
nove voxel: questi dati non sostengono un impedimento assoluto dovuto
alla risoluzione del decoder, né ai bordi in sé. Non isolano però gli
effetti di dimensione, contrasto, contenuto e tipo di manipolazione.

Al termine, injection sola: TP 1.284, FP 293, FN 16, precision 81,42%,
recall 98,77%; nella coppia TP 1.297, FP 567, FN 3, precision 69,58%,
recall 99,77%. Persiste un alone FP attorno al target, più esteso nel last
della coppia. Removal sola: TP 3.042, FP 111, FN zero; nella coppia
TP 3.042, FP 197, FN zero, con contorno quasi coincidente al cuboide.
Nei due negativi selezionati non resta alcun voxel sopra soglia 0,5.
Non equivale ad assenza di FP sui negativi non visti o sui volumi completi.

Gli FP injection sola sono 166 nella fascia 0–3, 114 in 4–7 e 13 in
8–15 voxel dal bordo; nella coppia 310, 214 e 43. L'interno ad almeno 16
voxel non ha FP, ma non contiene neppure target. Removal concentra a sua
volta gli errori vicino ai target ai bordi. Le fasce confondono vicinanza
al bordo e vicinanza alla lesione in questi due esempi: non dimostrano un
artefatto globale di padding. Il clipping globale è attivo in soli
6/8/7/5 aggiornamenti su 300 (ordine delle quattro righe della tabella),
non un blocco persistente dei gradienti.

**Probabilità e loss: indizio più concreto della sola risoluzione.** I valori
`soft_dice_loss` sono loss: il Soft Dice finale è **1 meno** quel valore.
Injection sola/coppia: 0,06744/0,07149; removal: 0,15426/0,15441,
nonostante il Dice binario quasi perfetto in quest'ultima. La formula
locale confronta le probabilità sigmoid senza soglia con la mask; il
denominatore somma anche le probabilità dei voxel di background.
Non è lo stesso conteggio TP/FP/FN usato dal gate.

Controllo aggiuntivo sulle PNG: letti i canali rossi dei pixel delle
heatmap, salvati come `floor(255*p)`, nei tre piani centrati sulla mask.
La mediana del background rappresentato è circa 0,125–0,129 in tutte le
prove; quella del foreground circa 0,992–0,996. Nei primi piani dei due
negativi la mediana è circa 0,129, pur con zero voxel sopra soglia.
Sono statistiche di sezioni selezionate, quantizzate a 8 bit, non quantili
delle mappe 3D complete. Confermano però nelle sezioni un fondo residuo
diffuso sotto soglia: per target inferiori al 1,2% del crop la sua massa
può dominare il denominatore del Soft Dice. Non attribuire il divario a
un bug della loss o a una causa precisa del training senza le mappe complete.

**Decisione e prossimo controllo.** Esclusa, nel solo caso removal verificato,
l'incapacità assoluta di rappresentare con questa architettura un piccolo
cuboide al bordo. Injection è quasi apprendibile al criterio richiesto,
con errore residuo soprattutto di sovrastima. Non cambiare ora decoder,
crop, LR o soglia sulla sola base del gate falso; non avviare piloti/full.
Il prossimo controllo motivato è **senza training**, sui best/last già
conservati in laboratorio: statistiche esatte delle probabilità complete
su foreground/background, massa probabilistica dentro/fuori mask e
Soft Dice ricalcolato, inclusi i due negativi, per distinguere fondo
diffuso da errore di contorno e dal peggioramento binario dopo il best.
Questo controllo deve precedere un eventuale confronto mirato della loss;
non costituisce autorizzazione a una ricerca di iperparametri o ad abbassare
il gate. I precedenti 300 passi sulle sei scansioni non sono appaiati per
esposizioni né pesi dei singoli crop a queste prove isolate.

**Limite.** Nessun `.pt` o array di probabilità completo è sincronizzato:
non rieseguite inferenza o misure volumetriche delle probabilità; le figure
descrivono il last, non il best della coppia injection. Nessuna conclusione
di generalizzazione dalle prove di memorizzazione. Aggiornata la guida con
l'esito; nessuna modifica a codice, loss, soglie, gate o output scientifici.

### 10.10 Probabilità complete best/last: valutatore implementato

**Intervento.** L'entrypoint `diagnose_segmentation_probabilities` valuta i
best/last delle quattro prove difficili, senza training. Riusa il controllo
di identità dei crop e le fasce spaziali del debug; controlla anche il
legame fra ciascun checkpoint e batch/stadio/config/manifest/metriche.
Caricamento completo strict degli stati encoder e decoder, valutazione
FP32 senza autocast e gradienti. Non serve il checkpoint MedicalNet
originale, perché i pesi sono già inclusi nei `.pt` allenati; nessuna
inizializzazione casuale può sostituire uno stato incompleto. Verifica gli
otto file prima di rileggere i CT e protegge run/riferimento da scritture.

**Misure prodotte.** Quantili esatti su tutti i voxel di ogni crop,
foreground/background/totale e fasce rispetto al bordo, somme delle
probabilità in float64, medie e frazioni sopra livelli descrittivi fissi.
Soft Dice ricalcolato dalle masse per i positivi e conteggi binari a 0,5;
loss FP32 originale e scostamento dalle metriche salvate. Le distribuzioni
distinguono un fondo diffuso da probabilità elevate concentrate in un
alone; i compartimenti vuoti restano indefiniti. Mappe complete float32 e
mask uint8 in `.npz`, figure di ogni best/last e hash dei checkpoint
consentono una successiva analisi locale senza trasferire tutti i pesi.

**Verifica.** Fixture con background 0,125 e un solo voxel positivo
mostra Dice binario 1 e Soft Dice inferiore a 0,001: verificati formula,
masse e quantili, senza confondere le due metriche. Test su TIFF sintetici
coprono gli otto ripristini offline, preservazione byte per byte degli
input, compartimenti vuoti, identificativi/hash errati, stati incompleti,
logits non finiti, assenza dei `.pt`, output esistente e differenze dalle
metriche salvate. Il comando riporta il completamento, non inventa un
nuovo gate o sostituisce quello fallito del protocollo.
Superati 75 test pertinenti (escluso il worker persistente, non usato dal
comando). Anche un checkpoint sintetico
con encoder MedicalNet ufficiale e crop 64³ è stato ripristinato offline
con metriche riprodotte: BCE/Soft Dice loss coincidenti, scarto del Dice
binario circa `3,3e-10`. È una verifica del percorso di misura, non un
risultato sul dataset del laboratorio.

**Stato al momento dell'implementazione.** Gli otto `.pt` allenati non
sono sincronizzati; il comando è destinato al PC del laboratorio. Gli
audit successivi sono analizzati nella sezione 10.11. Non basta il dato quantizzato delle PNG per
assegnare una causa alla loss. Nessuna variazione a training, loss,
normalizzazione, soglie o selezione dei checkpoint; piloti/full restano
bloccati in attesa delle misure e della loro interpretazione.

### 10.11 Audit delle probabilità: fondo diffuso confermato, loss e contorno divergono

**Evidenze e verifiche.** Ricevuti report, distribuzioni, tabelle spaziali
e PNG di tutti gli otto checkpoint. `complete: true`, nessun errore,
`all_reproduced_within_1e_4: true`; massimo scarto dalle metriche salvate
`5,96e-8`. Config, manifest e selezione corrispondono alla run hard-crops.
Verificata coerenza dei 12 crop valutati: TP/FP/FN, masse dentro/fuori,
conteggi delle soglie descrittive, medie e quantili ordinati; le fasce
sommano al totale e foreground più background ricompongono il crop.
Il Soft Dice ricalcolato dalle masse coincide con quello derivato dalla
loss FP32 entro `1e-7`. Ispezionate le figure positive best/last della
coppia injection e il best della coppia removal.

| Crop positivo / checkpoint | Dice binario | Soft Dice | Media dentro mask | Media fuori mask | Massa predetta fuori mask |
|---|---:|---:|---:|---:|---:|
| Injection sola, best=last (300) | 0,89260 | 0,06744 | 0,96186 | 0,13238 | 96,51% |
| Injection + negativo, best (250) | 0,89965 | 0,06264 | 0,97670 | 0,14556 | 96,76% |
| Injection + negativo, last (300) | 0,81985 | 0,07149 | 0,98375 | 0,12728 | 96,29% |
| Removal sola, best=last (300) | 0,98208 | 0,15426 | 0,98787 | 0,12703 | 91,63% |
| Removal + negativo, best=last (300) | 0,96864 | 0,15441 | 0,98985 | 0,12717 | 91,63% |

`best=last` qui indica stesso aggiornamento e metriche/distribuzioni
coincidenti, non un confronto diretto dei tensori dei pesi. La quota di
massa esterna non è una quota di voxel falsamente positivi: somma anche
probabilità inferiori a 0,5 su centinaia di migliaia di voxel negativi.
Nei crop negativi, estratti dagli stessi volumi manipolati e con mask
vuota, il fondo medio è 0,12763–0,14631 per injection
e 0,12943 per removal; tutti restano senza voxel sopra 0,5. Non sono nuove
scansioni reali né un test di generalizzazione.

**Fondo diffuso, non solo alone.** Nei positivi almeno il 99,68% del
background ha probabilità almeno 0,1 (circa 99,997% nell'injection sola).
Nella coppia injection il p99 del fondo scende da 0,15398 a 0,13404,
mentre il massimo sale da 0,89460 a 0,98189: il grosso del fondo migliora,
ma una piccola coda diventa più confidente. Anche la fascia interna,
senza target e senza FP binari, ha media 0,14472 nel best e 0,12821 nel
last. Il residuo non è limitato ai bordi del crop o al contorno annotato.

La massa fuori mask dell'injection best è 37.967,21 contro 1.269,71
dentro; nel last 33.199,37 contro 1.278,87. Dai soli conteggi si può anche
porre un limite: ciascun FP binario contribuisce al massimo uno alla
massa, quindi almeno il 99,23% della massa esterna nel best e il 98,29%
nel last proviene da voxel sotto 0,5. Non è una misura esatta di tale
quota, ma un limite inferiore indipendente dalle mappe mancanti.
È sufficiente a spiegare il piccolo Soft Dice, senza ipotizzare un suo
calcolo errato. Non dimostra da solo la causa degli errori di contorno.

**Confronto temporale decisivo: injection nella coppia.**

| Misura | Best, passo 250 | Last, passo 300 |
|---|---:|---:|
| BCE | 0,15767 | 0,13714 |
| Soft Dice loss | 0,93736 | 0,92851 |
| Loss combinata 0,5/0,5 | 0,54751 | 0,53283 |
| TP / FP / FN | 1.300 / 290 / 0 | 1.297 / 567 / 3 |
| FP con probabilità almeno 0,9 | 0 | 187 |
| Dice binario | 0,89965 | 0,81985 |

La massa di fondo diminuisce del 12,56% e la loss combinata del 2,68%,
mentre gli FP aumentano del 95,52%. Le figure mostrano un alone esterno
più ampio. Non è semplice incertezza dei voxel debolmente positivi:
187 FP del last hanno probabilità almeno 0,9. Abbassare il fondo diffuso
non garantisce quindi di restringere il contorno. È documentato un
disaccordo locale fra miglioramento dell'obiettivo continuo e qualità
binaria a 0,5; non è dimostrato che la sola loss spieghi l'insuccesso
sulle sei scansioni o che vi sia un errore nell'ottimizzatore.

**Confronto motivato (implementazione nella sezione 10.12).** Conservare
MedicalNet, decoder, preprocessing, seed, crop, sampling, LR, BCE e budget;
cambiare soltanto il termine Dice, confrontando la forma attuale con
quella a denominatore quadratico:

`D_quad = (2*sum(p*y)+eps) / (sum(p*p)+sum(y*y)+eps)`.

Per target binari `y*y=y`. Una probabilità di fondo 0,13 contribuisce
0,0169 invece di 0,13; una probabilità alta 0,98 contribuisce ancora
0,9604. Questo modifica il peso relativo del fondo diffuso rispetto alle
predizioni molto confidenti, non prova che il contorno migliorerà.
La variante è disponibile come `squared_pred=True` nella
[DiceLoss di MONAI 1.5.2](https://monai.readthedocs.io/en/1.5.2/losses.html#diceloss),
verificata anche nell'installazione locale. Resta una sola ipotesi da
confrontare con baseline appaiata, non una ricerca di loss/LR. Si conserva
la supervisione BCE dei crop vuoti e il Dice sui soli foreground.
Confrontare Dice binario, precision/recall e FP con il valutatore comune;
non confrontare i valori numerici delle due loss come prova di successo,
perché la definizione cambia. Un eventuale successo isolato richiede
ancora il gate delle sei scansioni prima dei piloti. Nessun training
avviato in questa analisi; piloti/full restano bloccati.

**Sincronizzazione e limiti.** Nessun `.npz` è arrivato localmente: la
regola globale `*.npz` nel `.gitignore` escludeva anche queste mappe.
I controlli sopra riguardano gli aggregati e le figure, non un nuovo
calcolo sui voxel grezzi o una nuova inferenza. Aggiunta un'eccezione
limitata a `outputs/segmentation_probability_diagnostics/**/probabilities.npz`
per consentire il trasferimento richiesto. La modifica è limitata a
questo percorso. Non occorre ripetere l'audit per sincronizzare i
file già prodotti nel laboratorio. Le conclusioni restano limitate a
questi crop di training, senza rivendicare generalizzazione.

### 10.12 Confronto appaiato Dice lineare/quadratico: implementazione

**Intervento.** `compare_segmentation_losses` esegue una nuova baseline
lineare e una variante con solo denominatore Dice quadratico, sulle
quattro prove fisse di `inj_2092/3` e `rem_133/2` (solo e coppia con
negativo crop 4). Otto training indipendenti, massimo 300 aggiornamenti
ciascuno, stesse esposizioni fra i rami e nessuno stop per semplice gate
superato/fallito. Si riusano preparazione e controllo SHA256 dei crop e
`run_fixed_stage`: ogni prova riparte da MedicalNet e deve riprodurre
l'hash iniziale del riferimento. Gli input sono preparati una sola volta
e conservati in RAM. Nessuna modifica a sampler, geometria, preprocessing,
LR, BCE, optimizer, augmentation, seed o budget.

**Loss ed evidenze confrontabili.** Il parametro opzionale `squared_pred`
è inoltrato solo dal nuovo confronto attraverso i componenti condivisi;
il default rimane falso. Dice su foreground, BCE su tutti i crop, pesi
0,5/0,5, smoothing `1e-5` e accumulo dei gradienti invariati. La formula
selezionata viene salvata in `training_loss` nei checkpoint/report e
marcata nei CSV di aggiornamento. Monitoraggio e selezione del best usano
sempre le stesse metriche binarie e il Soft Dice lineare comune, indipendenti
dalla loss scelta. Nessun confronto numerico delle loss diverse è usato
come criterio di successo. Le differenze squared meno linear riportano
Dice, precision/recall, FP/FN e probabilità media del fondo, oltre ai FP
sui negativi.

**Audit automatico.** I 16 checkpoint best/last vengono ripristinati e
valutati automaticamente al termine delle rispettive prove con il
valutatore delle probabilità complete. Controllata anche l'identità della
loss nei nuovi checkpoint; quelli precedenti senza questo metadato restano
leggibili. Quantili, masse, figure e array float32 sono già nell'output
del confronto. Eccezione Git limitata ai suoi `probabilities.npz`, senza
includere i pesi. Nessun vecchio output, checkpoint o gate viene sovrascritto.

**Verifica locale.** Superati 90 test pertinenti (escluso il worker
persistente, non usato da queste prove con accesso diretto ai volumi).
Formula confrontata con `DiceLoss` MONAI, default lineare invariato,
crop vuoti supervisionati solo da BCE e gradienti equivalenti per
accumulo 1×8/2×4, anche con normalizzatore foreground fisso. Fixture TIFF
verificano otto inizializzazioni identiche, stessi hash dei batch,
completamento dei budget, 16 esportazioni best/last, metadati della loss,
leggibilità dei vecchi checkpoint e protezione dei risultati precedenti;
gli errori tecnici lasciano un confronto esplicitamente incompleto.
Un aggiornamento CPU con encoder MedicalNet ufficiale, due crop sintetici
64³ e Dice quadratico ha prodotto gradienti/aggiornamenti positivi in
encoder e decoder, BatchNorm invariata e metriche comuni riprodotte dal
checkpoint (scarto Dice circa `2,6e-11`, BCE e loss Dice coincidenti).
Non è una prova di qualità della localizzazione o di memoria CUDA.

**Decisione e limiti.** È un esperimento diagnostico su crop di training,
non il gate originale sulle sei scansioni né una nuova modalità promossa
al training completo. Un esito positivo richiederà ancora quel controllo.
I risultati CT/GPU, inizialmente da acquisire nell'ambiente del laboratorio,
sono ora analizzati nella sezione 10.13;
il diverso stato iniziale ricreato nell'ambiente locale resta motivo di
blocco prima del training, senza fallback. Comando e interpretazione
nella sezione 18.2.5 della guida Windows. Piloti/full restano bloccati.

### 10.13 Confronto Dice lineare/quadratico: risultati del laboratorio

**Evidenze sincronizzate, 23 novembre 2026.**
`outputs/segmentation_loss_comparison/comparison.json` riporta confronto
completo, quattro prove finali quadratiche superate e riproduzione delle
16 valutazioni best/last entro `1e-4`. Ambiente dichiarato: PyTorch
`2.13.0+cu130`, MONAI `1.5.2`, CUDA. Sono presenti anche tutte le 16 mappe
`probabilities.npz`; i pesi `.pt` non sono sincronizzati.

**Controlli indipendenti.** Config effettivo e manifest coincidono con
il riferimento `segmentation_batch_comparison`; identici hash iniziale
`9a053aa1…` e pool `80c77851…`. Ogni prova registra 300 aggiornamenti
consecutivi e 300 esposizioni per crop, per 2.400 aggiornamenti totali.
Tutti i gradienti/aggiornamenti di encoder e decoder nei CSV sono finiti
e positivi; i preflight confermano BatchNorm invariata. L'unica variante
dichiarata della loss è `squared_pred`; BCE, pesi e composizione restano
invariati. La composizione della loss nei CSV coincide entro `3e-8`.

Ricalcolati direttamente dagli array float32, forma `B×1×64×64×64`,
conteggi TP/FP/FN alla soglia 0,5, Dice, masse e media del fondo coincidono
con il report entro `1e-9`. Probabilità finite in `[0,1]`, mask binarie,
hash delle mask coincidenti con la selezione originale e mask identiche
fra rami/checkpoint. Lo scarto massimo dichiarato fra metriche ripristinate
e salvate è `2,78e-8`. Questo verifica gli output grezzi; senza pesi non
è stata eseguita una nuova inferenza locale.

| Prova | Dice lineare best / last | Dice quadratico best / last | FP / FN lineare last | FP / FN quadratico last |
|---|---:|---:|---:|---:|
| `inj_2092/3` solo | 0,904081 / 0,895934 | 1,000000 / 1,000000 | 302 / 0 | 0 / 0 |
| `inj_2092/3` + negativo 4 | 0,925267 / 0,925267 | 1,000000 / 1,000000 | 210 / 0 | 0 / 0 |
| `rem_133/2` solo | 0,995745 / 0,995745 | 1,000000 / 1,000000 | 26 / 0 | 0 / 0 |
| `rem_133/2` + negativo 4 | 0,968875 / 0,965408 | 1,000000 / 1,000000 | 218 / 0 | 0 / 0 |

Il quadratico recupera esattamente i 1.300 voxel injection e i 3.042
voxel removal in tutte le prove finali: precision e recall pari a 1.
I crop negativi best/last hanno zero voxel predetti in **entrambi** i
rami: qui il guadagno riguarda l'estensione delle predizioni nei crop
positivi, senza un peggioramento misurato sui due negativi fissi.
Le figure nei tre piani sono coerenti: spariscono i voxel esterni alla
mask presenti nella nuova baseline injection. I target restano le
annotazioni cuboidali disponibili, non contorni precisi del tumore.

**Andamento.** Primo superamento osservato ogni 25 passi, quadratico
contro lineare: injection solo 225 contro 275, injection in coppia 200
contro 300, removal solo 100 contro 150, removal in coppia 100 contro
225. Il removal quadratico solo ridiscende a Dice 0,8933 al passo 125
e supera nuovamente il criterio da 150; non si rivendica convergenza
monotona. Il lineare injection solo supera a 275 e fallisce a 300.
Tutti i best quadratici coincidono con il last a 300.

**Probabilità residue.** La media fuori mask cala da 0,13188 a 0,11247
(injection solo), da 0,12400 a 0,10627 (injection in coppia), da 0,12642
a 0,11630 (removal solo) e da 0,12658 a 0,10926 (removal in coppia).
Il fondo rimane diffuso: circa il 98,96–99,51% dei voxel fuori mask dei
crop positivi quadratici ha probabilità almeno 0,1. Il Soft Dice lineare
comune rimane fra 0,08108 e 0,17557, pur con Dice binario 1 a soglia 0,5.
Le medie sui due negativi quadratici sono 0,10843 e 0,11264. Quindi non
sono dimostrate né calibrazione delle probabilità né eliminazione della
massa di fondo. Il massimo fondo injection in coppia è 0,49599, vicino
alla soglia: zero errori a 0,5 non implica robustezza a qualsiasi soglia.
Non confrontare le due loss di training come se fossero la stessa metrica.

**Interpretazione e decisione.** Il denominatore quadratico è una
modifica promettente per apprendere questi target piccoli e parziali ai
bordi, senza cambiare risoluzione, decoder o dimensioni della mask.
Il successo dimostra che questi due crop sono memorizzabili con il
modello attuale; non identifica una causa unica di tutti i fallimenti.
La nuova baseline lineare injection in coppia raggiunge 0,925267, meglio
del precedente best 0,899654: esiste variabilità fra esecuzioni anche
con gli stessi hash iniziali e input, di origine non isolata. Il confronto
valido è quello appaiato appena concluso, non la sottrazione fra run storici.

Il prossimo esperimento giustificato è il **gate originale su sei scansioni
e 48 crop fissi**, da inizializzazione MedicalNet fresca, con il solo Dice
quadratico e massimo 300 aggiornamenti. Mantenere manifest, preprocessing,
campionamento, seed, LR, batch e criteri originali: Dice medio positivo
almeno 0,90, FP sui negativi sotto 1%, gradienti finiti. Conservare best,
last e dettagli per crop; non ripartire dai pesi già memorizzati qui.
Al momento dell'analisi il normale `train_segmentation.run_training`
non inoltrava ancora `squared_pred`; l'integrazione successiva è descritta
nella sezione 10.14. Nessun vecchio gate è stato modificato;
piloti/full ancora bloccati. Due crop positivi, due negativi e
un seed non misurano la localizzazione su volumi interi o la generalizzazione.

### 10.14 Gate originale con Dice quadratico: integrazione

**Intervento, 23 novembre 2026.** Il trainer standard valida e inoltra
`training.squared_pred` alla loss condivisa, con default falso per
config/checkpoint precedenti. Aggiunto
`configs/segmentation_overfit_squared.yaml`: solo denominatore Dice e
percorsi output/gate diversi dall'overfit originale, stesso manifest,
48 crop fissi, seed 21, MedicalNet obbligatorio, preprocessing, LR,
campionamento e massimo 300 aggiornamenti complessivi. Nessun resume
dai pesi memorizzati nelle prove isolate; i criteri del gate e lo stop
al primo controllo superato rimangono invariati.

Checkpoint, gate finale e best dichiarano i pesi della loss, la variante
e il trattamento dei crop vuoti. Esposta in CLI l'opzione diagnostica
già presente nel trainer `--debug-updates`, per conservare componenti,
norme dei gradienti/aggiornamenti e storia per caso. CSV e immagini finali
per crop rimangono quelli del last; best conservato separatamente.
Il resume rifiuta il cambio di loss, ma considera equivalenti il default
assente storico e il valore esplicito falso. I prerequisiti pilota/full
richiedono lo stesso denominatore del gate superato: un gate storico
senza metadato vale come lineare. Le configurazioni pilota/full non
sono state modificate e non vengono avviate.

**Verifica.** Test dei due rami nel percorso sulle sei scansioni verificano
inoltro della variante, aggiornamenti reali, salvataggio di best intermedio
e last, metadati in report/checkpoint, componenti nei CSV e rifiuto del cambio di loss nel
resume. Verificati CLI e relativo exit code, differenze limitate del nuovo
config, tipo booleano obbligatorio e blocco di gate con loss diversa.
La suite con fixture TIFF copre ancora training, valutazione e ripristino
dei checkpoint precedenti; la formula e l'accumulo quadratico restano
coperti dai test del confronto con MONAI. Superati 95 test pertinenti;
escluso il test dei worker persistenti, non usati dal nuovo config con
`num_workers: 0`. Restano 18 warning MONAI già presenti. Comando Windows nella sezione
18.2.6 della guida. Nessun training CT/GPU eseguito localmente; gli output
successivamente acquisiti dal laboratorio sono analizzati nella sezione 10.15.

### 10.15 Gate quadratico sulle sei scansioni: miglioramento parziale, ancora fallito

**Evidenze sincronizzate.** `outputs/segmentation_overfit_squared/` contiene
gate finale e best, config/manifest, 50 epoche, 300 aggiornamenti, 12
valutazioni, 72 righe di storia per caso, 48 crop finali, sei riepiloghi
per scansione e 16 figure. La loss dichiarata è 0,5 BCE + 0,5 Dice
quadratico, BCE anche sui negativi. Il config differisce dall'overfit
originale solo per `squared_pred: true` e percorsi output/gate; manifest
identico, 16 crop positivi e 32 negativi, conteggi dei target coincidenti
con quelli dei precedenti esperimenti. Ogni scansione compare in 50
aggiornamenti; il totale è 300, non 300 esposizioni per crop.

**Controlli.** Ricalcolati Dice e FPR dai conteggi di `crops.csv`, con
identità TP+FN=target e TP+FP=predizione per tutti i crop: coincidono con
`gate.json`. Numerazione degli aggiornamenti completa, gradienti e
aggiornamenti dei due gruppi sempre finiti e positivi, identità della
loss nei CSV entro `1e-6`. Il best selezionato corrisponde al controllo
275. Questo verifica la coerenza degli aggregati, non una nuova inferenza:
non sono sincronizzati pesi o array di probabilità di questa run, né il
trainer salva qui l'hash iniziale completo o gli hash dei crop.

| Misura | Overfit lineare originale | Overfit quadratico last |
|---|---:|---:|
| Dice medio dei 16 crop positivi (criterio del gate) | 0,709937 | 0,782871 |
| FPR sui 32 crop negativi | 1,69959% | 0,37899% |
| Dice micro sui voxel dei crop positivi | 0,751059 | 0,762823 |
| Precision micro nei crop positivi | 0,644326 | 0,875212 |
| Recall micro nei crop positivi | 0,900175 | 0,676014 |
| FPR sui soli 16 crop reali | 1,32985% | 0,10369% |

Il best a 275 ha Dice medio 0,785942 e FPR 0,56068%: anch'esso fallisce.
Nessuno dei 12 controlli supera il gate; nessuno dei 16 crop positivi
finali raggiunge individualmente Dice 0,90. Il criterio negativo globale
è soddisfatto, quello Dice no. La riduzione del FPR negativo rispetto
all'originale è circa il 77,7%, ma non equivale a un recupero uniforme
della localizzazione: i falsi negativi nei crop positivi passano da
27.344 a 88.746, su 273.919 voxel target, mentre i FP scendono da 136.112
a 26.402. Migliorano 11 crop, peggiorano cinque. Sono confronti storici
a config/manifest coerenti, non una nuova replica appaiata con controllo
dello stato iniziale. Rispetto anche al precedente grouped lineare
(Dice micro 0,773298) e mixed lineare (0,788615), il micro quadratico è
inferiore: la crescita della media per crop non basta a promuovere la loss.

**Errori residui.** Dice medio per volume: `inj_2092` 0,792656,
`inj_1419` 0,681973, `rem_133` 0,804755, `rem_15` 0,852100.
Il peggiore è ora `inj_1419/1`: target 59.200 voxel, Dice 0,460777,
recall 0,331791, 39.558 FN (44,6% di tutti i FN positivi) e 6.414 FP.
La figura `01_pix2pix_inj_1419_worst_positive_crop1.png` mostra recupero
parziale del riquadro annotato e ampie zone FN, non solo un sottile bordo.
I suoi negativi producono 27.159 dei 31.792 FP negativi (85,4%); il crop
5 ha FPR 5,896%, pur con media globale sotto 1%. I reali hanno ancora
4.349 FP complessivi. Non dichiarare tutti i negativi corretti.

I due crop prima isolati valgono ora 0,597976 (`inj_2092/3`: recall 1,
1.748 FP) e 0,808953 (`rem_133/2`: 736 FP, 476 FN). Il loro Dice 1 nelle
prove isolate era ottenuto dopo 300 esposizioni ciascuno, contro le 50
di questa run: non è una prova appaiata di interferenza fra scansioni.
Inoltre il nuovo peggior target è grande; il problema residuo non è
spiegato soltanto dai target piccoli. La storia per caso resta oscillante:
fra 250 e 300 `inj_1419` scende da 0,7804 a 0,6820, mentre injection2092
e removal133 migliorano. Una competizione fra aggiornamenti per volume
è un'ipotesi da verificare, non una causa dimostrata. Clipping attivo in
86/300 passi, anche 21 degli ultimi 50: non è provato che causi gli errori.

**Decisione.** Piloti/full ancora bloccati. Non abbassare il criterio,
non cambiare soglia e non prolungare questa run oltre il budget deciso.
Il prossimo confronto motivato è **grouped contro mixed con Dice
quadratico in entrambi i rami**, stesso pool di 48 crop verificato con
hash, stessa inizializzazione fresca, 300 aggiornamenti/50 esposizioni,
ponderazione Dice foreground fissa a quattro e stesso valutatore. Cambia
solo la composizione dei batch; valutare anche recall e dettagli per crop,
senza accettare come successo soltanto la riduzione dei FP. Il precedente
`compare_segmentation_batches.run_arm` non inoltra `squared_pred` e usa
ancora il Dice lineare: lanciare quel comando con il nuovo YAML non
realizzerebbe il confronto desiderato. L'integrazione successiva è descritta
nella sezione 10.16. Nessun training avviato in questa analisi; nessuna
conclusione sulla validation o sulla generalizzazione.

### 10.16 Confronto grouped/mixed quadratico: implementazione

**Intervento.** Riutilizzato `compare_segmentation_batches` con
`configs/segmentation_overfit_squared.yaml` e output distinto
`outputs/segmentation_batch_comparison_squared`. Il runner ora inoltra
`training.squared_pred` alla loss in entrambi i rami; default falso per
le configurazioni precedenti. I pesi BCE/Dice, la variante e il trattamento
BCE-only dei crop vuoti sono registrati in checkpoint, gate e report;
ogni riga del CSV di aggiornamento identifica il denominatore usato.
Nessun nuovo entrypoint, config duplicato o dipendenza.

Il pool condiviso conserva i 48 crop originali, identificati dagli hash
di immagine e target. Due inizializzazioni fresche MedicalNet, stesso
seed e stato completo: il mixed verifica l'hash del grouped prima del
training. Restano fissi normalizzatore Dice foreground quattro, BCE su
otto crop, preprocessing, LR, batch fisico/accumulo, 300 aggiornamenti
per ramo e 50 esposizioni per crop. Si cambia solo la composizione dei
batch; un gate superato non accorcia il budget del ramo.

**Evidenze best/last.** Oltre ai precedenti output del last, ogni ramo
ricarica il best già salvato ed esporta `best/crops.csv`, `best/cases.csv`
e figure con il valutatore comune FP32 a soglia 0,5. Questo consente di
analizzare recall, precision e conteggi FP/FN anche sul best senza
richiedere un nuovo comando diagnostico. Le metriche del best devono
riprodursi entro `1e-4`, altrimenti l'errore interrompe il confronto.
Il last viene salvato prima di caricare il best, conservando i veri pesi
finali. Non vengono esportate nuove mappe di probabilità complete.

`comparison.json` distingue `complete` dal gate `passed` del last mixed;
rimangono ramo attivo ed eventuale errore tecnico. Il budget, le esposizioni
e la riproduzione del best devono essere completati prima di dichiarare
il confronto concluso. I checkpoint/gate precedenti non sono modificati,
una directory non vuota viene rifiutata e non è previsto resume. Nessuna
promozione automatica a piloti/full, nessuna conclusione sulla validation.

**Verifica locale.** Test a pesi fermi coprono uguaglianza delle loss e
dei gradienti per grouped/mixed anche con Dice quadratico e il fattore
errato 1,5 senza normalizzatore fisso. Verificato accumulo 1×8/2×4 con
entrambi i denominatori. Fixture TIFF esercitano i due rami, inoltro
effettivo della variante, hash iniziali/pool, esposizioni, metadati,
esportazioni dei best e mantenimento dei vecchi gate. Controllati budget
completo anche dopo successo precoce, distinzione fra pesi best e last,
blocco prima del training per hash iniziale diverso, rifiuto di metriche
best non riproducibili e CLI del nuovo comando. Superati 102 test pertinenti,
con un test dei worker persistenti escluso (non usati dal confronto) e
18 warning MONAI già presenti. Comando Windows nella
sezione 18.2.7 della guida. L'esecuzione CT/CUDA resta da svolgere nel
laboratorio; non sono state avviate nuove run scientifiche localmente.

## Come aggiornare questo registro

Per ogni punto chiuso, sostituire «Da compilare» con: **verifica eseguita**
(split, numero di casi e metodo), **risultato** (misure e figure), **decisione**
(problema confermato, assente nel perimetro verificato o ancora incerto) e
**limite**. Collegare gli output riproducibili; non chiamare «risolto» un punto
solo perché il codice per misurarlo è presente.
