# Baseline di segmentazione 3D U-Net e tracce CT-GAN su pix2pix

Questa pipeline affianca, senza sostituirla, la baseline patch-wise. Il classificatore assegna uno score a ogni patch; la 3D U-Net produce invece una probabilità per ogni voxel della patch e viene supervisionata direttamente dalle mask M3Dsynth.

## Ambito

- Manipolazioni sintetiche: soltanto `pix2pix` in training e validation.
- Negativi: TAC reali, con target interamente nullo.
- Split: quelli ufficiali presenti nei metadata, sempre a livello di volume.
- CycleGAN e Diffusion: non vengono usati in questa baseline. Rimangono indicati come futuri domini di test, senza essere caricati dal trainer.
- Sorgente dati: corpus pix2pix originale completo, senza ricampionamento isotropico. Le patch `64³` indicano quindi voxel, non millimetri.

La configurazione è in `configs/train_pix2pix_unet_baseline.yaml`.

Questa prima configurazione e il relativo risultato con Dice nullo sono mantenuti come controllo negativo **R0**. I nuovi esperimenti sono:

- **R1**, `configs/train_pix2pix_unet_ct.yaml`: CT normalizzata;
- **R2**, `configs/train_pix2pix_unet_highpass.yaml`: CT normalizzata più residuo high-pass 3D.
- **R3**, `configs/train_pix2pix_unet_ct_centered_32.yaml`: CT normalizzata, patch `32³` e crop positivi centrati sulla mask con jitter deterministico ±8 voxel.
- **R4**, `configs/train_pix2pix_unet_ct_grid_32.yaml`: CT normalizzata e patch `32³` prese soltanto dalla griglia regolare, senza centering o jitter.

Il corpus resta quello pix2pix originale completo. Non viene usato il corpus isotropico parziale.

R3 conserva negativi sulla griglia regolare `32³`/stride `16³`. Per ogni volume manipolato indicizza otto crop positivi attorno al centro della mask; il sampler ne seleziona quattro per batch insieme a quattro negativi. Il crop esattamente centrato è sempre presente e gli altri ricevono jitter riproducibile con seed 21, evitando che la rete veda la manipolazione sempre nella stessa posizione.

R4 mantiene patch, stride, batch, loss, split e seed di R3, ma usa anche per i positivi la griglia regolare. Isola quindi l'effetto della dimensione `32³` da quello dei crop centrati.

## Relazione con CT-GAN

Il ramo pix2pix di M3Dsynth usa [CT-GAN](https://arxiv.org/abs/1901.03597): un cubo fisico di lato 32 mm viene ricampionato a `32³`, la regione centrale viene rigenerata e il risultato subisce inverse scaling, aggiunta di rumore e blending. Le 100 TAC mostrate nell'esperimento CT-GAN appartengono alla valutazione clinica (80 blind e 20 open), non al training del generatore. I due GAN furono addestrati su cubi estratti da 888 TAC LIDC-IDRI. Il [repository ufficiale](https://github.com/ymirsky/CT-GAN) documenta il preprocessing e il touch-up; il [paper M3Dsynth](https://arxiv.org/abs/2309.07973) conferma il riuso di CT-GAN per pix2pix.

Il codice CT-GAN non viene copiato né eseguito. Viene usato soltanto per formulare l'ipotesi forense che interpolazione, touch-up e fusione lascino tracce locali ad alta frequenza.

## Dati e sampling

Il flusso riusa l'indice patch, l'allineamento scan/mask e la cache TIFF della pipeline esistente. Ogni elemento contiene:

- `image`: patch TAC normalizzata, forma `(1, 64, 64, 64)`;
- `mask`: target binario allineato, stessa forma;
- i metadati patch già usati dal classificatore.

R0 considera positiva una patch con almeno un voxel manipolato. R1/R2 richiedono invece un overlap minimo di `0.001`, scartando le patch marginali ambigue. Un batch proviene da un solo volume, evitando di decodificare ripetutamente TIFF molto grandi. La baseline usa patch `64³`, stride `32³`, batch size `2`, quattro patch positive su otto patch selezionate per volume e 512 patch al massimo per epoca.

## Modello

La rete è una U-Net 3D minima:

- tre livelli encoder con canali `16, 32, 64`;
- bottleneck a 128 canali;
- due convoluzioni `3×3×3` e ReLU per blocco;
- tre skip connection;
- upsampling con convoluzioni trasposte `2×2×2`;
- convoluzione finale `1×1×1` e sigmoide, con un solo canale di output.

Non sono presenti attention, residual block, deep supervision, dropout o normalizzazioni aggiuntive. Le dimensioni delle patch devono essere divisibili per 8.

## Loss, validation ed early stopping

R0 usa la loss originale BCE più soft-Dice. R1 e R2 usano `0.5 × focal(alpha=0.75, gamma=2) + 0.5 × soft-Dice`, per ridurre il collasso sul background dovuto allo sbilanciamento voxel-level. R2 concatena internamente alla CT il residuo `CT - media locale 3×3×3`, calcolato su GPU con padding riflesso.

A ogni epoca la validation riporta loss, soft-Dice, Dice a soglia 0.5, IoU, precision, recall e probabilità media sui voxel positivi e negativi. Per R1/R2 `best.pt` viene aggiornato sul soft-Dice, evitando che una soglia non calibrata nasconda l'apprendimento. Il training termina dopo cinque epoche consecutive senza miglioramento.

## Audit delle tracce

Prima del nuovo training confrontare 50 injection e 50 removal con le TAC pristine associate:

```powershell
python -m tesi_m3d.audit_ctgan_traces `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --mod pix2pix `
  --max-records 100 `
  --output-dir outputs\ctgan_trace_audit
```

L'audit produce `summary.json`, `records.csv` e sei pannelli ortogonali. Riporta energia e percentuale dei residui dentro la mask, nei gusci esterni e nel background, differenza high-pass, distanza tra coordinate metadata e centro mask, spacing e disallineamenti geometrici. Le mask non vengono ampliate.

## Esecuzione R1/R2

Eseguire il dry-run per entrambe le configurazioni e verificare che record, patch e quantili di overlap coincidano:

```powershell
python -m tesi_m3d.train_unet `
  --config configs\train_pix2pix_unet_ct.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda `
  --dry-run

python -m tesi_m3d.train_unet `
  --config configs\train_pix2pix_unet_highpass.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda `
  --dry-run
```

Prima delle run complete, ogni modello deve superare il micro-overfit su due patch positive con Dice maggiore di 0.90:

```powershell
python -m tesi_m3d.train_unet --config configs\train_pix2pix_unet_ct.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda --micro-overfit
python -m tesi_m3d.train_unet --config configs\train_pix2pix_unet_highpass.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda --micro-overfit
```

Training R1 e R2:

```powershell
python -m tesi_m3d.train_unet `
  --config configs\train_pix2pix_unet_ct.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda

python -m tesi_m3d.train_unet `
  --config configs\train_pix2pix_unet_highpass.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda
```

Per R3 eseguire nell'ordine dry-run, micro-overfit e training completo:

```powershell
python -m tesi_m3d.train_unet --config configs\train_pix2pix_unet_ct_centered_32.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda --dry-run
python -m tesi_m3d.train_unet --config configs\train_pix2pix_unet_ct_centered_32.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda --micro-overfit
python -m tesi_m3d.train_unet --config configs\train_pix2pix_unet_ct_centered_32.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda
```

Per R4 usare gli stessi tre controlli:

```powershell
python -m tesi_m3d.train_unet --config configs\train_pix2pix_unet_ct_grid_32.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda --dry-run
python -m tesi_m3d.train_unet --config configs\train_pix2pix_unet_ct_grid_32.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda --micro-overfit
python -m tesi_m3d.train_unet --config configs\train_pix2pix_unet_ct_grid_32.yaml --data-root "C:\Tesi Magistrale Piscopo" --device cuda
```

Output principali nelle directory definite dalle singole configurazioni:

- `dry_run_stats.json`: numerosità degli indici;
- `metrics.csv`: andamento per epoca;
- `best.pt`: checkpoint con miglior Dice per R0 o soft-Dice per R1/R2;
- `unet3d_last.pt`: stato dell'ultima epoca eseguita.

## Controlli prima della run completa

1. Eseguire il dry-run e verificare che train e validation contengano patch positive.
2. Eseguire una breve run con `epochs: 2` e controllare memoria GPU e tempi.
3. Se `64³ × batch 2` causa out-of-memory, ridurre soltanto il batch a 1; mantenere inizialmente patch e architettura invariati.
4. Controllare che Dice e recall non rimangano a zero e ispezionare visivamente alcune predizioni prima della run lunga.

## Valutazione full-volume

Dopo il micro-overfit e il training, eseguire la sliding-window `64³` con stride `32³` e blending gaussiano:

```powershell
python -m tesi_m3d.evaluate_unet `
  --checkpoint outputs\train_pix2pix_unet_ct\best.pt `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda `
  --batch-size 2 `
  --output-dir outputs\train_pix2pix_unet_ct\full_volume_validation

python -m tesi_m3d.evaluate_unet `
  --checkpoint outputs\train_pix2pix_unet_highpass\best.pt `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda `
  --batch-size 2 `
  --output-dir outputs\train_pix2pix_unet_highpass\full_volume_validation
```

La soglia è calibrata esclusivamente sulla validation. `summary.json` riporta Dice/IoU/precision/recall macro, voxel AUC/AP approssimate con 1000 bin, risultati injection/removal e false-positive rate sulle TAC reali; `volumes.csv` conserva i risultati per volume.

R3 usa automaticamente finestre `32³` con stride `16³`, ricavate dal checkpoint:

```powershell
python -m tesi_m3d.evaluate_unet --checkpoint outputs\train_pix2pix_unet_ct_centered_32\best.pt --data-root "C:\Tesi Magistrale Piscopo" --device cuda --batch-size 8 --output-dir outputs\train_pix2pix_unet_ct_centered_32\full_volume_validation
```

Evaluation R4:

```powershell
python -m tesi_m3d.evaluate_unet --checkpoint outputs\train_pix2pix_unet_ct_grid_32\best.pt --data-root "C:\Tesi Magistrale Piscopo" --device cuda --batch-size 8 --output-dir outputs\train_pix2pix_unet_ct_grid_32\full_volume_validation
```

Confrontare R0, R1 e R2 usando gli stessi 64 record di validation. Se R2 supera R1 di almeno 0.02 Dice macro senza peggiorare il false-positive rate, ripetere R1/R2 con tre seed e riportare media e deviazione standard. CycleGAN e Diffusion restano esclusi da questa fase.
