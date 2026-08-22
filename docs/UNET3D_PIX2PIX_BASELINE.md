# Baseline di segmentazione 3D U-Net su pix2pix

Questa pipeline affianca, senza sostituirla, la baseline patch-wise. Il classificatore assegna uno score a ogni patch; la 3D U-Net produce invece una probabilità per ogni voxel della patch e viene supervisionata direttamente dalle mask M3Dsynth.

## Ambito

- Manipolazioni sintetiche: soltanto `pix2pix` in training e validation.
- Negativi: TAC reali, con target interamente nullo.
- Split: quelli ufficiali presenti nei metadata, sempre a livello di volume.
- CycleGAN e Diffusion: non vengono usati in questa baseline. Rimangono indicati come futuri domini di test, senza essere caricati dal trainer.
- Sorgente dati: corpus pix2pix originale completo, senza ricampionamento isotropico. Le patch `64³` indicano quindi voxel, non millimetri.

La configurazione è in `configs/train_pix2pix_unet_baseline.yaml`.

## Dati e sampling

Il flusso riusa l'indice patch, l'allineamento scan/mask e la cache TIFF della pipeline esistente. Ogni elemento contiene:

- `image`: patch TAC normalizzata, forma `(1, 64, 64, 64)`;
- `mask`: target binario allineato, stessa forma;
- i metadati patch già usati dal classificatore.

Le patch con almeno un voxel manipolato sono considerate positive per il sampling. Un batch proviene da un solo volume, evitando di decodificare ripetutamente TIFF molto grandi. La baseline usa patch `64³`, stride `32³`, batch size `2`, quattro patch positive su otto patch selezionate per volume e 512 patch al massimo per epoca.

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

La loss è la media pesata in parti uguali di binary cross-entropy e soft Dice loss. La BCE viene calcolata in float32 anche quando il resto del forward usa mixed precision, perché riceve probabilità già passate dalla sigmoide.

A ogni epoca la validation riporta loss, Dice, IoU, precision e recall voxel-level, aggregati sulle patch selezionate. `best.pt` viene aggiornato quando migliora il Dice. Il training termina obbligatoriamente dopo cinque epoche consecutive senza miglioramento; `epochs: 100` è soltanto il limite massimo.

## Esecuzione

Verifica iniziale di dataset, indice e sampling senza training:

```powershell
python -m tesi_m3d.train_unet `
  --config configs\train_pix2pix_unet_baseline.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda `
  --dry-run
```

Training:

```powershell
python -m tesi_m3d.train_unet `
  --config configs\train_pix2pix_unet_baseline.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda
```

Output principali in `outputs/train_pix2pix_unet_baseline`:

- `dry_run_stats.json`: numerosità degli indici;
- `metrics.csv`: andamento per epoca;
- `best.pt`: checkpoint con miglior Dice validation;
- `unet3d_last.pt`: stato dell'ultima epoca eseguita.

## Controlli prima della run completa

1. Eseguire il dry-run e verificare che train e validation contengano patch positive.
2. Eseguire una breve run con `epochs: 2` e controllare memoria GPU e tempi.
3. Se `64³ × batch 2` causa out-of-memory, ridurre soltanto il batch a 1; mantenere inizialmente patch e architettura invariati.
4. Controllare che Dice e recall non rimangano a zero e ispezionare visivamente alcune predizioni prima della run lunga.

## Limiti intenzionali della prima baseline

La validation corrente è patch-level e usa soglia fissa 0.5. Sliding-window su volume completo, blending delle sovrapposizioni, calibrazione della soglia, metriche per volume/componente e test cross-generator saranno aggiunti dopo aver verificato che questa baseline apprenda mask pix2pix non banali.
