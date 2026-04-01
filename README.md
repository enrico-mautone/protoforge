# ProtoForge - Neural Protocol Compiler

Un sistema di machine learning per generare file Protocol Buffers (.proto) da descrizioni in linguaggio naturale.

## Setup Ambiente Virtuale

### 1. Creazione Virtual Environment
```bash
python3 -m venv venv
```

### 2. Attivazione
```bash
# Su Linux/Mac
source venv/bin/activate

# Su Windows
venv\Scripts\activate
```

### 3. Installazione Dipendenze
```bash
pip install -r requirements.txt
```

## Avvio Training

```bash
# Training con parametri di default
python protoforge_train.py

# Training con parametri custom
python protoforge_train.py --epochs 20 --batch-size 4 --lr 1e-4 --output-dir ./my_output
```

## Opzioni Training

- `--epochs`: Numero di epoche (default: 10)
- `--batch-size`: Dimensione batch (default: 8)
- `--lr`: Learning rate (default: 5e-4)
- `--use-lora`: Usa LoRA per efficient fine-tuning (default: True)
- `--output-dir`: Directory output (default: ./protoforge_output)

## Requisiti di Sistema

- Python 3.8+
- CUDA supportito (opzionale, per GPU acceleration)
- 8GB+ RAM consigliati
- Protobuf compiler per validazione: `sudo apt install protobuf-compiler`
