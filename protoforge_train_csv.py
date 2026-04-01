#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  ██████╗ ██████╗  ██████╗ ████████╗ ██████╗ ███████╗ ██████╗ ███████╗  ║
║  ██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝██╔═══██╗██╔════╝██╔═══██╗██╔════╝  ║
║  ██████╔╝██████╔╝██║   ██║   ██║   ██║   ██║█████╗  ██║   ██║███████╗  ║
║  ██╔═══╝ ██╔══██╗██║   ██║   ██║   ██║   ██║██╔══╝  ██║   ██║╚════██║  ║
║  ██║     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║     ╚██████╔╝███████║  ║
║  ╚═╝     ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝      ╚═════╝ ╚══════╝  ║
║                                                                    ║
║  Neural Compiler: Natural Language → Protocol Buffers (.proto)    ║
║  Versione 2.0 - Supporto CSV Dataset                             ║
╚══════════════════════════════════════════════════════════════════╝
"""

# FASE 1: Setup e Importazioni
# Importiamo tutte le librerie necessarie per il training del modello
import os  # Gestione percorsi file system
import sys  # Accesso a funzioni di sistema
import csv  # Lettura/scrittura file CSV
import json  # Gestione dati JSON
import argparse  # Gestione argomenti da linea di comando
import logging  # Sistema di logging per tracciare l'esecuzione
from pathlib import Path  # Gestione avanzata percorsi
from typing import List, Dict, Tuple, Optional  # Type hints per tipi di dati
from datetime import datetime  # Gestione timestamp

# Import PyTorch - framework principale per deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader  # Gestione dataset e batching

# Import scikit-learn per splitting dati
from sklearn.model_selection import train_test_split

# Import Transformers - HuggingFace per modelli pre-addestrati
from transformers import (
    T5ForConditionalGeneration,  # Modello T5 per generazione testo
    T5Tokenizer,  # Tokenizer per T5
    AdamW,  # Ottimizzatore Adam con weight decay
    get_linear_schedule_with_warmup,  # Scheduler per learning rate
    EarlyStoppingCallback,  # Callback per early stopping
    Trainer,  # Classe principale per training
    TrainingArguments,  # Configurazione parametri training
    DataCollatorForSeq2Seq  # Gestione batching per sequence-to-sequence
)

# Import PEFT per efficient fine-tuning (LoRA)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Import pandas per manipolazione dati CSV
import pandas as pd
from datasets import Dataset as HFDataset

# FASE 2: Configurazione del Sistema di Logging
# Impostiamo il logging per monitorare l'esecuzione del programma
logging.basicConfig(
    level=logging.INFO,  # Livello INFO per informazioni dettagliate
    format='%(asctime)s | %(levelname)s | %(message)s',  # Formato con timestamp e livello
    handlers=[
        # Salva log su file con timestamp automatico nel nome
        logging.FileHandler(f'protoforge_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        # Mostra log anche su console
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)  # Logger principale per questo modulo


# FASE 3: Classe Dataset per Gestione Dati CSV
# Questa classe gestisce il caricamento e la preprocessing dei dati dal CSV
class ProtoDatasetCSV(Dataset):
    """
    Dataset personalizzato per caricare dati da CSV e prepararli per il training.
    Eredita da Dataset di PyTorch per integrazione con DataLoader.
    """
    def __init__(self, data: List[Dict], tokenizer, max_length_nl: int = 128, max_length_proto: int = 512):
        """
        Inizializza il dataset.
        
        Args:
            data: Lista di dizionari con i dati di training
            tokenizer: Tokenizer di T5 per convertire testo in token
            max_length_nl: Lunghezza massima input (natural language)
            max_length_proto: Lunghezza massima output (codice proto)
        """
        self.data = data  # Salva i dati grezzi
        self.tokenizer = tokenizer  # Salva il tokenizer
        self.max_length_nl = max_length_nl  # Limiti di lunghezza per input
        self.max_length_proto = max_length_proto  # Limiti per output

    def __len__(self):
        """Ritorna il numero totale di campioni nel dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Prepara un singolo campione per il training.
        Questo metodo è chiamato automaticamente dal DataLoader.
        
        Args:
            idx: Indice del campione da recuperare
            
        Returns:
            Dizionario con input_ids, attention_mask, e labels per il modello
        """
        # FASE 3.1: Recupera il campione dal dataset
        item = self.data[idx]

        # FASE 3.2: Prepara l'input per il modello T5
        # Aggiungiamo un prefisso per indicare al modello cosa deve fare
        input_text = f"generate proto: {item['natural_language']}"

        # FASE 3.3: Prepara il target (output desiderato)
        target_text = item['proto_code']

        # FASE 3.4: Tokenizzazione dell'input
        # Convertiamo il testo in sequenze di numeri (token)
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length_nl,  # Tronca se troppo lungo
            padding='max_length',  # Aggiungi padding se troppo corto
            truncation=True,  # Abilita troncamento
            return_tensors='pt'  # Ritorna tensori PyTorch
        )

        # FASE 3.5: Tokenizzazione del target
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length_proto,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # FASE 3.6: Prepara le labels per la loss function
        # Cloniamo i target per creare le labels
        labels = targets['input_ids'].clone()
        # Sostituiamo i padding token con -100 per ignorarli nel calcolo della loss
        # Questo è standard practice in NLP per non penalizzare il padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        # FASE 3.7: Ritorna il dizionario con i tensori per il training
        return {
            'input_ids': inputs['input_ids'].squeeze(),  # Rimuovi dimensione extra
            'attention_mask': inputs['attention_mask'].squeeze(),  # Maschera per attention
            'labels': labels.squeeze()  # Target per calcolo loss
        }


# FASE 4: Classe Principale ProtoForge - Neural Compiler
# Questa è la classe principale che gestisce tutto il processo di training
class ProtoForge:
    """
    ProtoForge: Neural Compiler per Protocol Buffers
    
    Questa classe implementa un sistema completo per:
    1. Caricare modelli T5 pre-addestrati
    2. Applicare LoRA per efficient fine-tuning
    3. Eseguire training su dataset CSV
    4. Generare codice .proto da linguaggio naturale
    5. Validare il codice generato
    """

    def __init__(
        self,
        base_model: str = "t5-small",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device: str = None
    ):
        """
        Inizializza il Neural Compiler ProtoForge.
        
        Args:
            base_model: Nome del modello HuggingFace da usare come base
            use_lora: Se applicare LoRA per efficient fine-tuning
            lora_r: Rank della matrice LoRA (dimensione)
            lora_alpha: Scaling factor per LoRA
            lora_dropout: Dropout rate per LoRA
            device: Dispositivo di calcolo (cuda/cpu)
        """
        # FASE 4.1: Configurazione dispositivo di calcolo
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model_name = base_model

        # FASE 4.2: Logging delle informazioni di inizializzazione
        logger.info("="*70)
        logger.info("INIZIALIZZAZIONE PROTOFORGE")
        logger.info("="*70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Base model: {base_model}")
        logger.info(f"LoRA: {use_lora} (r={lora_r}, alpha={lora_alpha})")

        # FASE 4.3: Caricamento del modello e del tokenizer
        try:
            # Carica il tokenizer per convertire testo in token
            self.tokenizer = T5Tokenizer.from_pretrained(base_model)
            # Carica il modello T5 per sequence-to-sequence
            self.model = T5ForConditionalGeneration.from_pretrained(base_model)
            logger.info(f"Modello caricato con successo")
        except Exception as e:
            logger.error(f"Errore caricamento modello: {e}")
            raise

        # FASE 4.4: Applica LoRA se richiesto
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout)

        # FASE 4.5: Sposta il modello sul dispositivo corretto
        self.model.to(self.device)

        # FASE 4.6: Calcola e logga le statistiche dei parametri
        self._log_parameters()

    def _apply_lora(self, r: int, alpha: int, dropout: float):
        """
        FASE 4.7: Applica LoRA (Low-Rank Adaptation) per efficient fine-tuning.
        
        LoRA è una tecnica che permette di fare fine-tuning di modelli grandi
        senza dover aggiornare tutti i parametri, riducendo drasticamente
        i requisiti di memoria e calcolo.
        
        Args:
            r: Rank della matrice LoRA (quanti parametri aggiuntivi usare)
            alpha: Fattore di scaling per i parametri LoRA
            dropout: Rate di dropout per prevenire overfitting
        """
        # Configurazione LoRA
        config = LoraConfig(
            r=r,  # Rank delle matrici di adattamento
            lora_alpha=alpha,  # Fattore di scaling
            target_modules=["q", "v", "k", "o"],  # Moduli target nell'attention
            lora_dropout=dropout,  # Dropout per regolarizzazione
            bias="none",  # Nessun bias per i parametri LoRA
            task_type=TaskType.SEQ_2_SEQ_LM  # Tipo di task: sequence-to-sequence
        )

        # Applica LoRA al modello
        self.model = get_peft_model(self.model, config)
        logger.info(f"LoRA applicata: r={r}, alpha={alpha}, dropout={dropout}")

    def _log_parameters(self):
        """
        FASE 4.8: Calcola e logga le statistiche dei parametri del modello.
        
        Questo ci aiuta a capire quanto è grande il modello e quanti
        parametri verranno effettivamente addestrati.
        """
        # Calcola parametri totali
        total_params = sum(p.numel() for p in self.model.parameters())
        # Calcola solo i parametri trainabili (con requires_grad=True)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Logga le statistiche
        logger.info("-"*70)
        logger.info(f"Parametri totali:     {total_params:,}")
        logger.info(f"Parametri trainabili: {trainable_params:,}")
        logger.info(f"Percentuale trainable: {100*trainable_params/total_params:.2f}%")
        logger.info("="*70)

    def load_data_from_csv(self, csv_path: str, split_column: str = 'split') -> Tuple[List[Dict], List[Dict]]:
        """
        FASE 5: Caricamento e preprocessing dati da CSV.
        
        Questo metodo gestisce il caricamento dei dati di training da un file CSV,
        verificando la presenza delle colonne necessarie e dividendo i dati
        in training set e validation set.
        
        Args:
            csv_path: Percorso del file CSV contenente i dati
            split_column: Nome della colonna che indica se un campione è train/test
        
        Returns:
            Tuple[List[Dict], List[Dict]]: (train_data, test_data)
        """
        logger.info(f"Caricamento dati da: {csv_path}")

        # FASE 5.1: Verifica esistenza file
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File non trovato: {csv_path}")

        # FASE 5.2: Carica il CSV usando pandas
        df = pd.read_csv(csv_path)
        logger.info(f"Totale righe CSV: {len(df)}")

        # FASE 5.3: Verifica colonne richieste
        required_cols = ['natural_language', 'proto_code']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colonne mancanti: {missing}")

        # FASE 5.4: Divisione dati train/test
        if split_column in df.columns:
            # Usa la colonna split se presente nel CSV
            train_df = df[df[split_column] == 'train']
            test_df = df[df[split_column] == 'test']

            train_data = train_df.to_dict('records')
            test_data = test_df.to_dict('records')

            logger.info(f"Split da colonna '{split_column}':")
            logger.info(f"  Train: {len(train_data)} campioni")
            logger.info(f"  Test:  {len(test_data)} campioni")
        else:
            # Se non c'è colonna split, fai divisione automatica 80/20
            logger.warning(f"Colonna '{split_column}' non trovata, uso split 80/20")
            data = df.to_dict('records')
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

        return train_data, test_data

    def train(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        output_dir: str = "./protoforge_output",
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 5e-4,
        warmup_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 100,
        logging_steps: int = 10,
        early_stopping_patience: int = 3,
        gradient_accumulation_steps: int = 2,
        max_grad_norm: float = 1.0
    ):
        """
        FASE 6: Training completo del modello usando HuggingFace Trainer.
        
        Questo è il cuore del sistema: esegue il fine-tuning del modello T5
        sui dati di training per imparare a generare codice .proto da
        descrizioni in linguaggio naturale.
        
        Args:
            train_data: Dati di training (lista di dizionari)
            val_data: Dati di validazione
            output_dir: Directory dove salvare il modello
            num_epochs: Numero di epoche di training
            batch_size: Dimensione del batch per GPU
            learning_rate: Learning rate iniziale
            warmup_steps: Passi di warmup per lo scheduler
            save_steps: Ogni quanti passi salvare il modello
            eval_steps: Ogni quanti passi fare validazione
            logging_steps: Ogni quanti passi loggare metriche
            early_stopping_patience: Quante epoche aspettare prima di fermarsi
            gradient_accumulation_steps: Quanti batch accumulare prima di update
            max_grad_norm: Norma massima dei gradienti (gradient clipping)
            
        Returns:
            Trainer: L'oggetto trainer allenato
        """
        # FASE 6.1: Logging dei parametri di training
        logger.info("\n" + "="*70)
        logger.info("AVVIO TRAINING")
        logger.info("="*70)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")

        # FASE 6.2: Creazione dei dataset PyTorch
        # Converte i dati grezzi in dataset utilizzabili da PyTorch
        train_dataset = ProtoDatasetCSV(train_data, self.tokenizer)
        val_dataset = ProtoDatasetCSV(val_data, self.tokenizer)

        # FASE 6.3: Configurazione del Data Collator
        # Il data collator gestisce il batching dinamico e il padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True  # Aggiunge padding dinamico per ogni batch
        )

        # FASE 6.4: Configurazione dei parametri di training
        # TrainingArguments definisce TUTTI gli aspetti del training
        training_args = TrainingArguments(
            output_dir=output_dir,  # Directory output per checkpoint e logs
            num_train_epochs=num_epochs,  # Numero totale di epoche
            per_device_train_batch_size=batch_size,  # Batch size per GPU training
            per_device_eval_batch_size=batch_size,  # Batch size per GPU evaluation
            learning_rate=learning_rate,  # Learning rate iniziale
            warmup_steps=warmup_steps,  # Passi di warmup per lo scheduler
            weight_decay=0.01,  # Weight decay per regolarizzazione L2
            logging_dir=f"{output_dir}/logs",  # Directory per TensorBoard logs
            logging_steps=logging_steps,  # Frequenza logging metriche
            eval_strategy="steps",  # Strategia evaluation: ogni N passi
            eval_steps=eval_steps,  # Frequenza evaluation
            save_strategy="steps",  # Strategia salvataggio: ogni N passi
            save_steps=save_steps,  # Frequenza salvataggio checkpoint
            save_total_limit=3,  # Massimo numero di checkpoint salvati
            load_best_model_at_end=True,  # Carica il miglior modello alla fine
            metric_for_best_model="eval_loss",  # Metrica per determinare il miglior modello
            greater_is_better=False,  # Loss: più bassa è meglio
            fp16=torch.cuda.is_available(),  # Usa mixed precision se GPU disponibile
            gradient_accumulation_steps=gradient_accumulation_steps,  # Accumula gradienti
            max_grad_norm=max_grad_norm,  # Gradient clipping per stabilità
            report_to=["tensorboard"],  # Invia metriche a TensorBoard
            remove_unused_columns=False,  # Manti tutte le colonne del dataset
            dataloader_num_workers=2,  # Worker paralleli per data loading
        )

        # FASE 6.5: Creazione dell'oggetto Trainer
        # Il Trainer gestisce automaticamente tutto il ciclo di training
        trainer = Trainer(
            model=self.model,  # Il modello da allenare
            args=training_args,  # Configurazione training
            train_dataset=train_dataset,  # Dataset di training
            eval_dataset=val_dataset,  # Dataset di validazione
            data_collator=data_collator,  # Gestore batching
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]  # Early stopping
        )

        # FASE 6.6: Esecuzione del training
        # Questo è il punto in cui il modello effettivamente impara
        logger.info("\nTraining in corso...")
        trainer.train()  # Avvia il ciclo di training completo

        # FASE 6.7: Salvataggio del modello finale
        # Salva il miglior modello e il tokenizer per uso futuro
        final_path = f"{output_dir}/protoforge_final"
        trainer.save_model(final_path)  # Salva i pesi del modello
        self.tokenizer.save_pretrained(final_path)  # Salva il tokenizer

        # FASE 6.8: Salvataggio configurazione training
        # Crea un file JSON con tutti i parametri usati per riproducibilità
        config = {
            'base_model': self.base_model_name,
            'epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'timestamp': datetime.now().isoformat()
        }
        with open(f"{final_path}/training_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # FASE 6.9: Logging di completamento
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETATO")
        logger.info("="*70)
        logger.info(f"Modello salvato in: {final_path}")

        return trainer

    def generate_proto(
        self, 
        nl_description: str,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.95
    ) -> str:
        """
        FASE 7: Generazione di codice .proto da linguaggio naturale.
        
        Questo metodo usa il modello allenato per generare codice Protocol Buffers
        a partire da una descrizione in linguaggio naturale.
        
        Args:
            nl_description: Descrizione in linguaggio naturale
            max_length: Lunghezza massima del codice generato
            num_beams: Numero di beam per beam search
            temperature: Temperatura per campionamento (più alta = più creatività)
            do_sample: Se usare campionamento o greedy decoding
            top_p: Soglia per nucleus sampling
            
        Returns:
            str: Codice .proto generato
        """
        # FASE 7.1: Imposta modalità evaluation
        self.model.eval()  # Disabilita dropout e altri layer di training

        # FASE 7.2: Prepara l'input per il modello
        input_text = f"generate proto: {nl_description}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",  # Ritorna tensori PyTorch
            max_length=128,  # Limita lunghezza input
            truncation=True  # Tronca se troppo lungo
        ).to(self.device)  # Sposta sulla GPU/CPU corretta

        # FASE 7.3: Generazione del codice
        with torch.no_grad():  # Disabilita calcolo gradienti per efficienza
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,  # Lunghezza massima output
                num_beams=num_beams,  # Beam search per qualità migliore
                temperature=temperature,  # Controlla casualità
                do_sample=do_sample,  # Usa campionamento
                top_p=top_p,  # Nucleus sampling
                early_stopping=True,  # Ferma quando completa
                no_repeat_ngram_size=2,  # Evita ripetizioni
                pad_token_id=self.tokenizer.pad_token_id,  # Token padding
                eos_token_id=self.tokenizer.eos_token_id,  # Token fine sequenza
            )

        # FASE 7.4: Decodifica e ritorno del risultato
        proto_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return proto_code

    def validate_proto(self, proto_code: str) -> Tuple[bool, Optional[str]]:
        """
        FASE 8: Validazione sintattica del codice .proto generato.
        
        Questo metodo verifica se il codice .proto generato è sintatticamente
        corretto usando il compilatore Protocol Buffers (protoc).
        
        Args:
            proto_code: Codice .proto da validare
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
                - True, None: Codice valido
                - False, error: Codice invalido con messaggio di errore
                - None, error: protoc non installato
        """
        import tempfile
        import subprocess

        # FASE 8.1: Crea file temporaneo con il codice .proto
        with tempfile.NamedTemporaryFile(mode='w', suffix='.proto', delete=False, dir='/tmp') as f:
            f.write(proto_code)
            temp_path = f.name

        try:
            # FASE 8.2: Esegui protoc per validare il codice
            result = subprocess.run(
                ['protoc', f'--proto_path=/tmp', '--python_out=/tmp', temp_path],
                capture_output=True,  # Cattura stdout/stderr
                text=True,  # Output come stringa
                timeout=10  # Timeout di 10 secondi
            )

            # FASE 8.3: Pulisci file temporaneo
            os.unlink(temp_path)

            # FASE 8.4: Analisi del risultato
            if result.returncode == 0:
                return True, None  # Codice valido
            else:
                return False, result.stderr  # Codice invalido

        except FileNotFoundError:
            return None, "protoc non trovato (installa: apt-get install protobuf-compiler)"
        except subprocess.TimeoutExpired:
            return False, "Timeout validazione"

    def evaluate_on_test(self, test_data: List[Dict], num_samples: int = None) -> Dict:
        """
        FASE 9: Valutazione del modello su test set.
        
        Questo metodo testa il modello su dati non visti per misurare
        la qualità del codice .proto generato.
        
        Args:
            test_data: Lista di esempi di test
            num_samples: Numero di campioni da valutare (None = tutti)
            
        Returns:
            Dict: Dizionario con metriche di valutazione
        """
        # FASE 9.1: Limita il numero di campioni se richiesto
        if num_samples:
            test_data = test_data[:num_samples]

        logger.info(f"\nValutazione su {len(test_data)} campioni...")

        # FASE 9.2: Inizializza struttura risultati
        results = {
            'total': len(test_data),
            'valid_syntax': 0,  # Codice sintatticamente valido
            'invalid_syntax': 0,  # Codice sintatticamente invalido
            'errors': []  # Lista di errori per analisi
        }

        # FASE 9.3: Valuta ogni campione
        for item in test_data:
            # Genera codice .proto dalla descrizione
            generated = self.generate_proto(item['natural_language'])
            # Valida la sintassi del codice generato
            is_valid, error = self.validate_proto(generated)

            # Aggiorna contatori
            if is_valid:
                results['valid_syntax'] += 1
            elif is_valid is False:
                results['invalid_syntax'] += 1
                results['errors'].append({
                    'id': item.get('id', 'unknown'),
                    'error': error,
                    'generated': generated[:200]  # Primi 200 caratteri
                })

        # FASE 9.4: Calcola accuratezza
        results['accuracy'] = results['valid_syntax'] / results['total'] * 100

        # FASE 9.5: Logga risultati
        logger.info(f"\nRisultati:")
        logger.info(f"  Validi:   {results['valid_syntax']}/{results['total']} ({results['accuracy']:.1f}%)")
        logger.info(f"  Invalidi: {results['invalid_syntax']}/{results['total']}")

        return results


def main():
    """
    FASE 10: Funzione principale - Punto di ingresso del programma.
    
    Questa funzione gestisce:
    1. Parsing degli argomenti da linea di comando
    2. Inizializzazione del modello
    3. Caricamento dei dati
    4. Esecuzione del training
    5. Valutazione post-training
    6. Demo di generazione
    """
    # FASE 10.1: Setup parser argomenti linea di comando
    parser = argparse.ArgumentParser(
        description="ProtoForge Training - Neural Proto Compiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Mostra valori di default
    )

    # Argomenti per i dati
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path al file CSV con dati training")
    parser.add_argument("--split-column", type=str, default="split",
                        help="Nome colonna per split train/test")

    # Argomenti per il modello
    parser.add_argument("--base-model", type=str, default="t5-small",
                        help="Modello base HuggingFace")
    parser.add_argument("--use-lora", action="store_true", default=True,
                        help="Usa LoRA per fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="Rank LoRA")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="Alpha LoRA")

    # Argomenti per il training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Numero epoche")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--early-stopping", type=int, default=3,
                        help="Patience early stopping")

    # Argomenti per output
    parser.add_argument("--output-dir", type=str, default="./protoforge_output",
                        help="Directory output")
    parser.add_argument("--eval-after", action="store_true",
                        help="Esegui valutazione post-training")
    parser.add_argument("--test-samples", type=int, default=100,
                        help="Numero campioni per valutazione")

    # FASE 10.2: Parsing degli argomenti
    args = parser.parse_args()

    # FASE 10.3: Header informativo
    print("\n" + "="*70)
    print("   PROTOFORGE v2.0 - Neural Proto Compiler")
    print("   Training con supporto CSV")
    print("="*70 + "\n")

    # FASE 10.4: Inizializzazione del modello ProtoForge
    forge = ProtoForge(
        base_model=args.base_model,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )

    # FASE 10.5: Caricamento dei dati dal CSV
    train_data, test_data = forge.load_data_from_csv(args.csv_path, args.split_column)

    # FASE 10.6: Esecuzione del training
    trainer = forge.train(
        train_data=train_data,
        val_data=test_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        early_stopping_patience=args.early_stopping
    )

    # FASE 10.7: Valutazione post-training (se richiesto)
    if args.eval_after and test_data:
        print("\n" + "-"*70)
        print("VALUTAZIONE POST-TRAINING")
        print("-"*70)

        results = forge.evaluate_on_test(test_data, args.test_samples)

        # Salva risultati valutazione
        results_path = f"{args.output_dir}/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRisultati salvati in: {results_path}")

    # FASE 10.8: Demo di generazione codice
    print("\n" + "-"*70)
    print("DEMO INFERENZA")
    print("-"*70)

    demo_queries = [
        "Gestisci utenti con email e login",
        "Sistema ordini con prodotti e quantita",
        "API autenticazione con token JWT",
        "Servizio pagamenti con carta di credito"
    ]

    for query in demo_queries:
        print(f"\n📝 Input: {query}")
        proto = forge.generate_proto(query)
        print(f"⚙️  Output ({len(proto)} char):")
        print(proto[:300] + "..." if len(proto) > 300 else proto)

        # Validazione del codice generato
        is_valid, error = forge.validate_proto(proto)
        status = "✅ Valido" if is_valid else ("❌ Invalido" if is_valid is False else "⚠️  No protoc")
        print(f"   {status}")

    # FASE 10.9: Messaggio di completamento
    print("\n" + "="*70)
    print("COMPLETATO!")
    print(f"Modello salvato in: {args.output_dir}/protoforge_final")
    print("="*70)


if __name__ == "__main__":
    main()
