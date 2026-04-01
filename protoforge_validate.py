"""
╔══════════════════════════════════════════════════════════════════╗
║  PROTOFORGE VALIDATOR                                            ║
║  Valutazione completa modello addestrato                         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import csv
import tempfile
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
from tqdm import tqdm

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Metriche di validazione"""
    total_samples: int
    valid_syntax: int
    invalid_syntax: int
    exact_match: int
    partial_match: int  # Structura simile ma non identica
    avg_generation_time: float
    syntax_accuracy: float
    exact_match_rate: float
    
    def to_dict(self):
        return asdict(self)


class ProtoValidator:
    """
    Validatore completo per modelli ProtoForge
    """
    
    def __init__(self, model_path: str, base_model: str = "Salesforce/codet5-small"):
        """
        Args:
            model_path: path al modello addestrato (cartella con adapter)
            base_model: modello base usato per training
        """
        logger.info("="*70)
        logger.info("CARICAMENTO MODELLO")
        logger.info("="*70)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device: {self.device}")
        
        # Carica tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        # Carica modello base + adapter
        base = T5ForConditionalGeneration.from_pretrained(base_model)
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Modello caricato da: {model_path}")
        
        # Statistiche
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Parametri totali: {total:,}")
    
    def validate_syntax(self, proto_code: str) -> Tuple[bool, Optional[str]]:
        """
        Valida sintassi .proto usando protoc
        
        Returns:
            (is_valid, error_message)
            is_valid = None se protoc non è installato
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.proto', 
                                        delete=False, dir='/tmp') as f:
            f.write(proto_code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['protoc', f'--proto_path=/tmp', '--python_out=/tmp', temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            os.unlink(temp_path)
            
            if result.returncode == 0:
                return True, None
            else:
                # Estrai errore rilevante
                error = result.stderr.strip()
                return False, error
                
        except FileNotFoundError:
            logger.warning("protoc non trovato! Installa: sudo apt-get install protobuf-compiler")
            return None, "protoc non installato"
        except subprocess.TimeoutExpired:
            return False, "Timeout validazione"
    
    def generate(self, nl_description: str) -> Tuple[str, float]:
        """
        Genera proto e misura tempo
        
        Returns:
            (proto_code, generation_time_ms)
        """
        import time
        
        self.model.eval()
        inputs = self.tokenizer(
            f"generate proto: {nl_description}",
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).to(self.device)
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        gen_time = (time.time() - start) * 1000  # ms
        
        proto = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return proto, gen_time
    
    def calculate_similarity(self, generated: str, expected: str) -> float:
        """
        Calcola similarità strutturale tra proto generato e atteso
        Non è exact match, ma verifica se ha la struttura corretta
        """
        gen_lines = [l.strip() for l in generated.split('\n') if l.strip()]
        exp_lines = [l.strip() for l in expected.split('\n') if l.strip()]
        
        # Controlla presenza elementi chiave
        checks = {
            'syntax': 'syntax = "proto3";' in generated,
            'message': 'message ' in generated,
            'service': 'service ' in generated,
            'fields': sum(1 for l in gen_lines if '=' in l and l.strip()[0].islower())
        }
        
        # Score basato su checks
        score = sum(checks.values()) / 4.0
        return score
    
    def validate_dataset(self, test_data: List[Dict], 
                        output_json: str = None) -> ValidationMetrics:
        """
        Valuta il modello su un dataset di test
        
        Args:
            test_data: lista di dict con 'natural_language' e 'proto_code'
            output_json: path dove salvare risultati dettagliati
        
        Returns:
            ValidationMetrics con statistiche
        """
        logger.info("\n" + "="*70)
        logger.info("INIZIO VALIDAZIONE")
        logger.info(f"Campioni: {len(test_data)}")
        logger.info("="*70)
        
        results = []
        total_time = 0
        
        valid_count = 0
        invalid_count = 0
        exact_match = 0
        partial_match = 0
        
        for item in tqdm(test_data, desc="Validazione"):
            nl = item['natural_language']
            expected = item['proto_code']
            
            # Generazione
            generated, gen_time = self.generate(nl)
            total_time += gen_time
            
            # Validazione sintassi
            is_valid, error = self.validate_syntax(generated)
            
            # Match esatto
            is_exact = generated.strip() == expected.strip()
            
            # Similarità strutturale
            similarity = self.calculate_similarity(generated, expected)
            is_partial = similarity >= 0.75  # 75% struttura corretta
            
            # Contatori
            if is_valid:
                valid_count += 1
            elif is_valid is False:
                invalid_count += 1
            
            if is_exact:
                exact_match += 1
            if is_partial:
                partial_match += 1
            
            # Salva risultato
            results.append({
                'id': item.get('id', 'unknown'),
                'natural_language': nl,
                'expected': expected,
                'generated': generated,
                'generation_time_ms': gen_time,
                'syntax_valid': is_valid,
                'syntax_error': error,
                'exact_match': is_exact,
                'structural_similarity': similarity,
                'partial_match': is_partial
            })
        
        # Calcola metriche
        n = len(test_data)
        metrics = ValidationMetrics(
            total_samples=n,
            valid_syntax=valid_count,
            invalid_syntax=invalid_count,
            exact_match=exact_match,
            partial_match=partial_match,
            avg_generation_time=total_time / n,
            syntax_accuracy=(valid_count / n * 100) if n > 0 else 0,
            exact_match_rate=(exact_match / n * 100) if n > 0 else 0
        )
        
        # Salva risultati dettagliati
        if output_json:
            with open(output_json, 'w') as f:
                json.dump({
                    'metrics': metrics.to_dict(),
                    'samples': results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"\nRisultati salvati in: {output_json}")
        
        # Stampa riepilogo
        self._print_summary(metrics, results)
        
        return metrics
    
    def _print_summary(self, metrics: ValidationMetrics, results: List[Dict]):
        """Stampa riepilogo validazione"""
        logger.info("\n" + "="*70)
        logger.info("RIEPILOGO VALIDAZIONE")
        logger.info("="*70)
        
        logger.info(f"\n📊 Metriche:")
        logger.info(f"  Totale campioni:     {metrics.total_samples}")
        logger.info(f"  Sintassi valida:     {metrics.valid_syntax} ({metrics.syntax_accuracy:.1f}%)")
        logger.info(f"  Sintassi invalida:   {metrics.invalid_syntax}")
        logger.info(f"  Exact match:         {metrics.exact_match} ({metrics.exact_match_rate:.1f}%)")
        logger.info(f"  Partial match (75%+): {metrics.partial_match}")
        logger.info(f"  Tempo medio/gen:     {metrics.avg_generation_time:.1f} ms")
        
        # Mostra errori comuni
        errors = [r for r in results if r['syntax_error'] and r['syntax_error'] != "protoc non installato"]
        if errors:
            logger.info(f"\n❌ Errori sintassi comuni (primi 3):")
            for i, err in enumerate(errors[:3], 1):
                logger.info(f"\n  Errore {i}:")
                logger.info(f"    Input: {err['natural_language'][:60]}...")
                logger.info(f"    Errore: {err['syntax_error'][:100]}...")
        
        # Esempi di successo
        successes = [r for r in results if r['syntax_valid']][:2]
        if successes:
            logger.info(f"\n✅ Esempi di successo:")
            for s in successes:
                logger.info(f"\n  Input: {s['natural_language']}")
                logger.info(f"  Output: {s['generated'][:150]}...")
        
        logger.info("\n" + "="*70)
    
    def interactive_test(self):
        """Modalità interattiva per testare query"""
        print("\n" + "="*70)
        print("MODALITA INTERATTIVA")
        print("Inserisci descrizioni in italiano (quit per uscire)")
        print("="*70)
        
        while True:
            try:
                query = input("\n📝 Input: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                proto, time_ms = self.generate(query)
                is_valid, error = self.validate_syntax(proto)
                
                print(f"\n⚙️  Output ({time_ms:.1f} ms):")
                print(proto)
                
                if is_valid:
                    print("\n✅ Sintassi valida")
                elif is_valid is False:
                    print(f"\n❌ Errore: {error}")
                else:
                    print("\n⚠️  protoc non installato - impossibile validare")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Errore: {e}")
        
        print("\nUscita...")


def load_csv_dataset(csv_path: str) -> List[Dict]:
    """Carica dataset da CSV"""
    df = pd.read_csv(csv_path)
    return df.to_dict('records')


def main():
    parser = argparse.ArgumentParser(
        description="ProtoForge Validator - Valutazione modello",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path al modello addestrato (cartella)")
    parser.add_argument("--test-csv", type=str, required=True,
                        help="CSV con dati di test")
    parser.add_argument("--output-json", type=str, default="validation_results.json",
                        help="Path output JSON con risultati")
    parser.add_argument("--base-model", type=str, default="Salesforce/codet5-small",
                        help="Modello base HuggingFace")
    parser.add_argument("--interactive", action="store_true",
                        help="Modalità interattiva dopo validazione")
    
    args = parser.parse_args()
    
    # Verifica file esistano
    if not os.path.exists(args.model_path):
        logger.error(f"Modello non trovato: {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.test_csv):
        logger.error(f"CSV non trovato: {args.test_csv}")
        sys.exit(1)
    
    # Inizializza validatore
    validator = ProtoValidator(args.model_path, args.base_model)
    
    # Carica dataset
    test_data = load_csv_dataset(args.test_csv)
    
    # Esegui validazione
    metrics = validator.validate_dataset(test_data, args.output_json)
    
    # Modalità interattiva opzionale
    if args.interactive:
        validator.interactive_test()
    
    # Exit code basato su risultati
    if metrics.syntax_accuracy >= 80:
        logger.info("\n🎉 Ottimo risultato! Modello pronto per produzione.")
        sys.exit(0)
    elif metrics.syntax_accuracy >= 60:
        logger.info("\n⚠️  Risultato accettabile. Considera più training data.")
        sys.exit(0)
    else:
        logger.info("\n❌ Risultato insufficiente. Rivedi training.")
        sys.exit(1)


if __name__ == "__main__":
    main()



