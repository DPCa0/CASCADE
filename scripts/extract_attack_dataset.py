#!/usr/bin/env python3
"""
Extract High-Quality Attack Dataset from AgentDojo Pre-computed Results

This script processes all pre-computed attack execution results from AgentDojo's
runs directory and creates a comprehensive attack dataset containing only 
successful attacks with complete execution traces.

Features:
- Extracts from ALL available models (GPT-4o, Claude, Gemini, etc.)
- Filters out failed attacks (only includes successful security breaches)
- Removes duplicates based on attack content and execution patterns
- Includes complete message traces with tool calls and responses
- Preserves all metadata (model, attack type, suite, etc.)
- Real-time progress tracking and statistics
"""

import sys
import json
import os
import logging
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional, Tuple
import time
from datetime import datetime

# Add agentdojo src to path for any utility functions we might need
agentdojo_src = Path(__file__).parent.parent / "agentdojo" / "src"
sys.path.insert(0, str(agentdojo_src))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttackDatasetExtractor:
    """Extract high-quality attack data from AgentDojo pre-computed results"""
    
    def __init__(self, agentdojo_runs_path: str, output_path: str = None):
        self.runs_path = Path(agentdojo_runs_path)
        self.output_path = output_path
        self.extracted_attacks = []
        self.duplicate_hashes = set()
        self.stats = {
            'models_processed': 0,
            'total_files_found': 0,
            'successful_attacks': 0,
            'failed_attacks': 0,
            'duplicates_removed': 0,
            'parsing_errors': 0,
            'final_dataset_size': 0
        }
        
        if not self.runs_path.exists():
            raise ValueError(f"AgentDojo runs path does not exist: {self.runs_path}")
        
        logger.info(f"Initialized AttackDatasetExtractor")
        logger.info(f"  Runs path: {self.runs_path}")
        logger.info(f"  Output path: {output_path}")
    
    def discover_models(self) -> List[str]:
        """Discover all available models with attack data"""
        models = []
        for item in self.runs_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if this model directory has attack data
                has_attack_data = False
                for suite_dir in item.iterdir():
                    if suite_dir.is_dir():
                        for task_dir in suite_dir.iterdir():
                            if task_dir.is_dir() and task_dir.name.startswith('user_task_'):
                                # Check for attack subdirectories
                                attack_dirs = [d for d in task_dir.iterdir() 
                                             if d.is_dir() and d.name != 'none']
                                if attack_dirs:
                                    has_attack_data = True
                                    break
                        if has_attack_data:
                            break
                
                if has_attack_data:
                    models.append(item.name)
        
        models.sort()
        logger.info(f"Discovered {len(models)} models with attack data:")
        for model in models:
            logger.info(f"  - {model}")
        
        return models
    
    def generate_attack_hash(self, attack_data: Dict) -> str:
        """Generate a hash to identify duplicate attacks"""
        # Create hash from key components that make an attack unique
        hash_components = {
            'suite_name': attack_data.get('suite_name', ''),
            'user_task_id': attack_data.get('user_task_id', ''),
            'injection_task_id': attack_data.get('injection_task_id', ''),
            'attack_type': attack_data.get('attack_type', ''),
            'injections': str(attack_data.get('injections', {})),
            # Include the actual attack execution result
            'final_message_content': '',
            'tool_calls_signature': ''
        }
        
        # Extract execution signature from messages
        messages = attack_data.get('messages', [])
        if messages:
            # Get the final assistant message
            for msg in reversed(messages):
                if msg.get('role') == 'assistant' and msg.get('content'):
                    hash_components['final_message_content'] = msg['content'][:200]
                    break
            
            # Get tool call signature
            tool_calls = []
            for msg in messages:
                if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                    for call in msg['tool_calls']:
                        func_name = call.get('function', '')
                        args_str = str(call.get('args', {}))
                        tool_calls.append(f"{func_name}({args_str})")
            hash_components['tool_calls_signature'] = '|'.join(tool_calls)
        
        # Create hash
        hash_string = json.dumps(hash_components, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def extract_tool_parameters(self, messages: List[Dict]) -> List[Dict]:
        """Extract detailed tool parameters from message trace"""
        tool_parameters = []
        
        for turn_idx, message in enumerate(messages):
            if message.get('role') == 'assistant' and message.get('tool_calls'):
                for call_idx, tool_call in enumerate(message['tool_calls']):
                    try:
                        tool_parameters.append({
                            'function': tool_call.get('function', ''),
                            'args': tool_call.get('args', {}),
                            'turn': turn_idx,
                            'call_id': tool_call.get('id', f'call_{turn_idx}_{call_idx}'),
                            'call_index': call_idx
                        })
                    except Exception as e:
                        logger.debug(f"Failed to parse tool call: {e}")
                        continue
        
        return tool_parameters
    
    def process_attack_file(self, file_path: Path, model_name: str) -> Optional[Dict]:
        """Process a single attack execution file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip if this is not an attack (benign execution)
            if file_path.parent.name == 'none':
                return None
            
            # Skip failed attacks - we only want successful security breaches
            security_success = data.get('security', False)
            if not security_success:
                self.stats['failed_attacks'] += 1
                return None
            
            # Skip if there was an execution error
            if data.get('error') is not None:
                self.stats['parsing_errors'] += 1
                return None
            
            # Skip if messages are missing or incomplete
            messages = data.get('messages', [])
            if not messages or len(messages) < 2:
                self.stats['parsing_errors'] += 1
                return None
            
            # Extract comprehensive attack information
            attack_record = {
                # Core identifiers
                'is_attack': True,
                'model_name': model_name,
                'suite_name': data.get('suite_name', ''),
                'user_task_id': data.get('user_task_id', ''),
                'injection_task_id': data.get('injection_task_id', ''),
                'attack_type': data.get('attack_type', ''),
                
                # Attack success indicators
                'task_successful': data.get('utility', False),  # Did the user task complete?
                'security_successful': data.get('security', False),  # Did the attack succeed?
                'attack_successful': True,  # We only include successful attacks
                
                # Injection content
                'injections': data.get('injections', {}),
                'injection_content': str(data.get('injections', {})),
                
                # Execution metadata
                'execution_duration': data.get('duration', 0.0),
                'message_count': len(messages),
                'error_count': 1 if data.get('error') else 0,
                'execution_success_rate': 1.0 if data.get('security', False) else 0.0,
                
                # Tool usage analysis
                'tool_parameters': self.extract_tool_parameters(messages),
                'tool_call_count': sum(1 for msg in messages 
                                      if msg.get('role') == 'assistant' and msg.get('tool_calls')),
                
                # Message trace (complete conversation)
                'messages': messages,
                
                # Generation metadata
                'generated_method': f'agentdojo_precomputed_{model_name}',
                'generation_timestamp': int(time.time()),
                'source_file': str(file_path.relative_to(self.runs_path))
            }
            
            # Calculate additional metrics
            tool_functions = set()
            for tool_param in attack_record['tool_parameters']:
                tool_functions.add(tool_param['function'])
            attack_record['unique_tools_used'] = list(tool_functions)
            attack_record['unique_tool_count'] = len(tool_functions)
            
            self.stats['successful_attacks'] += 1
            return attack_record
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            self.stats['parsing_errors'] += 1
            return None
    
    def extract_attacks_from_model(self, model_name: str) -> List[Dict]:
        """Extract all successful attacks for a specific model"""
        logger.info(f"  Processing model: {model_name}")
        model_path = self.runs_path / model_name
        attacks = []
        files_processed = 0
        
        if not model_path.exists():
            logger.warning(f"  Model path not found: {model_path}")
            return attacks
        
        # Walk through all suite/task/attack combinations
        for suite_dir in model_path.iterdir():
            if not suite_dir.is_dir():
                continue
                
            suite_name = suite_dir.name
            logger.info(f"    Processing suite: {suite_name}")
            
            # Process user tasks and injection tasks
            for task_dir in suite_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                task_name = task_dir.name
                
                # Process attack types in this task
                for attack_dir in task_dir.iterdir():
                    if not attack_dir.is_dir() or attack_dir.name == 'none':
                        continue
                    
                    attack_type = attack_dir.name
                    
                    # Process all attack execution files
                    for attack_file in attack_dir.iterdir():
                        if attack_file.suffix == '.json':
                            files_processed += 1
                            self.stats['total_files_found'] += 1
                            
                            attack_data = self.process_attack_file(attack_file, model_name)
                            if attack_data:
                                attacks.append(attack_data)
        
        logger.info(f"    Processed {files_processed} files, extracted {len(attacks)} successful attacks")
        return attacks
    
    def remove_duplicates(self, attacks: List[Dict]) -> List[Dict]:
        """Remove duplicate attacks based on content hash"""
        logger.info("🔍 Removing duplicates...")
        unique_attacks = []
        
        for attack in attacks:
            attack_hash = self.generate_attack_hash(attack)
            
            if attack_hash not in self.duplicate_hashes:
                self.duplicate_hashes.add(attack_hash)
                attack['attack_hash'] = attack_hash
                unique_attacks.append(attack)
            else:
                self.stats['duplicates_removed'] += 1
        
        logger.info(f"  Removed {self.stats['duplicates_removed']} duplicates")
        logger.info(f"  Unique attacks: {len(unique_attacks)}")
        
        return unique_attacks
    
    def extract_all_attacks(self) -> List[Dict]:
        """Extract successful attacks from all available models"""
        logger.info("🚀 Starting attack dataset extraction...")
        
        # Discover all models
        models = self.discover_models()
        self.stats['models_processed'] = len(models)
        
        all_attacks = []
        
        # Process each model
        for model_name in models:
            model_attacks = self.extract_attacks_from_model(model_name)
            all_attacks.extend(model_attacks)
            
            logger.info(f"  Model {model_name}: {len(model_attacks)} successful attacks")
        
        logger.info(f"📊 Total attacks before deduplication: {len(all_attacks)}")
        
        # Remove duplicates
        unique_attacks = self.remove_duplicates(all_attacks)
        
        self.stats['final_dataset_size'] = len(unique_attacks)
        return unique_attacks
    
    def analyze_dataset(self, attacks: List[Dict]) -> Dict:
        """Analyze the extracted attack dataset"""
        logger.info("📊 Analyzing extracted dataset...")
        
        # Basic statistics
        total_attacks = len(attacks)
        
        # Distribution by model
        model_dist = Counter(attack['model_name'] for attack in attacks)
        
        # Distribution by suite
        suite_dist = Counter(attack['suite_name'] for attack in attacks)
        
        # Distribution by attack type
        attack_type_dist = Counter(attack['attack_type'] for attack in attacks)
        
        # Tool usage statistics
        all_tools = []
        for attack in attacks:
            all_tools.extend(attack['unique_tools_used'])
        tool_usage = Counter(all_tools)
        
        # Success rate analysis
        task_success_rate = sum(1 for attack in attacks if attack['task_successful']) / total_attacks
        security_success_rate = sum(1 for attack in attacks if attack['security_successful']) / total_attacks
        
        # Message complexity
        message_counts = [attack['message_count'] for attack in attacks]
        avg_messages = sum(message_counts) / len(message_counts) if message_counts else 0
        
        # Tool call complexity
        tool_call_counts = [attack['tool_call_count'] for attack in attacks]
        avg_tool_calls = sum(tool_call_counts) / len(tool_call_counts) if tool_call_counts else 0
        
        analysis = {
            'total_attacks': total_attacks,
            'model_distribution': dict(model_dist),
            'suite_distribution': dict(suite_dist), 
            'attack_type_distribution': dict(attack_type_dist),
            'tool_usage_distribution': tool_usage.most_common(20),
            'task_success_rate': task_success_rate,
            'security_success_rate': security_success_rate,
            'avg_messages_per_attack': avg_messages,
            'avg_tool_calls_per_attack': avg_tool_calls,
            'extraction_stats': self.stats
        }
        
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """Print detailed dataset analysis"""
        print(f"\n🎯 ATTACK DATASET EXTRACTION ANALYSIS")
        print(f"=" * 60)
        
        print(f"Total Successful Attacks: {analysis['total_attacks']:,}")
        print(f"Models Processed: {analysis['extraction_stats']['models_processed']}")
        print(f"Files Scanned: {analysis['extraction_stats']['total_files_found']:,}")
        
        print(f"\nSuccess Rates:")
        print(f"  Attack Success Rate: {analysis['security_success_rate']:.1%} (filtered to 100%)")
        print(f"  Task Completion Rate: {analysis['task_success_rate']:.1%}")
        
        print(f"\nExecution Complexity:")
        print(f"  Avg Messages per Attack: {analysis['avg_messages_per_attack']:.1f}")
        print(f"  Avg Tool Calls per Attack: {analysis['avg_tool_calls_per_attack']:.1f}")
        
        print(f"\nModel Distribution:")
        for model, count in sorted(analysis['model_distribution'].items(), 
                                 key=lambda x: x[1], reverse=True):
            percentage = count / analysis['total_attacks'] * 100
            print(f"  {model:<40}: {count:5,} ({percentage:4.1f}%)")
        
        print(f"\nSuite Distribution:")
        for suite, count in sorted(analysis['suite_distribution'].items()):
            percentage = count / analysis['total_attacks'] * 100
            print(f"  {suite:<15}: {count:5,} ({percentage:4.1f}%)")
        
        print(f"\nTop Attack Types:")
        for attack_type, count in sorted(analysis['attack_type_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
            percentage = count / analysis['total_attacks'] * 100
            print(f"  {attack_type:<35}: {count:4,} ({percentage:4.1f}%)")
        
        print(f"\nTop Tool Usage:")
        tool_items = analysis['tool_usage_distribution']
        if tool_items:
            # Debug: Check the structure of tool_items
            logger.debug(f"Tool items type: {type(tool_items)}, first few items: {tool_items[:3] if isinstance(tool_items, list) else 'not a list'}")
            
            try:
                if isinstance(tool_items, dict):
                    # If it's a dict, convert to list of tuples
                    tool_list = list(tool_items.items())
                else:
                    # It should be a list of tuples from most_common()
                    tool_list = tool_items
                
                total_tool_uses = sum(count for tool, count in tool_list)
                for tool, count in tool_list:
                    percentage = count / total_tool_uses * 100
                    print(f"  {tool:<25}: {count:4,} ({percentage:4.1f}%)")
                    
            except Exception as e:
                logger.error(f"Error processing tool usage data: {e}")
                print(f"  Error displaying tool usage statistics: {e}")
        else:
            print("  No tool usage data available")
        
        print(f"\nExtraction Statistics:")
        stats = analysis['extraction_stats']
        print(f"  Successful Attacks: {stats['successful_attacks']:,}")
        print(f"  Failed Attacks: {stats['failed_attacks']:,}")
        print(f"  Duplicates Removed: {stats['duplicates_removed']:,}")
        print(f"  Parsing Errors: {stats['parsing_errors']:,}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract High-Quality Attack Dataset from AgentDojo')
    parser.add_argument('--runs-path', 
                       default='/Users/yummy/IPI_Defense/agentdojo/runs',
                       help='Path to AgentDojo runs directory')
    parser.add_argument('--output-dataset',
                       default='/Users/yummy/IPI_Defense/dataset/attack_dataset.json',
                       help='Output path for attack dataset')
    parser.add_argument('--models', nargs='*',
                       help='Specific models to process (default: all available)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show extraction plan without processing')
    parser.add_argument('--max-attacks', type=int,
                       help='Maximum number of attacks to extract (for testing)')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print(f"🔍 DRY RUN - Attack Dataset Extraction Plan")
        print(f"=" * 60)
        print(f"Runs path: {args.runs_path}")
        print(f"Output: {args.output_dataset}")
        print(f"Models: {'All available' if not args.models else ', '.join(args.models)}")
        print(f"Max attacks: {'Unlimited' if not args.max_attacks else f'{args.max_attacks:,}'}")
        
        print(f"\n🎯 Extraction Method:")
        print(f"   ✅ Process all pre-computed AgentDojo attack executions")
        print(f"   ✅ Filter for successful attacks only (security=True)")
        print(f"   ✅ Remove duplicates based on content hash")
        print(f"   ✅ Extract complete message traces and tool calls")
        print(f"   ✅ Include comprehensive metadata")
        
        print(f"\n📊 Expected Benefits:")
        print(f"   • Zero API costs (pre-computed data)")
        print(f"   • High-quality successful attack examples")
        print(f"   • Multiple state-of-the-art models")
        print(f"   • Diverse attack types and scenarios")
        print(f"   • Complete execution traces for Layer 2/3 training")
        
        return 0
    
    try:
        # Initialize extractor
        extractor = AttackDatasetExtractor(args.runs_path, args.output_dataset)
        
        # Extract all attacks
        attacks = extractor.extract_all_attacks()
        
        # Filter specific models if requested
        if args.models:
            attacks = [attack for attack in attacks if attack['model_name'] in args.models]
            logger.info(f"Filtered to {len(attacks)} attacks from specified models")
        
        # Limit dataset size if requested
        if args.max_attacks and len(attacks) > args.max_attacks:
            import random
            attacks = random.sample(attacks, args.max_attacks)
            logger.info(f"Randomly sampled {len(attacks)} attacks")
        
        # Analyze dataset
        analysis = extractor.analyze_dataset(attacks)
        extractor.print_analysis(analysis)
        
        # Save dataset
        output_path = Path(args.output_dataset)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(attacks, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ ATTACK DATASET EXTRACTION COMPLETE!")
        print(f"   Saved to: {output_path}")
        print(f"   Total attacks: {len(attacks):,}")
        print(f"   Models: {len(analysis['model_distribution'])} different models")
        print(f"   Attack types: {len(analysis['attack_type_distribution'])} different types")
        print(f"   Quality: 100% successful attacks only")
        print(f"   Cost savings: Potentially $10,000+ in API fees")
        print(f"\n🚀 Ready for high-quality IPI defense training!")
        
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Attack dataset extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())