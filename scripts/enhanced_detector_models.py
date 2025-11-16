"""
Enhanced Unified Detector Models with Position-Aware Features and Attention

Key improvements:
1. Position-aware feature extraction for Layer 2
2. Attention-based pooling instead of mean pooling
3. Transition pattern encoding with positional context
4. Escalation detection features
"""

import torch
import torch.nn as nn
import numpy as np
import re
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import logging
import math

logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor:
    """Enhanced feature extractor with position-aware temporal features."""

    def __init__(self, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.layer1_scaler = StandardScaler()
        self.layer2_scaler = StandardScaler()

        # Layer 2 components
        self.function_vocab = {}
        self.transition_vocab = {}
        self.transition_counts = defaultdict(int)
        self.suspicious_transitions = set()

        # Track vocabulary for consistent encoding
        self.vocab_size = 0
        self.transition_vocab_size = 0

    def extract_layer1_features(self, function_name: str, parameter_value: str, suite_name: str) -> np.ndarray:
        """Extract Layer 1 comprehensive features (unchanged from original)."""
        features = []

        # 1. Semantic Features
        param_embedding = self.embedding_model.encode([parameter_value])[0]
        func_embedding = self.embedding_model.encode([function_name])[0]
        suite_embedding = self.embedding_model.encode([suite_name])[0]

        # Statistical moments of parameter embedding
        semantic_features = [
            np.mean(param_embedding), np.std(param_embedding), np.min(param_embedding), np.max(param_embedding),
            np.median(param_embedding), np.percentile(param_embedding, 25), np.percentile(param_embedding, 75),
            np.sum(param_embedding > 0), np.sum(param_embedding < 0), np.linalg.norm(param_embedding),
            np.linalg.norm(param_embedding, ord=1), np.var(param_embedding)
        ]
        features.extend(semantic_features)

        # 2. Text Statistics
        text_stats = [
            len(parameter_value), len(parameter_value.split()), len(set(parameter_value.lower())),
            parameter_value.count(' '), parameter_value.count('.'), parameter_value.count(','),
            parameter_value.count('!'), parameter_value.count('?'), parameter_value.count(':'),
            parameter_value.count(';')
        ]
        features.extend(text_stats)

        # 3. Linguistic Patterns
        linguistic_features = [
            sum(1 for c in parameter_value if c.isupper()) / max(len(parameter_value), 1),
            sum(1 for c in parameter_value if c.islower()) / max(len(parameter_value), 1),
            sum(1 for c in parameter_value if c.isdigit()) / max(len(parameter_value), 1),
            sum(1 for c in parameter_value if c.isalnum()) / max(len(parameter_value), 1),
        ]
        features.extend(linguistic_features)

        # 4. General Text Patterns
        pattern_features = [
            len(re.findall(r'[A-Z]{3,}', parameter_value)),
            len(re.findall(r'!+', parameter_value)),
            len(re.findall(r'\b[A-Z]+\b', parameter_value)),
            parameter_value.count('*') + parameter_value.count('#'),
        ]
        features.extend(pattern_features)

        # 5. Function and suite context
        context_features = [
            np.mean(func_embedding), np.std(func_embedding),
            np.mean(suite_embedding), np.std(suite_embedding),
            len(function_name), function_name.count('_')
        ]
        features.extend(context_features)

        return np.array(features, dtype=np.float32)

    def build_layer2_vocabulary(self, sequences: List[List[Dict]]):
        """Build vocabulary for Layer 2 sequence encoding including transitions."""
        functions = set()
        transitions = defaultdict(int)

        for sequence in sequences:
            # Collect functions
            for item in sequence:
                if isinstance(item, dict) and 'function_name' in item:
                    functions.add(item['function_name'])
                elif isinstance(item, str):
                    functions.add(item)

            # Collect transitions
            for i in range(len(sequence) - 1):
                curr_func = sequence[i].get('function_name', '') if isinstance(sequence[i], dict) else str(sequence[i])
                next_func = sequence[i+1].get('function_name', '') if isinstance(sequence[i+1], dict) else str(sequence[i+1])

                if curr_func and next_func:
                    transition = f"{curr_func}→{next_func}"
                    transitions[transition] += 1

        # Build function vocabulary
        self.function_vocab = {func: idx + 1 for idx, func in enumerate(sorted(functions))}
        self.function_vocab['<PAD>'] = 0
        self.vocab_size = len(self.function_vocab)

        # Build transition vocabulary (top N most common)
        sorted_transitions = sorted(transitions.items(), key=lambda x: -x[1])
        self.transition_vocab = {trans: idx + 1 for idx, (trans, _) in enumerate(sorted_transitions[:500])}
        self.transition_vocab['<UNK>'] = 0
        self.transition_vocab_size = len(self.transition_vocab)

        # Store transition counts for risk scoring
        self.transition_counts = transitions

        logger.info(f"Built vocabulary: {self.vocab_size} functions, {self.transition_vocab_size} transitions")

    def analyze_suspicious_patterns(self, attack_sequences: List[List[Dict]], benign_sequences: List[List[Dict]]):
        """Analyze attack vs benign sequences to identify suspicious transitions."""

        attack_transitions = defaultdict(int)
        benign_transitions = defaultdict(int)

        for sequence in attack_sequences:
            for i in range(len(sequence) - 1):
                curr = sequence[i].get('function_name', '') if isinstance(sequence[i], dict) else str(sequence[i])
                next_ = sequence[i+1].get('function_name', '') if isinstance(sequence[i+1], dict) else str(sequence[i+1])
                if curr and next_:
                    attack_transitions[f"{curr}→{next_}"] += 1

        for sequence in benign_sequences:
            for i in range(len(sequence) - 1):
                curr = sequence[i].get('function_name', '') if isinstance(sequence[i], dict) else str(sequence[i])
                next_ = sequence[i+1].get('function_name', '') if isinstance(sequence[i+1], dict) else str(sequence[i+1])
                if curr and next_:
                    benign_transitions[f"{curr}→{next_}"] += 1

        # Identify transitions that appear significantly more in attacks
        self.suspicious_transitions = set()
        for transition, attack_count in attack_transitions.items():
            benign_count = benign_transitions.get(transition, 0)
            total_attack = sum(attack_transitions.values())
            total_benign = sum(benign_transitions.values())

            attack_rate = attack_count / max(total_attack, 1)
            benign_rate = benign_count / max(total_benign, 1)

            # Flag as suspicious if appears 3x more often in attacks
            if attack_rate > 3 * benign_rate and attack_count >= 5:
                self.suspicious_transitions.add(transition)

        logger.info(f"Identified {len(self.suspicious_transitions)} suspicious transitions")

    def extract_layer2_features(self, sequence: List[Dict], max_seq_length: int = 20) -> np.ndarray:
        """
        Extract enhanced Layer 2 features with position-aware information.

        Returns:
            Array with structure: [func_sequence (20) + position_features (20) +
                                  transition_features (57) + seq_stats (10)]
        """
        if not sequence:
            return np.zeros(max_seq_length + max_seq_length + 57 + 10, dtype=np.float32)

        # 1. Function sequence (as before)
        func_sequence = []
        for item in sequence:
            if isinstance(item, dict) and 'function_name' in item:
                func_name = item['function_name']
            else:
                func_name = str(item)
            idx = self.function_vocab.get(func_name, 0)
            if idx >= self.vocab_size:
                idx = 0
            func_sequence.append(idx)

        # Pad or truncate
        actual_length = len(func_sequence)
        if len(func_sequence) > max_seq_length:
            func_sequence = func_sequence[:max_seq_length]
        else:
            func_sequence.extend([0] * (max_seq_length - len(func_sequence)))

        func_sequence = [min(idx, self.vocab_size - 1) for idx in func_sequence]

        # 2. NEW: Position encoding (normalized position in sequence)
        position_features = []
        for i in range(max_seq_length):
            if i < actual_length:
                # Normalized position (0 to 1)
                pos = i / max(actual_length - 1, 1)
                position_features.append(pos)
            else:
                position_features.append(0.0)  # Padding

        # 3. NEW: Transition features with position-aware risk scoring
        transition_features = []

        for i in range(min(len(sequence) - 1, max_seq_length - 1)):
            curr_func = sequence[i].get('function_name', '') if isinstance(sequence[i], dict) else str(sequence[i])
            next_func = sequence[i+1].get('function_name', '') if isinstance(sequence[i+1], dict) else str(sequence[i+1])

            transition = f"{curr_func}→{next_func}"

            # Transition ID
            trans_id = self.transition_vocab.get(transition, 0)

            # Position weight (transitions early vs late in sequence)
            position_weight = i / max(actual_length - 1, 1)

            # Suspicious transition indicator
            is_suspicious = 1.0 if transition in self.suspicious_transitions else 0.0

            transition_features.extend([trans_id, position_weight, is_suspicious])

        # Pad transition features
        while len(transition_features) < (max_seq_length - 1) * 3:
            transition_features.extend([0, 0, 0])

        # 4. Enhanced sequence statistics
        seq_stats = [
            actual_length,  # Actual length
            len(set(func_sequence)) / max(len(func_sequence), 1),  # Unique ratio
            func_sequence.count(0) / max_seq_length,  # Padding ratio
            np.std(func_sequence) if len(func_sequence) > 1 else 0,  # Sequence variance
            len([i for i in range(1, len(func_sequence)) if func_sequence[i] != func_sequence[i-1]]),  # Transitions

            # NEW: Escalation indicators
            self._detect_escalation_pattern(sequence),
            self._calculate_diversity_entropy(func_sequence),
            self._calculate_burst_score(sequence),

            # Position-aware statistics
            np.mean([func_sequence[i] for i in range(min(5, actual_length))]) if actual_length > 0 else 0,  # Early functions
            np.mean([func_sequence[i] for i in range(max(0, actual_length-5), actual_length)]) if actual_length > 0 else 0,  # Late functions
        ]

        # Combine all features
        all_features = func_sequence + position_features + transition_features + seq_stats

        return np.array(all_features, dtype=np.float32)

    def _detect_escalation_pattern(self, sequence: List[Dict]) -> float:
        """
        Detect escalation patterns (e.g., read → modify → delete).
        Returns score 0-1 indicating escalation severity.
        """
        if len(sequence) < 2:
            return 0.0

        # Define privilege levels
        privilege_map = {
            'read': 1, 'get': 1, 'list': 1, 'search': 1, 'view': 1,
            'modify': 2, 'update': 2, 'edit': 2, 'change': 2,
            'delete': 3, 'remove': 3, 'destroy': 3,
            'transfer': 3, 'send': 3, 'execute': 3
        }

        # Map functions to privilege levels
        levels = []
        for item in sequence:
            func = item.get('function_name', '').lower() if isinstance(item, dict) else str(item).lower()
            level = 0
            for keyword, priv in privilege_map.items():
                if keyword in func:
                    level = max(level, priv)
            levels.append(level)

        # Calculate escalation score
        escalations = 0
        max_escalation = 0

        for i in range(len(levels) - 1):
            if levels[i+1] > levels[i]:
                escalation = levels[i+1] - levels[i]
                escalations += escalation
                max_escalation = max(max_escalation, escalation)

        # Normalize
        score = min(escalations / (len(levels) * 2), 1.0)
        return score

    def _calculate_diversity_entropy(self, func_sequence: List[int]) -> float:
        """Calculate Shannon entropy of function distribution."""
        non_zero = [f for f in func_sequence if f > 0]
        if not non_zero:
            return 0.0

        # Count frequencies
        counts = {}
        for f in non_zero:
            counts[f] = counts.get(f, 0) + 1

        # Calculate entropy
        total = len(non_zero)
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p) if p > 0 else 0

        # Normalize by max entropy
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1
        return entropy / max(max_entropy, 0.001)

    def _calculate_burst_score(self, sequence: List[Dict]) -> float:
        """
        Detect sudden bursts of activity (e.g., many calls in short period).
        Returns score 0-1 indicating burst intensity.
        """
        if len(sequence) < 3:
            return 0.0

        # Simple heuristic: consecutive identical or similar functions
        consecutive_runs = []
        current_run = 1

        for i in range(len(sequence) - 1):
            curr_func = sequence[i].get('function_name', '') if isinstance(sequence[i], dict) else str(sequence[i])
            next_func = sequence[i+1].get('function_name', '') if isinstance(sequence[i+1], dict) else str(sequence[i+1])

            if curr_func == next_func:
                current_run += 1
            else:
                if current_run > 1:
                    consecutive_runs.append(current_run)
                current_run = 1

        if current_run > 1:
            consecutive_runs.append(current_run)

        # Score based on longest run
        max_run = max(consecutive_runs) if consecutive_runs else 0
        score = min(max_run / 10, 1.0)
        return score


class AttentionPooling(nn.Module):
    """Attention mechanism for pooling LSTM outputs."""

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, attention_dim),  # *2 for bidirectional
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, lstm_out, mask=None):
        """
        Args:
            lstm_out: [batch, seq_len, hidden_dim*2]
            mask: [batch, seq_len] - 1 for valid positions, 0 for padding

        Returns:
            attended: [batch, hidden_dim*2]
            attention_weights: [batch, seq_len, 1] (for interpretability)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_out)  # [batch, seq_len, 1]

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        # Softmax to get weights
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted sum
        attended = torch.sum(lstm_out * attention_weights, dim=1)

        return attended, attention_weights


class EnhancedIPIDetector(nn.Module):
    """
    Enhanced unified detector with:
    1. Position-aware Layer 2 features
    2. Attention-based pooling for LSTM outputs
    3. Multi-scale temporal features
    """

    def __init__(self, layer1_dim: int = 36, layer2_dim: int = 107,  # Updated dim
                 vocab_size: int = 100, embedding_dim: int = 32, hidden_dim: int = 64):
        super(EnhancedIPIDetector, self).__init__()

        # Layer 1 feature processor (unchanged)
        self.l1_processor = nn.Sequential(
            nn.Linear(layer1_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Layer 2: Enhanced sequence processor with position awareness
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Position encoding added to embeddings
        self.position_encoder = nn.Linear(1, embedding_dim)

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, batch_first=True,
                           bidirectional=True, num_layers=2, dropout=0.2)

        # NEW: Attention pooling instead of mean
        self.attention_pooling = AttentionPooling(hidden_dim, attention_dim=64)

        # Transition processor
        self.transition_embedding = nn.Embedding(500, 16, padding_idx=0)  # 500 transitions

        # Process enhanced Layer 2 features
        # Input: LSTM attended (128) + transition features (processed) + seq_stats (10)
        self.l2_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32 + 10, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(64, 48),  # 32 + 32
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Dropout(0.3),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 1)
        )

    def forward(self, layer1_features, layer2_sequences):
        """
        Forward pass with enhanced temporal processing.

        Args:
            layer1_features: [batch, layer1_dim]
            layer2_sequences: [batch, layer2_dim] where layer2_dim = 20 + 20 + 57 + 10
        """
        batch_size = layer1_features.size(0)

        # Process Layer 1 features
        l1_out = self.l1_processor(layer1_features)

        # Parse Layer 2 sequences
        # Structure: [func_seq (20), positions (20), transitions (57), stats (10)]
        func_seq = layer2_sequences[:, :20].long()  # Function IDs
        positions = layer2_sequences[:, 20:40]  # Position encodings
        transitions = layer2_sequences[:, 40:97]  # Transition features (19 * 3)
        seq_stats = layer2_sequences[:, 97:]  # Sequence statistics

        # Clamp function indices
        func_seq = torch.clamp(func_seq, 0, self.embedding.num_embeddings - 1)

        # Embed functions
        func_embedded = self.embedding(func_seq)  # [batch, 20, 32]

        # Encode positions
        pos_encoded = self.position_encoder(positions.unsqueeze(-1))  # [batch, 20, 32]

        # Combine function embeddings with position encodings
        combined_embedded = torch.cat([func_embedded, pos_encoded], dim=-1)  # [batch, 20, 64]

        # Create mask for padding (positions == 0 means padding)
        mask = (positions != 0).float()  # [batch, 20]

        # LSTM processing
        lstm_out, _ = self.lstm(combined_embedded)  # [batch, 20, 128]

        # NEW: Attention pooling instead of mean
        lstm_features, attention_weights = self.attention_pooling(lstm_out, mask)
        # lstm_features: [batch, 128]

        # Process transition features
        # Extract transition IDs (every 3rd element starting from 0)
        trans_ids = transitions[:, ::3].long()  # [batch, 19]
        trans_ids = torch.clamp(trans_ids, 0, self.transition_embedding.num_embeddings - 1)

        # Embed transitions
        trans_embedded = self.transition_embedding(trans_ids)  # [batch, 19, 16]

        # Extract position weights and suspicious flags
        trans_positions = transitions[:, 1::3]  # [batch, 19]
        trans_suspicious = transitions[:, 2::3]  # [batch, 19]

        # Combine transition information
        trans_features = torch.cat([
            trans_embedded.mean(dim=1),  # [batch, 16]
            trans_positions.mean(dim=1, keepdim=True),  # [batch, 1]
            trans_suspicious.sum(dim=1, keepdim=True),  # [batch, 1] - count suspicious
            trans_suspicious.mean(dim=1, keepdim=True),  # [batch, 1] - ratio suspicious
        ], dim=1)  # [batch, 19]

        # Pad to 32 dimensions
        trans_features = torch.cat([
            trans_features,
            torch.zeros(batch_size, 32 - trans_features.size(1), device=trans_features.device)
        ], dim=1)

        # Combine all Layer 2 features
        l2_input = torch.cat([lstm_features, trans_features, seq_stats], dim=1)
        l2_out = self.l2_processor(l2_input)

        # Fusion
        combined = torch.cat([l1_out, l2_out], dim=1)
        output = self.fusion(combined)

        return torch.sigmoid(output)

    def get_attention_weights(self, layer2_sequences):
        """
        Extract attention weights for interpretability.
        Useful for understanding which positions the model focuses on.
        """
        with torch.no_grad():
            func_seq = layer2_sequences[:, :20].long()
            positions = layer2_sequences[:, 20:40]

            func_seq = torch.clamp(func_seq, 0, self.embedding.num_embeddings - 1)
            func_embedded = self.embedding(func_seq)
            pos_encoded = self.position_encoder(positions.unsqueeze(-1))
            combined_embedded = torch.cat([func_embedded, pos_encoded], dim=-1)

            mask = (positions != 0).float()
            lstm_out, _ = self.lstm(combined_embedded)
            _, attention_weights = self.attention_pooling(lstm_out, mask)

            return attention_weights.squeeze(-1)  # [batch, seq_len]