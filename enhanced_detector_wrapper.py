"""
Enhanced Pre-Execution Detector for AgentDojo

Wraps the enhanced detector model with position-aware features and attention pooling
for use in AgentDojo benchmark.
"""

import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections.abc import Sequence

# Add paths
sys.path.insert(0, '/Users/yummy/IPI_Defense/scripts')
sys.path.insert(0, '/Users/yummy/IPI_Defense/agentdojo/src')

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

# Import enhanced model classes
from enhanced_detector_models import EnhancedFeatureExtractor, EnhancedIPIDetector


class EnhancedPreExecutionDetector(BasePipelineElement):
    """
    Enhanced pre-execution detector with position-aware features and attention.

    This detector checks tool calls BEFORE execution, using:
    1. Position-aware temporal features
    2. Attention-based pooling for LSTM
    3. Transition pattern analysis
    """

    def __init__(
        self,
        model_path: str = "/Users/yummy/IPI_Defense/models/enhanced_detector.pkl",
        threshold: float = 0.5,
        raise_on_injection: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the enhanced pre-execution detector.

        Args:
            model_path: Path to the enhanced model file
            threshold: Detection threshold (0-1)
            raise_on_injection: Whether to abort on detection
            debug: Enable debug logging
        """
        super().__init__()
        self.threshold = threshold
        self.raise_on_injection = raise_on_injection
        self.debug = debug

        # Load the enhanced model
        if self.debug:
            print(f"[EnhancedPreExec] Loading enhanced detector from: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Check model type
        if model_data.get('model_type') != 'enhanced':
            print(f"[EnhancedPreExec] Warning: Model type is '{model_data.get('model_type')}', expected 'enhanced'")

        self.feature_extractor = model_data['feature_extractor']

        # Reconstruct the enhanced model
        layer1_dim = model_data['model_config']['layer1_dim']
        layer2_dim = model_data['model_config']['layer2_dim']
        vocab_size = model_data['model_config']['vocab_size']
        embedding_dim = model_data['model_config'].get('embedding_dim', 32)
        hidden_dim = model_data['model_config'].get('hidden_dim', 64)

        self.model = EnhancedIPIDetector(
            layer1_dim, layer2_dim,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()

        if self.debug:
            print(f"[EnhancedPreExec] ✓ Enhanced detector loaded (threshold: {threshold})")
            print(f"[EnhancedPreExec]   Layer 1 dim: {layer1_dim}, Layer 2 dim: {layer2_dim}")
            print(f"[EnhancedPreExec]   Vocab size: {vocab_size}")

        # Statistics
        self.stats = {
            'total_checks': 0,
            'detections': 0,
            'aborts': 0,
        }

    def extract_tool_calls_from_latest_message(
        self, messages: Sequence[ChatMessage]
    ) -> List[Dict]:
        """Extract tool calls from the LATEST assistant message."""
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                tool_calls = []
                for tool_call in message["tool_calls"]:
                    if hasattr(tool_call, 'function'):
                        func_name = tool_call.function.name if hasattr(tool_call.function, 'name') else str(tool_call.function)
                        args = tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else {}
                    else:
                        func_name = tool_call.get('function', {}).get('name', '')
                        args = tool_call.get('function', {}).get('arguments', {})

                    if isinstance(args, str):
                        try:
                            import json
                            args = json.loads(args)
                        except:
                            args = {}

                    tool_calls.append({
                        'function': func_name,
                        'args': args
                    })

                return tool_calls

        return []

    def predict_attack_probability(
        self, tool_calls: List[Dict], suite_name: str = "unknown"
    ) -> float:
        """
        Predict attack probability using enhanced features.

        Args:
            tool_calls: List of tool call dictionaries
            suite_name: Name of the suite

        Returns:
            Probability of attack (0-1)
        """
        if not tool_calls:
            return 0.0

        # Convert to sequence format
        sequence = [
            {
                'function_name': tc.get('function', ''),
                'parameters': tc.get('args', {})
            }
            for tc in tool_calls
        ]

        # Extract enhanced features from all string parameters
        layer1_features = []
        layer2_features = []

        for tool_call in tool_calls:
            function_name = tool_call.get('function', '')
            args = tool_call.get('args', {})

            if not isinstance(args, dict):
                continue

            for param_name, param_value in args.items():
                if isinstance(param_value, str) and param_value.strip() and len(param_value.strip()) >= 3:
                    # Extract Layer 1 features
                    l1_feat = self.feature_extractor.extract_layer1_features(
                        function_name, param_value.strip(), suite_name
                    )

                    # Extract enhanced Layer 2 features (with position-aware info)
                    l2_feat = self.feature_extractor.extract_layer2_features(sequence)

                    layer1_features.append(l1_feat)
                    layer2_features.append(l2_feat)

        if not layer1_features:
            return 0.0

        # Convert to arrays and scale
        layer1_features = np.array(layer1_features)
        layer2_features = np.array(layer2_features)

        layer1_features = self.feature_extractor.layer1_scaler.transform(layer1_features)
        layer2_features = self.feature_extractor.layer2_scaler.transform(layer2_features)

        # Convert to tensors
        test_l1 = torch.FloatTensor(layer1_features)
        test_l2 = torch.FloatTensor(layer2_features)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(test_l1, test_l2).squeeze()
            if outputs.dim() == 0:
                probabilities = outputs.item()
            else:
                probabilities = outputs.numpy()

        # Aggregate: take maximum probability
        if isinstance(probabilities, np.ndarray):
            attack_prob = float(np.max(probabilities))
        else:
            attack_prob = float(probabilities)

        return attack_prob

    def get_attention_visualization(
        self, tool_calls: List[Dict], suite_name: str = "unknown"
    ) -> Optional[np.ndarray]:
        """
        Get attention weights for interpretability.
        Returns attention weights over sequence positions.
        """
        if not tool_calls:
            return None

        sequence = [
            {
                'function_name': tc.get('function', ''),
                'parameters': tc.get('args', {})
            }
            for tc in tool_calls
        ]

        # Extract Layer 2 features
        l2_feat = self.feature_extractor.extract_layer2_features(sequence)
        l2_feat_scaled = self.feature_extractor.layer2_scaler.transform([l2_feat])

        # Convert to tensor
        test_l2 = torch.FloatTensor(l2_feat_scaled)

        # Get attention weights
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(test_l2)
            return attention_weights.numpy()

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        """
        AgentDojo pipeline query method.

        This runs BEFORE ToolsExecutor, checking tool calls before execution.
        """

        self.stats['total_checks'] += 1

        # Extract tool calls from latest message
        tool_calls = self.extract_tool_calls_from_latest_message(messages)

        if not tool_calls:
            if self.debug:
                print(f"[EnhancedPreExec] No tool calls in latest message")
            return query, runtime, env, messages, extra_args

        # Extract suite name
        suite_name = extra_args.get('suite_name', 'unknown')
        if hasattr(env, 'suite_name'):
            suite_name = env.suite_name

        # Detect injection using enhanced features
        attack_prob = self.predict_attack_probability(tool_calls, suite_name)
        is_injection = attack_prob > self.threshold

        # Store detection results
        extra_args['enhanced_detector_prob'] = attack_prob
        extra_args['enhanced_is_injection'] = is_injection

        if self.debug or is_injection:
            print(f"[EnhancedPreExec] Checked {len(tool_calls)} tool calls")
            print(f"[EnhancedPreExec] Attack probability: {attack_prob:.4f}")
            print(f"[EnhancedPreExec] Is injection: {is_injection} (threshold: {self.threshold})")
            if is_injection:
                print(f"[EnhancedPreExec] Tool calls: {[tc['function'] for tc in tool_calls]}")

                # Get attention visualization if debug
                if self.debug:
                    attention = self.get_attention_visualization(tool_calls, suite_name)
                    if attention is not None:
                        print(f"[EnhancedPreExec] Attention weights: {attention[0][:len(tool_calls)]}")

        if is_injection:
            self.stats['detections'] += 1

        if self.raise_on_injection and is_injection:
            self.stats['aborts'] += 1

            # Mark that we blocked an attack
            extra_args['enhanced_blocked_attack'] = True

            if self.debug:
                print(f"[EnhancedPreExec] 🛑 BLOCKING execution (prob: {attack_prob:.4f})")

            # Raise error to abort execution
            raise AbortAgentError(
                f"Enhanced detector: Prompt injection detected (confidence: {attack_prob:.4f}, threshold: {self.threshold})",
                messages,
                env,
            )

        return query, runtime, env, messages, extra_args

    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return {
            **self.stats,
            'detection_rate': (self.stats['detections'] / self.stats['total_checks'] * 100)
            if self.stats['total_checks'] > 0 else 0,
        }


def load_enhanced_detector(
    model_path: str = "/Users/yummy/IPI_Defense/models/enhanced_detector.pkl",
    threshold: float = 0.5,
    raise_on_injection: bool = True,
    debug: bool = False,
) -> EnhancedPreExecutionDetector:
    """
    Load and return a configured enhanced detector.

    Args:
        model_path: Path to the enhanced model file
        threshold: Detection threshold (0-1)
        raise_on_injection: Whether to abort on detection
        debug: Enable debug logging

    Returns:
        Configured EnhancedPreExecutionDetector instance
    """
    return EnhancedPreExecutionDetector(
        model_path=model_path,
        threshold=threshold,
        raise_on_injection=raise_on_injection,
        debug=debug,
    )


if __name__ == "__main__":
    # Test loading the enhanced detector
    print("Testing enhanced detector loading...")
    try:
        detector = load_enhanced_detector(debug=True)
        print("✓ Enhanced detector loaded successfully!")
        print("\nInitial stats:", detector.get_stats())
    except FileNotFoundError:
        print("⚠️  Enhanced model not found. Train it first with:")
        print("  python scripts/train_enhanced_detector.py")