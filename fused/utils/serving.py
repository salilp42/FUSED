"""
Model serving utilities for FUSED.

This module provides tools for exporting and serving FUSED models
in production environments.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
import logging


class ModelExporter:
    """
    Exporter for FUSED models.
    
    This class provides methods for exporting models to various formats,
    including PyTorch, ONNX, and TorchScript.
    """
    
    def __init__(self, model: nn.Module, save_dir: str = "exported_models"):
        """
        Initialize the model exporter.
        
        Args:
            model: FUSED model to export
            save_dir: Directory to save exported models
        """
        self.model = model
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def export_pytorch(self, 
                      filename: str = "model.pt",
                      include_config: bool = True) -> str:
        """
        Export model in PyTorch format.
        
        Args:
            filename: Name of the exported file
            include_config: Whether to include model configuration
            
        Returns:
            Path to the exported model
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Prepare state dict and config
        state_dict = self.model.state_dict()
        save_dict = {"state_dict": state_dict}
        
        # Include configuration if requested
        if include_config and hasattr(self.model, "config"):
            save_dict["config"] = self.model.config
        
        # Save model
        save_path = os.path.join(self.save_dir, filename)
        torch.save(save_dict, save_path)
        
        logging.info(f"PyTorch model exported to {save_path}")
        
        return save_path
    
    def export_torchscript(self, 
                          filename: str = "model.pt",
                          example_inputs: Optional[Dict[str, torch.Tensor]] = None,
                          method: str = "trace") -> str:
        """
        Export model in TorchScript format.
        
        Args:
            filename: Name of the exported file
            example_inputs: Example inputs for tracing
            method: Export method ('trace' or 'script')
            
        Returns:
            Path to the exported model
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Prepare save path
        save_path = os.path.join(self.save_dir, filename)
        
        # Export using specified method
        if method.lower() == "trace":
            if example_inputs is None:
                raise ValueError("Example inputs must be provided for tracing")
                
            # Apply input transformation if model has it
            if hasattr(self.model, "transform_inputs"):
                example_inputs = self.model.transform_inputs(example_inputs)
                
            # Trace the model
            traced_model = torch.jit.trace(self.model, example_inputs=example_inputs.values())
            torch.jit.save(traced_model, save_path)
            
        elif method.lower() == "script":
            # Script the model
            scripted_model = torch.jit.script(self.model)
            torch.jit.save(scripted_model, save_path)
            
        else:
            raise ValueError(f"Unknown export method: {method}. Use 'trace' or 'script'.")
        
        logging.info(f"TorchScript model exported to {save_path}")
        
        return save_path
    
    def export_onnx(self, 
                   filename: str = "model.onnx",
                   example_inputs: Optional[Dict[str, torch.Tensor]] = None,
                   input_names: Optional[List[str]] = None,
                   output_names: Optional[List[str]] = None,
                   dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                   opset_version: int = 11) -> str:
        """
        Export model in ONNX format.
        
        Args:
            filename: Name of the exported file
            example_inputs: Example inputs for ONNX export
            input_names: Names of input tensors
            output_names: Names of output tensors
            dynamic_axes: Dictionary defining dynamic axes
            opset_version: ONNX opset version
            
        Returns:
            Path to the exported model
        """
        try:
            import onnx
        except ImportError:
            warnings.warn("ONNX not installed. Install with: pip install onnx")
        
        try:
            import onnxruntime
        except ImportError:
            warnings.warn("ONNX Runtime not installed. Install with: pip install onnxruntime")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Check if example inputs are provided
        if example_inputs is None:
            raise ValueError("Example inputs must be provided for ONNX export")
            
        # Apply input transformation if model has it
        if hasattr(self.model, "transform_inputs"):
            example_inputs = self.model.transform_inputs(example_inputs)
            
        # Prepare input arguments
        if input_names is None:
            input_names = list(example_inputs.keys())
            
        if output_names is None:
            output_names = ["output"]
            
        # Prepare dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {}
            for input_name in input_names:
                dynamic_axes[input_name] = {0: "batch_size"}
            for output_name in output_names:
                dynamic_axes[output_name] = {0: "batch_size"}
        
        # Prepare save path
        save_path = os.path.join(self.save_dir, filename)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            tuple(example_inputs.values()),
            save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        # Check the model
        try:
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            logging.info(f"ONNX model checked: {onnx.helper.printable_graph(onnx_model.graph)}")
        except Exception as e:
            warnings.warn(f"ONNX model checking failed: {e}")
        
        logging.info(f"ONNX model exported to {save_path}")
        
        return save_path
    
    def export_config(self, filename: str = "config.json") -> str:
        """
        Export model configuration to JSON.
        
        Args:
            filename: Name of the exported file
            
        Returns:
            Path to the exported configuration
        """
        # Check if model has configuration
        if not hasattr(self.model, "config"):
            warnings.warn("Model does not have a 'config' attribute")
            return None
            
        # Prepare save path
        save_path = os.path.join(self.save_dir, filename)
        
        # Convert non-serializable objects to strings
        def convert_config(obj):
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [convert_config(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_config(value) for key, value in obj.items()}
            else:
                return str(obj)
        
        # Convert configuration
        config_dict = convert_config(self.model.config)
        
        # Save configuration
        with open(save_path, "w") as f:
            json.dump(config_dict, f, indent=2)
            
        logging.info(f"Model configuration exported to {save_path}")
        
        return save_path
    
    def export_all(self, 
                  base_filename: str = "model",
                  example_inputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, str]:
        """
        Export model in all formats.
        
        Args:
            base_filename: Base name for exported files
            example_inputs: Example inputs for export
            
        Returns:
            Dictionary mapping format names to file paths
        """
        # Prepare result dictionary
        result = {}
        
        # Export PyTorch model
        pt_filename = f"{base_filename}.pt"
        result["pytorch"] = self.export_pytorch(filename=pt_filename)
        
        # Export configuration
        config_filename = f"{base_filename}_config.json"
        config_path = self.export_config(filename=config_filename)
        if config_path:
            result["config"] = config_path
        
        # Export TorchScript model if example inputs are provided
        if example_inputs is not None:
            ts_filename = f"{base_filename}_torchscript.pt"
            result["torchscript"] = self.export_torchscript(
                filename=ts_filename,
                example_inputs=example_inputs
            )
            
            # Export ONNX model
            onnx_filename = f"{base_filename}.onnx"
            try:
                result["onnx"] = self.export_onnx(
                    filename=onnx_filename,
                    example_inputs=example_inputs
                )
            except Exception as e:
                warnings.warn(f"ONNX export failed: {e}")
        
        return result


class ModelServer:
    """
    Server for FUSED models.
    
    This class provides methods for serving FUSED models via HTTP
    or loading them for fast inference.
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the model server.
        
        Args:
            model_path: Path to the saved model
            device: Device to load the model on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device
        
        # Load the model
        self.model = self._load_model()
        
    def _load_model(self) -> nn.Module:
        """
        Load the model from the specified path.
        
        Returns:
            Loaded model
        """
        # Check file extension
        if self.model_path.endswith(".pt") or self.model_path.endswith(".pth"):
            # Load PyTorch model
            model_data = torch.load(self.model_path, map_location=self.device)
            
            # Check if it's a state dict only or a full save
            if "state_dict" in model_data:
                # Full save with state dict and possibly config
                state_dict = model_data["state_dict"]
                
                # If config is included, use it to initialize the model
                if "config" in model_data:
                    # Dynamically import the FUSED model
                    from fused.models.model import FUSEDModel
                    model = FUSEDModel(model_data["config"])
                    model.load_state_dict(state_dict)
                else:
                    # Try to find model class and initialize it
                    warnings.warn("No model configuration found. Using a dummy model.")
                    model = torch.nn.Module()
                    model.load_state_dict(state_dict)
            else:
                # Assume it's a state dict only
                model = torch.nn.Module()
                model.load_state_dict(model_data)
                
        elif self.model_path.endswith(".onnx"):
            # Return a wrapper for ONNX model
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(self.model_path)
                
                # Create a wrapper class
                class ONNXWrapper(nn.Module):
                    def __init__(self, session):
                        super().__init__()
                        self.session = session
                        self.input_names = [input.name for input in session.get_inputs()]
                        self.output_names = [output.name for output in session.get_outputs()]
                        
                    def forward(self, *args, **kwargs):
                        # Convert args and kwargs to a dict for ONNX
                        if len(args) > 0:
                            # Assume args match input_names order
                            inputs = {name: arg for name, arg in zip(self.input_names, args)}
                        else:
                            inputs = kwargs
                            
                        # Convert PyTorch tensors to numpy
                        for name, tensor in inputs.items():
                            if isinstance(tensor, torch.Tensor):
                                inputs[name] = tensor.detach().cpu().numpy()
                                
                        # Run inference
                        outputs = self.session.run(self.output_names, inputs)
                        
                        # If there's only one output, return it directly
                        if len(outputs) == 1:
                            return torch.tensor(outputs[0])
                        
                        # Otherwise, return a dictionary of outputs
                        return {name: torch.tensor(output) for name, output in zip(self.output_names, outputs)}
                
                model = ONNXWrapper(session)
                
            except ImportError:
                raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime")
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")
        
        # Move model to device and set to evaluation mode
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Make predictions using the loaded model.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of model outputs
        """
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Convert outputs to dictionary if not already
        if not isinstance(outputs, dict):
            outputs = {"output": outputs}
            
        return outputs
    
    def start_http_server(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Start HTTP server for model serving.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise ImportError("Flask not installed. Install with: pip install flask")
            
        app = Flask(__name__)
        
        @app.route("/predict", methods=["POST"])
        def predict():
            # Get JSON data from request
            data = request.json
            
            # Convert inputs to tensors
            inputs = {}
            for key, value in data.items():
                # Handle different input types
                if isinstance(value, list):
                    if all(isinstance(item, list) for item in value):
                        # 2D or higher list
                        inputs[key] = torch.tensor(value)
                    else:
                        # 1D list
                        inputs[key] = torch.tensor(value)
                else:
                    # Scalar
                    inputs[key] = torch.tensor([value])
            
            # Make prediction
            outputs = self.predict(inputs)
            
            # Convert outputs to JSON-serializable format
            result = {}
            for key, tensor in outputs.items():
                if isinstance(tensor, torch.Tensor):
                    # Convert tensor to list
                    result[key] = tensor.detach().cpu().numpy().tolist()
                else:
                    # Keep as is
                    result[key] = tensor
            
            return jsonify(result)
        
        # Start the server
        app.run(host=host, port=port)
        
    def start_grpc_server(self, host: str = "0.0.0.0", port: int = 50051):
        """
        Start gRPC server for model serving.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        try:
            import grpc
            from concurrent import futures
        except ImportError:
            raise ImportError("gRPC not installed. Install with: pip install grpcio grpcio-tools")
            
        # This is just a placeholder since implementing a full gRPC server
        # would require defining protobuf messages and services.
        warnings.warn("gRPC server implementation requires custom protobuf definitions.")
        logging.info(f"gRPC server would start on {host}:{port}")
        
        # Actual implementation would include:
        # 1. Define protobuf messages and services
        # 2. Generate Python code from protobuf
        # 3. Implement service methods
        # 4. Start server with those methods
        
        # Example pseudo-code:
        # server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        # add_servicer_to_server(self, server)
        # server.add_insecure_port(f'{host}:{port}')
        # server.start()
        # server.wait_for_termination()


def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a FUSED model from a saved file.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded model
    """
    return ModelServer(model_path, device).model
