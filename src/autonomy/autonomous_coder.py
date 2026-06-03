# path: JuniorLLM/src/autonomy/autonomous_coder.py
#!/usr/bin/env python3
"""
Autonomous Coder

Gives agents the ability to autonomously generate, review, and propose
code changes in isolated environments (containers).

This is a key step toward agents building and improving their own codebase.
"""

import logging
from typing import Any, Dict, Optional

from .code_generator import CodeGenerator

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class AutonomousCoder:
    """
    Allows an agent to act as an autonomous developer.
    """

    def __init__(self, code_generator: Optional[CodeGenerator] = None):
        self.code_generator = code_generator or CodeGenerator()
        logging.info("AutonomousCoder initialized")

    def propose_code_change(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        generation = self.code_generator.generate_code(task, context=context)
        review = self.code_generator.review_code(generation["code"])

        return {
            "proposal": generation,
            "review": review,
            "approved_for_execution": review.get("safe", False),
        }

    def execute_in_container(self, code: str, container_manager=None) -> Dict[str, Any]:
        # Placeholder: In real implementation, this would use DockerManager
        # from JuniorHome to run the code in an isolated container.
        logging.info("Would execute generated code in isolated container (sandboxed)")
        return {
            "executed": False,
            "reason": "Container execution not yet wired (use JuniorHome DockerManager)",
            "code": code,
        }
