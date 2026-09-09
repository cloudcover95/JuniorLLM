"""MCP server + client — raw JSON-RPC 2.0, no SDK."""
from __future__ import annotations

import json
from typing import Any, Callable


class McpError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class McpServer:
    def __init__(self) -> None:
        self.methods: dict[str, Callable[..., Any]] = {}
        self.tools: dict[str, str] = {}

    def tool(self, name: str, desc: str, fn: Callable[..., Any]) -> None:
        self.tools[name] = desc
        self.methods[name] = fn
        self.methods.setdefault("tools/list", self._list)

    def _list(self) -> dict:
        return {"tools": [{"name": n, "description": d} for n, d in self.tools.items()]}

    def handle(self, raw: str | dict) -> dict:
        req = json.loads(raw) if isinstance(raw, str) else raw
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        if method not in self.methods:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown {method}"}}
        try:
            if isinstance(params, list):
                result = self.methods[method](*params)
            else:
                result = self.methods[method](**params) if params else self.methods[method]()
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(exc)}}


class McpClient:
    def __init__(self, server: McpServer):
        self.server = server
        self._n = 0

    def call(self, method: str, params: dict | list | None = None) -> Any:
        self._n += 1
        resp = self.server.handle({"jsonrpc": "2.0", "id": self._n, "method": method, "params": params or {}}
        )
        if "error" in resp:
            raise McpError(resp["error"]["code"], resp["error"]["message"])
        return resp["result"]
